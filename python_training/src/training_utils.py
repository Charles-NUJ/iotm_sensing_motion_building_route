"""Training and TFLite export helpers for the link-hold MLPs."""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Mapping

import numpy as np


def _tensorflow():
    """Import TensorFlow lazily so data-only tests stay lightweight."""

    import tensorflow as tf

    return tf


def train_with_early_stopping(
    model,
    train_inputs,
    train_labels: np.ndarray,
    validation_inputs,
    validation_labels: np.ndarray,
    maximum_epochs: int,
    batch_size: int,
    patience: int,
):
    """训练模型并返回 Keras History 和实际训练秒数。"""

    if maximum_epochs <= 0 or batch_size <= 0 or patience < 0:
        raise ValueError("训练轮数、批量大小必须为正，patience 不能为负")
    tf = _tensorflow()
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        min_delta=1e-5,
    )
    start = time.perf_counter()
    history = model.fit(
        train_inputs,
        train_labels,
        validation_data=(validation_inputs, validation_labels),
        epochs=maximum_epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=2,
        shuffle=True,
    )
    return history, time.perf_counter() - start


def predict_probabilities(model, model_inputs, batch_size: int = 2048) -> np.ndarray:
    """批量预测并返回一维概率数组。"""

    return np.asarray(model.predict(model_inputs, batch_size=batch_size, verbose=0)).reshape(-1)


def probability_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """计算 MAE、Brier、对数损失和阈值分类准确率。"""

    prediction = np.clip(np.asarray(predictions, dtype=float), 1e-7, 1.0 - 1e-7)
    target = np.asarray(targets, dtype=float)
    if prediction.shape != target.shape:
        raise ValueError("predictions 和 targets 形状必须一致")
    error = prediction - target
    log_loss = -np.mean(target * np.log(prediction) + (1.0 - target) * np.log(1.0 - prediction))
    threshold_accuracy = np.mean((prediction >= 0.5) == (target >= 0.5))
    return {
        "mae": float(np.mean(np.abs(error))),
        "brier": float(np.mean(error**2)),
        "log_loss": float(log_loss),
        "threshold_accuracy": float(threshold_accuracy),
    }


def save_history(history, output_path: Path, seed: int, model_name: str) -> None:
    """把每轮训练和验证指标保存为可追溯 CSV。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.history)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["seed", "model", "epoch", *keys],
        )
        writer.writeheader()
        for epoch in range(len(history.history[keys[0]])):
            row: dict[str, object] = {"seed": seed, "model": model_name, "epoch": epoch + 1}
            row.update({key: history.history[key][epoch] for key in keys})
            writer.writerow(row)


def save_model_and_tflite(
    model,
    model_directory: Path,
    *,
    optimize: bool = True,
) -> tuple[Path, Path]:
    """保存 TensorFlow SavedModel 与面向端侧部署的 TFLite 文件。"""

    tf = _tensorflow()
    model_directory.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_directory, include_optimizer=False)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_directory))
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path = model_directory.parent / f"{model_directory.name}.tflite"
    tflite_path.write_bytes(tflite_bytes)
    return model_directory, tflite_path


def tflite_predict(tflite_path: Path, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """用 TFLite 单批前向，验证中心训练后的本地部署模型。"""

    tf = _tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    requested_batch = next(iter(inputs.values())).shape[0]
    for detail in input_details:
        shape = list(detail["shape"])
        shape[0] = requested_batch
        interpreter.resize_tensor_input(detail["index"], shape, strict=False)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    for detail in input_details:
        tensor_name = detail["name"].split(":", 1)[0]
        matching_key = next(
            key for key in inputs if key in tensor_name or tensor_name.endswith(key)
        )
        interpreter.set_tensor(detail["index"], np.asarray(inputs[matching_key], dtype=np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return np.asarray(output).reshape(-1)


def benchmark_tflite_single_link(
    tflite_path: Path,
    inputs: Mapping[str, np.ndarray],
    warmup_runs: int = 100,
    measured_runs: int = 2000,
) -> dict[str, float]:
    """测量复用解释器时单条链路的 TFLite CPU 前向延迟。

    参数：
        tflite_path: 已导出的 TFLite 模型路径。
        inputs: 至少包含一个样本的五类局部模型输入。
        warmup_runs: 不计入统计的预热次数。
        measured_runs: 正式计时次数。

    返回微秒单位的均值、中位数和 95 分位数。统计包含五个输入张量写入和
    一次解释器调用，但不包含模型加载、解释器创建和内存分配。
    """

    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs 不能为负，measured_runs 必须大于 0")
    sample_count = next(iter(inputs.values())).shape[0]
    if sample_count <= 0 or any(value.shape[0] != sample_count for value in inputs.values()):
        raise ValueError("全部输入必须具有相同且非零的样本数")
    tf = _tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    input_details = interpreter.get_input_details()
    for detail in input_details:
        shape = list(detail["shape"])
        shape[0] = 1
        interpreter.resize_tensor_input(detail["index"], shape, strict=False)
    interpreter.allocate_tensors()
    bindings = []
    for detail in interpreter.get_input_details():
        tensor_name = detail["name"].split(":", 1)[0]
        matching_key = next(
            key for key in inputs if key in tensor_name or tensor_name.endswith(key)
        )
        bindings.append((detail["index"], matching_key))

    def invoke(sample_index: int) -> None:
        """写入一条链路输入并执行一次前向。"""

        for tensor_index, input_name in bindings:
            value = np.asarray(inputs[input_name][sample_index : sample_index + 1], dtype=np.float32)
            interpreter.set_tensor(tensor_index, value)
        interpreter.invoke()

    for run in range(warmup_runs):
        invoke(run % sample_count)
    latencies = np.empty(measured_runs, dtype=float)
    for run in range(measured_runs):
        start = time.perf_counter_ns()
        invoke(run % sample_count)
        latencies[run] = (time.perf_counter_ns() - start) / 1000.0
    return {
        "mean_microseconds": float(np.mean(latencies)),
        "median_microseconds": float(np.median(latencies)),
        "p95_microseconds": float(np.quantile(latencies, 0.95)),
        "maximum_microseconds": float(np.max(latencies)),
        "warmup_runs": float(warmup_runs),
        "measured_runs": float(measured_runs),
    }


def parameter_count(model) -> int:
    """返回模型可训练和不可训练参数总数。"""

    return int(model.count_params())
