#!/usr/bin/env python3
"""Train and export the compact relative-motion link-hold model."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import Ridge
import tensorflow as tf

from src.flight_graph_data import load_flight_routes
from src.training_utils import (
    benchmark_tflite_single_link,
    parameter_count,
    predict_probabilities,
    save_history,
    save_model_and_tflite,
    tflite_predict,
    train_with_early_stopping,
)
from src.link_hold_model import (
    build_link_hold_default_bundle,
    build_link_hold_mlp,
    build_rich_link_hold_mlp,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    errors = np.asarray(predictions) - np.asarray(targets)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "persistence_accuracy": float(
            np.mean((predictions >= 0.5) == (targets >= 0.5))
        ),
    }


def competing_link_ranking_metrics(
    predictions: np.ndarray,
    dataset,
    *,
    minimum_target_gap: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate local ordering among links incident on the same node.

    A target gap of 0.05 corresponds to half a second of the 10-s holding
    horizon.  Smaller differences are treated as operational ties.
    """

    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    targets = np.asarray(dataset.target, dtype=float).reshape(-1)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (time_index, first, second) in enumerate(
        zip(dataset.time, dataset.first_node, dataset.second_node)
    ):
        groups.setdefault((int(time_index), int(first)), []).append(index)
        groups.setdefault((int(time_index), int(second)), []).append(index)

    correct_pairs = 0.0
    pair_count = 0
    top1_correct = 0
    top1_count = 0
    top1_regrets: list[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for offset, first_index in enumerate(indices[:-1]):
            for second_index in indices[offset + 1 :]:
                target_delta = (
                    targets[first_index] - targets[second_index]
                )
                if abs(target_delta) < minimum_target_gap:
                    continue
                prediction_delta = (
                    predictions[first_index] - predictions[second_index]
                )
                pair_count += 1
                if prediction_delta == 0.0:
                    correct_pairs += 0.5
                elif prediction_delta * target_delta > 0.0:
                    correct_pairs += 1.0

        ordered = sorted(
            indices,
            key=lambda index: targets[index],
            reverse=True,
        )
        if (
            targets[ordered[0]] - targets[ordered[1]]
            < minimum_target_gap
        ):
            continue
        selected = max(indices, key=lambda index: predictions[index])
        top1_count += 1
        top1_correct += int(selected == ordered[0])
        top1_regrets.append(
            float(targets[ordered[0]] - targets[selected])
        )

    return {
        "ranking_minimum_target_gap": minimum_target_gap,
        "competing_edge_pairs": pair_count,
        "competing_edge_pairwise_accuracy": (
            correct_pairs / pair_count if pair_count else math.nan
        ),
        "clear_best_neighbor_groups": top1_count,
        "clear_best_neighbor_accuracy": (
            top1_correct / top1_count if top1_count else math.nan
        ),
        "mean_selected_hold_regret": (
            float(np.mean(top1_regrets)) if top1_regrets else math.nan
        ),
    }


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    flight_directory = Path(
        os.environ.get(
            "P2_FLIGHTROUTES_DIR",
            str(ROOT / "data" / "flightroutes"),
        )
    ).resolve()
    if not flight_directory.is_dir():
        raise FileNotFoundError(
            "Flight Routes dataset not found. Clone "
            "https://github.com/Charles-NUJ/flightroutes at commit "
            "81a09f48 and set P2_FLIGHTROUTES_DIR to that checkout."
        )
    seeds = [
        int(value)
        for value in os.environ.get(
            "P2_LINK_HOLD_SEEDS",
            "20260721,20260722,20260723",
        ).split(",")
        if value.strip()
    ]
    routes = load_flight_routes(flight_directory, maximum_routes=100)
    bundle = build_link_hold_default_bundle(routes)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    deployment_rows: list[dict[str, object]] = []
    total_start = time.perf_counter()

    for seed in seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_link_hold_mlp()
        history, training_seconds = train_with_early_stopping(
            model,
            bundle.train.inputs,
            bundle.train.target,
            bundle.validation.inputs,
            bundle.validation.target,
            maximum_epochs=60,
            batch_size=1024,
            patience=8,
        )
        model_path = MODELS / f"link_hold_seed_{seed}"
        _, tflite_path = save_model_and_tflite(
            model,
            model_path,
            optimize=False,
        )
        save_history(
            history,
            RESULTS / f"training_history_link_hold_seed_{seed}.csv",
            seed,
            "relative_motion_link_hold_mlp",
        )
        rich_model = build_rich_link_hold_mlp()
        rich_history, rich_training_seconds = train_with_early_stopping(
            rich_model,
            bundle.train.rich_inputs,
            bundle.train.target,
            bundle.validation.rich_inputs,
            bundle.validation.target,
            maximum_epochs=60,
            batch_size=1024,
            patience=8,
        )
        rich_model_path = MODELS / f"rich_link_hold_seed_{seed}"
        _, rich_tflite_path = save_model_and_tflite(
            rich_model,
            rich_model_path,
            optimize=False,
        )
        save_history(
            rich_history,
            RESULTS / f"training_history_rich_link_hold_seed_{seed}.csv",
            seed,
            "neighbor_context_link_hold_mlp",
        )
        ridge = Ridge(alpha=1.0)
        ridge.fit(bundle.train.inputs, bundle.train.target)
        joblib.dump(
            ridge,
            MODELS / f"link_hold_ridge_seed_{seed}.joblib",
        )

        for split_name, dataset in (
            ("train", bundle.train),
            ("validation", bundle.validation),
            ("test", bundle.test),
        ):
            predictions = predict_probabilities(model, dataset.inputs)
            rich_predictions = predict_probabilities(
                rich_model,
                dataset.rich_inputs,
            )
            baselines = {
                "analytic_constant_velocity": dataset.inputs[:, 4],
                "same_input_ridge": np.clip(
                    ridge.predict(dataset.inputs),
                    0.0,
                    1.0,
                ),
                "relative_motion_link_hold_mlp": predictions,
                "neighbor_context_link_hold_mlp": rich_predictions,
            }
            for model_name, values in baselines.items():
                if model_name == "relative_motion_link_hold_mlp":
                    parameters = parameter_count(model)
                    seconds = training_seconds
                    epochs = len(history.history["loss"])
                elif model_name == "neighbor_context_link_hold_mlp":
                    parameters = parameter_count(rich_model)
                    seconds = rich_training_seconds
                    epochs = len(rich_history.history["loss"])
                else:
                    parameters = 0
                    seconds = 0.0
                    epochs = 0
                metric_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "split": split_name,
                        "samples": len(dataset),
                        "parameters": parameters,
                        "training_seconds": seconds,
                        "epochs": epochs,
                        **regression_metrics(values, dataset.target),
                        **competing_link_ranking_metrics(
                            values,
                            dataset,
                        ),
                    }
                )

        sample = bundle.test.inputs[:128]
        tensorflow_predictions = predict_probabilities(model, sample)
        tflite_predictions = tflite_predict(
            tflite_path,
            {"link_hold_features": sample},
        )
        latency = benchmark_tflite_single_link(
            tflite_path,
            {"link_hold_features": sample},
            warmup_runs=100,
            measured_runs=2000,
        )
        rich_sample = bundle.test.rich_inputs[:128]
        rich_tensorflow_predictions = predict_probabilities(
            rich_model,
            rich_sample,
        )
        rich_tflite_predictions = tflite_predict(
            rich_tflite_path,
            {"rich_link_hold_features": rich_sample},
        )
        rich_latency = benchmark_tflite_single_link(
            rich_tflite_path,
            {"rich_link_hold_features": rich_sample},
            warmup_runs=100,
            measured_runs=2000,
        )
        deployment_rows.append(
            {
                "seed": seed,
                "relative_parameters": parameter_count(model),
                "relative_tflite_bytes": tflite_path.stat().st_size,
                "relative_training_seconds": training_seconds,
                "relative_epochs": len(history.history["loss"]),
                "relative_tflite_max_absolute_error": float(
                    np.max(
                        np.abs(
                            tensorflow_predictions - tflite_predictions
                        )
                    )
                ),
                "relative_mean_microseconds": latency["mean_microseconds"],
                "relative_median_microseconds": latency[
                    "median_microseconds"
                ],
                "relative_p95_microseconds": latency["p95_microseconds"],
                "rich_parameters": parameter_count(rich_model),
                "rich_tflite_bytes": rich_tflite_path.stat().st_size,
                "rich_training_seconds": rich_training_seconds,
                "rich_epochs": len(rich_history.history["loss"]),
                "rich_tflite_max_absolute_error": float(
                    np.max(
                        np.abs(
                            rich_tensorflow_predictions
                            - rich_tflite_predictions
                        )
                    )
                ),
                "rich_mean_microseconds": rich_latency["mean_microseconds"],
                "rich_median_microseconds": rich_latency[
                    "median_microseconds"
                ],
                "rich_p95_microseconds": rich_latency["p95_microseconds"],
                "warmup_runs": latency["warmup_runs"],
                "measured_runs": latency["measured_runs"],
            }
        )

    write_csv(RESULTS / "link_hold_prediction_metrics.csv", metric_rows)
    write_csv(RESULTS / "link_hold_deployment_metrics.csv", deployment_rows)
    output = {
        "scope": (
            "matched five-input and rich-neighbor endpoint-symmetric "
            "prediction of observed relative-motion link hold quality"
        ),
        "flight_routes_repository": (
            "https://github.com/Charles-NUJ/flightroutes"
        ),
        "flight_routes_commit": "81a09f48",
        "model_seeds": seeds,
        "traffic_horizon_s": 10,
        "communication_range_m": 200,
        "training_nodes": 60,
        "validation_nodes": 80,
        "test_nodes": 100,
        "training_times": [0, 171, 2],
        "validation_times": [181, 231, 2],
        "test_times": [241, 291, 1],
        "target_window_s": 10,
        "temporal_embargoes": [[171, 180], [231, 240]],
        "split_rule": (
            "each sample's complete future target window remains inside "
            "its temporal split"
        ),
        "total_training_seconds": time.perf_counter() - total_start,
        "metrics": metric_rows,
        "deployment": deployment_rows,
        "boundaries": [
            "the target is future connectivity in recorded trajectories",
            "the model does not predict a next hop or full route",
            "constant-velocity analytical lifetime remains a fallback",
            "queue length is intentionally excluded from both models",
            "the rich comparator adds endpoint and one-hop neighbor summaries",
        ],
    }
    with (
        RESULTS / "link_hold_model_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

