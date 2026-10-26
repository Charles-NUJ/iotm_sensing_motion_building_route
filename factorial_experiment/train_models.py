#!/usr/bin/env python3
"""Train scene, speed, and joint link-persistence models from ns-3 traces."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

try:
    from .common import (
        REPOSITORY,
        artifact_root,
        band_for_speed,
        condition_id,
        conditions,
        load_config,
    )
except ImportError:
    from common import (
        REPOSITORY,
        artifact_root,
        band_for_speed,
        condition_id,
        conditions,
        load_config,
    )


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    family: np.ndarray
    band: np.ndarray
    speed: np.ndarray
    seed: np.ndarray

    def select(self, mask: np.ndarray) -> "Samples":
        return Samples(
            self.x[mask],
            self.y[mask],
            self.family[mask],
            self.band[mask],
            self.speed[mask],
            self.seed[mask],
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symmetric_features(
    first_position: np.ndarray,
    first_velocity: np.ndarray,
    second_position: np.ndarray,
    second_velocity: np.ndarray,
    communication_range_m: float,
    horizon_s: float,
) -> np.ndarray:
    relative_position = second_position - first_position
    relative_velocity = second_velocity - first_velocity
    distance = float(np.linalg.norm(relative_position))
    relative_speed = float(np.linalg.norm(relative_velocity))
    radial = (
        float(np.dot(relative_position, relative_velocity) / distance)
        if distance > 1e-12
        else 0.0
    )
    tangential = math.sqrt(max(0.0, relative_speed**2 - radial**2))
    aa = float(np.dot(relative_velocity, relative_velocity))
    bb = 2.0 * float(np.dot(relative_position, relative_velocity))
    cc = distance**2 - communication_range_m**2
    lifetime = 0.0
    if cc <= 1e-12:
        if aa <= 1e-18:
            lifetime = math.inf
        else:
            discriminant = bb * bb - 4.0 * aa * cc
            if -1e-8 < discriminant < 0.0:
                discriminant = 0.0
            if discriminant >= 0.0:
                lifetime = max(
                    0.0, (-bb + math.sqrt(discriminant)) / (2.0 * aa)
                )
    return np.asarray(
        [
            min(1.5, distance / communication_range_m),
            min(2.0, relative_speed / 25.0),
            float(np.clip(radial / 25.0, -2.0, 2.0)),
            min(2.0, tangential / 25.0),
            1.0 if math.isinf(lifetime) else min(1.0, lifetime / horizon_s),
        ],
        dtype=np.float32,
    )


def first_exit_fraction(
    future_distances: np.ndarray,
    communication_range_m: float,
    sample_period_s: float,
    horizon_s: float,
) -> float:
    exits = np.flatnonzero(future_distances > communication_range_m)
    if not len(exits):
        return 1.0
    exit_index = int(exits[0])
    if exit_index == 0:
        return 0.0
    inside = float(future_distances[exit_index - 1])
    outside = float(future_distances[exit_index])
    interpolation = (
        (communication_range_m - inside) / (outside - inside)
        if outside > inside
        else 0.0
    )
    holding_s = (
        exit_index - 1 + float(np.clip(interpolation, 0.0, 1.0))
    ) * sample_period_s
    return float(min(1.0, holding_s / horizon_s))


def read_trace(path: Path, nodes: int) -> np.ndarray:
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ["time_s", "node", "x", "y", "z", "vx", "vy", "vz"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected trace header: {path}")
        for index, row in enumerate(reader):
            if int(row["node"]) != index % nodes:
                raise ValueError(f"node order differs at row {index + 2}: {path}")
            values.append(
                [float(row[name]) for name in ("x", "y", "z", "vx", "vy", "vz")]
            )
    array = np.asarray(values, dtype=np.float64)
    if len(array) % nodes:
        raise ValueError(f"trace length is not divisible by node count: {path}")
    return array.reshape(-1, nodes, 6)


def trace_samples(
    path: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = int(config["nodes"])
    state = read_trace(path, nodes)
    dt = float(config["mobility_sample_period_s"])
    horizon_s = float(config["link_hold_horizon_s"])
    range_m = float(config["communication_range_m"])
    horizon_steps = int(round(horizon_s / dt))
    stride = int(round(float(config["training"]["sample_period_s"]) / dt))
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_speed: list[float] = []
    for step in range(0, len(state) - horizon_steps, stride):
        current = state[step]
        for first in range(nodes):
            for second in range(first + 1, nodes):
                distance = float(
                    np.linalg.norm(current[first, :3] - current[second, :3])
                )
                if distance > range_m:
                    continue
                future_distances = np.linalg.norm(
                    state[
                        step : step + horizon_steps + 1, first, :3
                    ]
                    - state[
                        step : step + horizon_steps + 1, second, :3
                    ],
                    axis=1,
                )
                rows_x.append(
                    symmetric_features(
                        current[first, :3],
                        current[first, 3:],
                        current[second, :3],
                        current[second, 3:],
                        range_m,
                        horizon_s,
                    )
                )
                rows_y.append(
                    first_exit_fraction(
                        future_distances, range_m, dt, horizon_s
                    )
                )
                rows_speed.append(
                    float(
                        np.linalg.norm(
                            current[first, 3:] - current[second, 3:]
                        )
                    )
                )
    return (
        np.asarray(rows_x, dtype=np.float32),
        np.asarray(rows_y, dtype=np.float32),
        np.asarray(rows_speed, dtype=np.float32),
    )


def collect_split(
    split: str, output_root: Path, config: dict[str, Any]
) -> Samples:
    families = list(config["families"])
    speeds = [float(value) for value in config["absolute_speed_m_s"]]
    bands = [str(row["key"]) for row in config["relative_speed_bands"]]
    cap = int(config["training"]["maximum_samples_per_family_band_seed"])
    sampling_seed = int(config["training"]["sampling_seed"])
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("x", "y", "family", "band", "speed", "seed")
    }
    for condition in conditions(config):
        family_index = families.index(str(condition["family"]))
        speed_index = speeds.index(float(condition["absolute_speed_m_s"]))
        for seed in config["seeds"][split]:
            trace = (
                output_root
                / "mobility"
                / split
                / f"{split}_{condition_id(condition)}_s{int(seed)}.mobility.csv"
            )
            if not trace.is_file():
                raise FileNotFoundError(trace)
            x, y, relative_speed = trace_samples(trace, config)
            assigned = np.asarray(
                [
                    bands.index(key) if key in bands else -1
                    for key in (
                        band_for_speed(float(value), config)
                        for value in relative_speed
                    )
                ],
                dtype=np.int8,
            )
            for band_index in range(len(bands)):
                available = np.flatnonzero(assigned == band_index)
                if len(available) > cap:
                    sequence = np.random.SeedSequence(
                        [sampling_seed, family_index, band_index, int(seed)]
                    )
                    generator = np.random.default_rng(sequence)
                    available = np.sort(
                        generator.choice(available, cap, replace=False)
                    )
                if not len(available):
                    continue
                parts["x"].append(x[available])
                parts["y"].append(y[available])
                parts["family"].append(
                    np.full(len(available), family_index, dtype=np.int8)
                )
                parts["band"].append(
                    np.full(len(available), band_index, dtype=np.int8)
                )
                parts["speed"].append(
                    np.full(len(available), speed_index, dtype=np.int8)
                )
                parts["seed"].append(
                    np.full(len(available), int(seed), dtype=np.int32)
                )
    if not parts["y"]:
        raise RuntimeError(f"no registered link samples found for {split}")
    return Samples(**{key: np.concatenate(value) for key, value in parts.items()})


def make_model():
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(5, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid(),
    )


def load_base_state(path: Path):
    import torch

    tokens = path.read_text(encoding="ascii").split()
    header = tuple(int(value) for value in tokens[:4])
    if header != (1, 5, 16, 8):
        raise ValueError(f"unexpected model header: {header}")
    values = np.asarray([float(value) for value in tokens[4:]], dtype=np.float32)
    if len(values) != 241:
        raise ValueError(f"expected 241 model parameters, found {len(values)}")
    cursor = 0

    def take(count: int, shape: tuple[int, ...]):
        nonlocal cursor
        result = torch.from_numpy(values[cursor : cursor + count].copy())
        cursor += count
        return result.reshape(shape)

    return {
        "0.weight": take(80, (16, 5)),
        "0.bias": take(16, (16,)),
        "2.weight": take(128, (8, 16)),
        "2.bias": take(8, (8,)),
        "4.weight": take(8, (1, 8)),
        "4.bias": take(1, (1,)),
    }


def train_one(
    training: Samples,
    validation: Samples,
    groups_training: np.ndarray,
    groups_validation: np.ndarray,
    base_state,
    config: dict[str, Any],
    seed: int,
):
    import torch

    torch.manual_seed(seed)
    model = make_model()
    model.load_state_dict(copy.deepcopy(base_state))
    tx = torch.from_numpy(training.x)
    ty = torch.from_numpy(training.y)
    vx = torch.from_numpy(validation.x)
    vy = torch.from_numpy(validation.y)
    tg = torch.from_numpy(groups_training.astype(np.int64))
    vg = torch.from_numpy(groups_validation.astype(np.int64))
    unique, counts = torch.unique(tg, return_counts=True)
    count_by_group = {
        int(group): int(count) for group, count in zip(unique, counts)
    }
    weights = torch.tensor(
        [1.0 / count_by_group[int(group)] for group in tg.tolist()],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True, generator=generator
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tx, ty),
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )

    def balanced_validation_mse() -> float:
        model.eval()
        with torch.no_grad():
            prediction = model(vx).squeeze(1)
            values = [
                torch.mean((prediction[vg == group] - vy[vg == group]) ** 2)
                for group in torch.unique(vg)
            ]
        return float(torch.mean(torch.stack(values)))

    best_score = math.inf
    best_epoch = 0
    stale = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, int(config["training"]["maximum_epochs"]) + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = torch.mean((model(batch_x).squeeze(1) - batch_y) ** 2)
            loss.backward()
            optimizer.step()
        score = balanced_validation_mse()
        if score < best_score - 1e-10:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(config["training"]["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "balanced_validation_mse": best_score,
        "training_samples": len(training.y),
        "validation_samples": len(validation.y),
    }


def export_model(model, path: Path) -> None:
    state = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "1 5 16 8\n"
    for name in (
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    ):
        text += (
            " ".join(
                format(float(value), ".17g")
                for value in state[name].detach().cpu().numpy().ravel()
            )
            + "\n"
        )
    path.write_text(text, encoding="ascii")


def deploy_aliases(
    trained: dict[str, Path], output_root: Path, config: dict[str, Any]
) -> None:
    families = [str(value) for value in config["families"]]
    bands = [str(row["key"]) for row in config["relative_speed_bands"]]
    for scope in ("scene", "speed", "joint"):
        directory = output_root / "deploy" / scope
        directory.mkdir(parents=True, exist_ok=True)
        for family in families:
            for band in bands:
                if scope == "scene":
                    source = trained[f"scene/{family}"]
                elif scope == "speed":
                    source = trained[f"speed/{band}"]
                else:
                    source = trained[f"joint/{family}/{band}"]
                shutil.copyfile(
                    source, directory / f"family_{family}_{band}.dat"
                )
    frozen = output_root / "deploy" / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        REPOSITORY / str(config["training"]["base_model"]),
        frozen / "frozen.dat",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = artifact_root(args.output_root)
    families = [str(value) for value in config["families"]]
    bands = [str(row["key"]) for row in config["relative_speed_bands"]]
    plan = {
        "scene_models": families,
        "speed_models": bands,
        "joint_models": [f"{family}/{band}" for family in families for band in bands],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    import torch

    torch.set_num_threads(1)
    training = collect_split("training", output_root, config)
    validation = collect_split("validation", output_root, config)
    base_path = REPOSITORY / str(config["training"]["base_model"])
    base_state = load_base_state(base_path)
    trained: dict[str, Path] = {}
    receipt: dict[str, Any] = {
        "base_model": str(base_path.relative_to(REPOSITORY)).replace("\\", "/"),
        "base_model_sha256": sha256(base_path),
        "models": {},
    }
    model_seed = 20262000

    for family_index, family in enumerate(families):
        train_mask = training.family == family_index
        validation_mask = validation.family == family_index
        model, metrics = train_one(
            training.select(train_mask),
            validation.select(validation_mask),
            training.band[train_mask],
            validation.band[validation_mask],
            base_state,
            config,
            model_seed + family_index,
        )
        path = output_root / "models" / "scene" / f"{family}.dat"
        export_model(model, path)
        trained[f"scene/{family}"] = path
        receipt["models"][f"scene/{family}"] = {
            **metrics,
            "sha256": sha256(path),
        }

    for band_index, band in enumerate(bands):
        train_mask = training.band == band_index
        validation_mask = validation.band == band_index
        model, metrics = train_one(
            training.select(train_mask),
            validation.select(validation_mask),
            training.family[train_mask],
            validation.family[validation_mask],
            base_state,
            config,
            model_seed + 100 + band_index,
        )
        path = output_root / "models" / "speed" / f"{band}.dat"
        export_model(model, path)
        trained[f"speed/{band}"] = path
        receipt["models"][f"speed/{band}"] = {
            **metrics,
            "sha256": sha256(path),
        }

    for family_index, family in enumerate(families):
        for band_index, band in enumerate(bands):
            train_mask = (training.family == family_index) & (
                training.band == band_index
            )
            validation_mask = (validation.family == family_index) & (
                validation.band == band_index
            )
            model, metrics = train_one(
                training.select(train_mask),
                validation.select(validation_mask),
                training.speed[train_mask],
                validation.speed[validation_mask],
                base_state,
                config,
                model_seed + 200 + 10 * family_index + band_index,
            )
            path = (
                output_root
                / "models"
                / "joint"
                / f"{family}_{band}.dat"
            )
            export_model(model, path)
            trained[f"joint/{family}/{band}"] = path
            receipt["models"][f"joint/{family}/{band}"] = {
                **metrics,
                "sha256": sha256(path),
            }

    deploy_aliases(trained, output_root, config)
    receipt_path = output_root / "training_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trained_models": len(receipt["models"])}))


if __name__ == "__main__":
    main()
