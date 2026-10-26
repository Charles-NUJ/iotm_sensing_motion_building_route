#!/usr/bin/env python3
"""Run paired Hop, LET, Frozen, and fine-tuned routing variants."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

try:
    from .common import (
        REPOSITORY,
        artifact_root,
        band_for_speed,
        condition_id,
        conditions,
        formation_arguments,
        load_config,
    )
except ImportError:
    from common import (
        REPOSITORY,
        artifact_root,
        band_for_speed,
        condition_id,
        conditions,
        formation_arguments,
        load_config,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_relative_speed_p95(
    trace: Path, nodes: int, communication_range_m: float
) -> float:
    rows: list[list[float]] = []
    with trace.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if int(row["node"]) != index % nodes:
                raise ValueError(f"node order differs at row {index + 2}: {trace}")
            rows.append(
                [float(row[name]) for name in ("x", "y", "z", "vx", "vy", "vz")]
            )
    state = np.asarray(rows, dtype=float).reshape(-1, nodes, 6)
    values: list[float] = []
    for frame in state:
        for first in range(nodes):
            for second in range(first + 1, nodes):
                if (
                    np.linalg.norm(frame[first, :3] - frame[second, :3])
                    <= communication_range_m
                ):
                    values.append(
                        float(
                            np.linalg.norm(
                                frame[first, 3:] - frame[second, 3:]
                            )
                        )
                    )
    if not values:
        raise ValueError(f"trace has no connected endpoint pair: {trace}")
    return float(np.percentile(np.asarray(values), 95))


def assignment_rows(
    output_root: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions(config):
        for seed in config["seeds"]["evaluation"]:
            identifier = f"evaluation_{condition_id(condition)}_s{int(seed)}"
            trace = (
                output_root / "mobility" / "evaluation" / f"{identifier}.mobility.csv"
            )
            if not trace.is_file():
                raise FileNotFoundError(trace)
            statistic = measured_relative_speed_p95(
                trace,
                int(config["nodes"]),
                float(config["communication_range_m"]),
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "identifier": identifier,
                    "relative_speed_p95_m_s": statistic,
                    "relative_speed_band": band_for_speed(statistic, config),
                }
            )
    return rows


def command_for(
    assignment: dict[str, Any],
    variant: str,
    binary: Path,
    output_root: Path,
    config: dict[str, Any],
) -> tuple[list[str], Path]:
    condition = assignment["condition"]
    band = assignment["relative_speed_band"]
    prefix = (
        output_root
        / "evaluation"
        / f"{assignment['identifier']}_{variant}"
    )
    command = [
        str(binary),
        f"--mission={condition['family']}",
        f"--speed={float(condition['absolute_speed_m_s']):g}",
        f"--nodes={int(config['nodes'])}",
        f"--seed={int(assignment['seed'])}",
        "--estimator=kf",
        "--summaryOnly=1",
        f"--output={prefix}",
        *formation_arguments(condition, config),
    ]
    if variant == "hop":
        command.append("--method=hop")
    elif variant == "let":
        command.append("--method=analytical")
    elif variant == "frozen":
        command.extend(
            [
                "--method=frozen",
                f"--input={output_root / 'deploy' / 'frozen'}",
            ]
        )
    elif variant in ("scene_ft", "speed_ft", "joint_ft"):
        if band is None:
            raise ValueError("fine-tuned method requires a registered speed band")
        directory = {
            "scene_ft": "scene",
            "speed_ft": "speed",
            "joint_ft": "joint",
        }[variant]
        command.extend(
            [
                "--method=family_adapted",
                f"--adaptationBand={band}",
                f"--input={output_root / 'deploy' / directory}",
            ]
        )
    else:
        raise ValueError(f"unknown method: {variant}")
    return command, prefix


def run_one(
    assignment: dict[str, Any],
    variant: str,
    binary: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    command, prefix = command_for(
        assignment, variant, binary, output_root, config
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs" / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_stem = f"{assignment['identifier']}_{variant}"
    stdout_path = log_dir / f"{log_stem}.stdout.txt"
    stderr_path = log_dir / f"{log_stem}.stderr.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, timeout=240, check=False
        )
    summary = prefix.with_suffix(".json")
    record: dict[str, Any] = {
        **assignment,
        "variant": variant,
        "command": command,
        "exit_code": completed.returncode,
        "valid": completed.returncode == 0 and summary.is_file(),
    }
    if record["valid"]:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        record.update(
            {
                "summary": str(summary.relative_to(output_root)).replace("\\", "/"),
                "summary_sha256": sha256(summary),
                "metrics": {
                    "pdr": float(payload["pdr"]),
                    "tracking_rmse_m": float(payload["tracking_rmse_m"]),
                    "minimum_delivered_hops": payload[
                        "minimum_delivered_hops"
                    ],
                },
            }
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = artifact_root(args.output_root)
    binary = (
        args.binary.resolve()
        if args.binary
        else (REPOSITORY / str(config["binary"])).resolve()
    )
    assignments = assignment_rows(output_root, config)
    registered = [
        row for row in assignments if row["relative_speed_band"] is not None
    ]
    tasks = [
        (assignment, variant)
        for assignment in registered
        for variant in config["evaluation"]["methods"]
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    assignment_path = output_root / "evaluation_assignments.json"
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(
        json.dumps(assignments, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        commands = [
            command_for(assignment, variant, binary, output_root, config)[0]
            for assignment, variant in tasks
        ]
        print(
            json.dumps(
                {
                    "registered_tasks": len(registered),
                    "protocol_runs": len(tasks),
                    "commands": commands,
                },
                indent=2,
            )
        )
        return
    if not binary.is_file():
        raise SystemExit(f"ns-3 executable is missing: {binary}")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                assignment,
                variant,
                binary,
                output_root,
                config,
            ): (assignment, variant)
            for assignment, variant in tasks
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: (str(row["identifier"]), str(row["variant"])))
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registered_tasks": len(registered),
        "protocol_runs": len(tasks),
        "successful_runs": sum(bool(row["valid"]) for row in records),
        "records": records,
    }
    receipt_path = output_root / "evaluation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "registered_tasks",
                    "protocol_runs",
                    "successful_runs",
                )
            }
        )
    )
    if receipt["successful_runs"] != receipt["protocol_runs"]:
        raise SystemExit("one or more protocol tasks failed")


if __name__ == "__main__":
    main()
