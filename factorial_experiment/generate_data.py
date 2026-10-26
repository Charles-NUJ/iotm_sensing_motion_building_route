#!/usr/bin/env python3
"""Generate native ns-3 mobility traces for disjoint seed groups."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

try:
    from .common import (
        REPOSITORY,
        artifact_root,
        condition_id,
        conditions,
        formation_arguments,
        load_config,
    )
except ImportError:
    from common import (
        REPOSITORY,
        artifact_root,
        condition_id,
        conditions,
        formation_arguments,
        load_config,
    )


@dataclass(frozen=True)
class MobilityTask:
    split: str
    condition: dict[str, Any]
    seed: int

    @property
    def identifier(self) -> str:
        return f"{self.split}_{condition_id(self.condition)}_s{self.seed}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_matrix(config: dict[str, Any], splits: list[str]) -> list[MobilityTask]:
    return [
        MobilityTask(split, condition, int(seed))
        for split in splits
        for condition in conditions(config)
        for seed in config["seeds"][split]
    ]


def command_for(
    task: MobilityTask,
    binary: Path,
    output_root: Path,
    config: dict[str, Any],
) -> tuple[list[str], Path]:
    prefix = output_root / "mobility" / task.split / task.identifier
    command = [
        str(binary),
        "--mobilityOnly=1",
        f"--mission={task.condition['family']}",
        f"--speed={float(task.condition['absolute_speed_m_s']):g}",
        f"--nodes={int(config['nodes'])}",
        f"--seed={task.seed}",
        "--method=hop",
        "--summaryOnly=1",
        f"--output={prefix}",
        *formation_arguments(task.condition, config),
    ]
    return command, prefix


def run_task(
    task: MobilityTask,
    binary: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    command, prefix = command_for(task, binary, output_root, config)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs" / "mobility"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{task.identifier}.stdout.txt"
    stderr_path = log_dir / f"{task.identifier}.stderr.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            command, stdout=stdout, stderr=stderr, timeout=240, check=False
        )
    trace = prefix.with_suffix(".mobility.csv")
    summary = prefix.with_suffix(".json")
    record: dict[str, Any] = {
        "split": task.split,
        "condition": task.condition,
        "seed": task.seed,
        "identifier": task.identifier,
        "command": command,
        "exit_code": completed.returncode,
        "valid": completed.returncode == 0 and trace.is_file() and summary.is_file(),
    }
    if record["valid"]:
        record.update(
            {
                "trace": str(trace.relative_to(output_root)).replace("\\", "/"),
                "trace_sha256": sha256(trace),
                "summary": str(summary.relative_to(output_root)).replace("\\", "/"),
                "summary_sha256": sha256(summary),
            }
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("training", "validation", "evaluation", "all"),
        default="all",
    )
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
    splits = (
        ["training", "validation", "evaluation"]
        if args.split == "all"
        else [args.split]
    )
    tasks = task_matrix(config, splits)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if args.dry_run:
        plan = [
            command_for(task, binary, output_root, config)[0] for task in tasks
        ]
        print(json.dumps({"tasks": len(tasks), "commands": plan}, indent=2))
        return
    if not binary.is_file():
        raise SystemExit(f"ns-3 executable is missing: {binary}")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_task, task, binary, output_root, config): task
            for task in tasks
        }
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda row: str(row["identifier"]))
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": len(tasks),
        "successful_tasks": sum(bool(row["valid"]) for row in records),
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "mobility_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("tasks", "successful_tasks")}))
    if receipt["successful_tasks"] != receipt["tasks"]:
        raise SystemExit("one or more native mobility tasks failed")


if __name__ == "__main__":
    main()
