#!/usr/bin/env python3
"""Create paired summaries and the two-panel relative-gain figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import artifact_root, load_config
except ImportError:
    from common import artifact_root, load_config


METHODS = (
    ("frozen", "Frozen"),
    ("scene_ft", "Scene FT"),
    ("speed_ft", "Speed FT"),
    ("joint_ft", "Scene-speed FT"),
    ("let", "LET"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_paired_tasks(
    output_root: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    receipt_path = output_root / "evaluation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_methods = set(config["evaluation"]["methods"])
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for record in receipt["records"]:
        if not record.get("valid"):
            continue
        indexed.setdefault(str(record["identifier"]), {})[
            str(record["variant"])
        ] = record
    rows: list[dict[str, Any]] = []
    minimum_hops = int(config["evaluation"]["minimum_hop_count"])
    for identifier, variants in indexed.items():
        if set(variants) != expected_methods:
            continue
        hop_hops = variants["hop"]["metrics"]["minimum_delivered_hops"]
        if hop_hops is None or int(hop_hops) < minimum_hops:
            continue
        reference_pdr = float(variants["hop"]["metrics"]["pdr"])
        reference_rmse = float(variants["hop"]["metrics"]["tracking_rmse_m"])
        sample = variants["hop"]
        band = str(sample["relative_speed_band"])
        display_groups = config["evaluation"]["display_groups"]
        display = next(
            (
                name
                for name, members in display_groups.items()
                if band in members
            ),
            None,
        )
        if display is None:
            continue
        row: dict[str, Any] = {
            "identifier": identifier,
            "family": str(sample["condition"]["family"]),
            "absolute_speed_m_s": float(
                sample["condition"]["absolute_speed_m_s"]
            ),
            "relative_speed_p95_m_s": float(
                sample["relative_speed_p95_m_s"]
            ),
            "training_relative_speed_band": band,
            "display_relative_speed_group": display,
            "seed": int(sample["seed"]),
            "hop_minimum_delivered_hops": int(hop_hops),
            "hop_pdr": reference_pdr,
            "hop_tracking_rmse_m": reference_rmse,
        }
        for method, _ in METHODS:
            metrics = variants[method]["metrics"]
            pdr = float(metrics["pdr"])
            rmse = float(metrics["tracking_rmse_m"])
            row[f"{method}_pdr"] = pdr
            row[f"{method}_tracking_rmse_m"] = rmse
            row[f"{method}_pdr_gain_over_hop_pp"] = 100.0 * (
                pdr - reference_pdr
            )
            row[f"{method}_rmse_reduction_over_hop_m"] = (
                reference_rmse - rmse
            )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["display_relative_speed_group"]),
            int(row["seed"]),
            str(row["identifier"]),
        )
    )
    return rows


def seed_level(
    tasks: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in config["families"]:
        for display_group in config["evaluation"]["display_groups"]:
            for seed in config["seeds"]["evaluation"]:
                selected = [
                    row
                    for row in tasks
                    if row["family"] == family
                    and row["display_relative_speed_group"] == display_group
                    and int(row["seed"]) == int(seed)
                ]
                if not selected:
                    continue
                record: dict[str, Any] = {
                    "family": family,
                    "display_relative_speed_group": display_group,
                    "seed": int(seed),
                    "paired_tasks": len(selected),
                    "absolute_speed_min_m_s": min(
                        float(row["absolute_speed_m_s"]) for row in selected
                    ),
                    "absolute_speed_max_m_s": max(
                        float(row["absolute_speed_m_s"]) for row in selected
                    ),
                    "relative_speed_min_m_s": min(
                        float(row["relative_speed_p95_m_s"]) for row in selected
                    ),
                    "relative_speed_max_m_s": max(
                        float(row["relative_speed_p95_m_s"]) for row in selected
                    ),
                }
                for method, _ in METHODS:
                    for metric in (
                        "pdr_gain_over_hop_pp",
                        "rmse_reduction_over_hop_m",
                    ):
                        record[f"{method}_{metric}"] = float(
                            np.mean(
                                [
                                    float(row[f"{method}_{metric}"])
                                    for row in selected
                                ]
                            )
                        )
                rows.append(record)
    return rows


def cell_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities = sorted(
        {
            (
                str(row["family"]),
                str(row["display_relative_speed_group"]),
            )
            for row in seed_rows
        }
    )
    for family, display_group in identities:
        selected = [
            row
            for row in seed_rows
            if row["family"] == family
            and row["display_relative_speed_group"] == display_group
        ]
        for method, label in METHODS:
            record: dict[str, Any] = {
                "family": family,
                "display_relative_speed_group": display_group,
                "method": method,
                "method_label": label,
                "whole_mission_seeds": len(selected),
            }
            for metric in (
                "pdr_gain_over_hop_pp",
                "rmse_reduction_over_hop_m",
            ):
                values = np.asarray(
                    [float(row[f"{method}_{metric}"]) for row in selected],
                    dtype=float,
                )
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_median"] = float(np.median(values))
                record[f"{metric}_minimum"] = float(np.min(values))
                record[f"{metric}_maximum"] = float(np.max(values))
            rows.append(record)
    return rows


def draw(seed_rows: list[dict[str, Any]], output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.4,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(3.55, 5.0))
    family_order = ("waypoint", "gauss_markov", "formation")
    family_labels = {
        "waypoint": "Waypoint",
        "gauss_markov": "Gauss-Markov",
        "formation": "Formation",
    }
    groups = (("lower", "#d7d7d7", ""), ("high", "#555555", "//"))
    legend = [
        Patch(facecolor=color, edgecolor="#111111", hatch=hatch, label=group.title())
        for group, color, hatch in groups
    ]
    for axis in axes:
        axis.axhline(0, color="#555555", linewidth=0.7, linestyle="--")
        axis.grid(axis="y", color="#e2e2e2", linewidth=0.45)
        axis.legend(
            handles=legend,
            loc="best",
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#777777",
            handlelength=1.6,
            ncol=1,
        )

    positions: list[float] = []
    labels: list[str] = []
    for family_index, family in enumerate(family_order):
        start = family_index * 6.3
        for method_index, (method, label) in enumerate(METHODS):
            center = start + 0.7 + 1.15 * method_index
            positions.append(center)
            labels.append(label)
            for group_index, (group, color, hatch) in enumerate(groups):
                selected = [
                    row
                    for row in seed_rows
                    if row["family"] == family
                    and row["display_relative_speed_group"] == group
                ]
                if len(selected) < 2:
                    continue
                x = center + (-0.16 if group_index == 0 else 0.16)
                distributions = (
                    [
                        float(row[f"{method}_pdr_gain_over_hop_pp"])
                        for row in selected
                    ],
                    [
                        float(row[f"{method}_rmse_reduction_over_hop_m"])
                        for row in selected
                    ],
                )
                for axis, values in zip(axes, distributions):
                    values_array = np.asarray(values)
                    if float(np.ptp(values_array)) < 1e-12:
                        axis.plot(
                            [x - 0.12, x + 0.12],
                            [values_array[0], values_array[0]],
                            color="#111111",
                            linewidth=0.9,
                        )
                    else:
                        violin = axis.violinplot(
                            [values_array],
                            positions=[x],
                            widths=0.27,
                            showmeans=True,
                            showextrema=False,
                        )
                        body = violin["bodies"][0]
                        body.set_facecolor(color)
                        body.set_edgecolor("#111111")
                        body.set_alpha(0.95)
                        body.set_linewidth(0.6)
                        body.set_hatch(hatch)
                        violin["cmeans"].set_color("#111111")
                        violin["cmeans"].set_linewidth(0.7)
        for axis in axes:
            axis.text(
                start + 3.0,
                1.015,
                family_labels[family],
                ha="center",
                transform=axis.get_xaxis_transform(),
            )
            if family_index < len(family_order) - 1:
                axis.axvline(start + 6.0, color="#c8c8c8", linewidth=0.6)

    axes[0].set_ylabel("PDR gain over Hop (percentage points)")
    axes[1].set_ylabel("RMSE reduction over Hop (m)")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", labelsize=6.2, pad=2)
        for tick in axis.get_xticklabels():
            tick.set_rotation(32)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
    axes[0].text(0.5, -0.29, "(a)", ha="center", transform=axes[0].transAxes)
    axes[1].text(0.5, -0.29, "(b)", ha="center", transform=axes[1].transAxes)
    figure.subplots_adjust(
        left=0.18, right=0.99, top=0.96, bottom=0.075, hspace=0.46
    )
    figure.savefig(
        output_root / "figure5_relative_gain.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output_root / "figure5_relative_gain.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = artifact_root(args.output_root)
    tasks = load_paired_tasks(output_root, config)
    seeds = seed_level(tasks, config)
    summary = cell_summary(seeds)
    write_csv(output_root / "paired_tasks.csv", tasks)
    write_csv(output_root / "seed_level_results.csv", seeds)
    write_csv(output_root / "cell_summary.csv", summary)
    draw(seeds, output_root)
    record = {
        "paired_tasks": len(tasks),
        "seed_level_rows": len(seeds),
        "minimum_hop_count": int(config["evaluation"]["minimum_hop_count"]),
        "methods": [method for method, _ in METHODS],
    }
    (output_root / "analysis_receipt.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
