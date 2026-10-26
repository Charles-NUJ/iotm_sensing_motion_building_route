"""Shared configuration and task identities for the public workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
DEFAULT_CONFIG = HERE / "config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    target = DEFAULT_CONFIG if path is None else Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def number_token(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def condition_id(condition: dict[str, Any]) -> str:
    stem = (
        f"{condition['family']}_"
        f"v{number_token(float(condition['absolute_speed_m_s']))}"
    )
    target = condition.get("formation_relative_speed_target_m_s")
    if target is not None:
        stem += f"_r{number_token(float(target))}"
    return stem


def conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    speeds = [float(value) for value in config["absolute_speed_m_s"]]
    rows: list[dict[str, Any]] = []
    for family in ("waypoint", "gauss_markov"):
        for speed in speeds:
            rows.append(
                {
                    "family": family,
                    "absolute_speed_m_s": speed,
                    "formation_relative_speed_target_m_s": None,
                }
            )

    formation = config["formation"]
    rows.append(
        {
            "family": "formation",
            "absolute_speed_m_s": 0.0,
            "formation_relative_speed_target_m_s": float(
                formation["zero_speed_target_m_s"]
            ),
        }
    )
    for speed in speeds:
        if speed <= 0:
            continue
        for band in config["relative_speed_bands"]:
            target = float(
                formation["relative_speed_target_m_s"][str(band["key"])]
            )
            if target <= 2.0 * speed + 1e-12:
                rows.append(
                    {
                        "family": "formation",
                        "absolute_speed_m_s": speed,
                        "formation_relative_speed_target_m_s": target,
                    }
                )

    identifiers = [condition_id(row) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("condition identifiers must be unique")
    return rows


def band_for_speed(relative_speed_m_s: float, config: dict[str, Any]) -> str | None:
    matches = [
        str(band["key"])
        for band in config["relative_speed_bands"]
        if float(band["lower_m_s"]) <= relative_speed_m_s
        <= float(band["upper_m_s"])
    ]
    if len(matches) > 1:
        raise ValueError("relative-speed bands overlap")
    return matches[0] if matches else None


def formation_arguments(
    condition: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    target = condition.get("formation_relative_speed_target_m_s")
    if target is None:
        return []
    formation = config["formation"]
    return [
        f"--formationRelativeSpeedTarget={float(target):g}",
        f"--formationExcitationPeriod={float(formation['excitation_period_s']):g}",
        (
            "--formationExcitationSharpness="
            f"{float(formation['excitation_sharpness']):g}"
        ),
    ]


def artifact_root(path: Path | None = None) -> Path:
    return (
        REPOSITORY / "factorial_experiment" / "artifacts"
        if path is None
        else Path(path).resolve()
    )
