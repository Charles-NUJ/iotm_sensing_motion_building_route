"""Compare one native ns-3 summary with the archived reference run."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    actual = json.loads(args.result.read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "examples/formation_n24_s4501_motion.reference.json").read_text(encoding="utf-8"))
    for key in ("engine", "method", "mission", "seed", "nodes", "offered", "delivered", "minimum_delivered_hops"):
        print(f"{key}: actual={actual[key]!r}, archived={expected[key]!r}")
    for key in ("pdr", "tracking_rmse_m", "mean_delay_s", "estimated_retry_fraction"):
        print(f"{key}: actual={actual[key]:.12g}, archived={expected[key]:.12g}, difference={actual[key]-expected[key]:+.3g}")
    same = all(actual[k] == expected[k] for k in ("engine", "method", "mission", "seed", "nodes", "offered", "delivered", "minimum_delivered_hops"))
    same &= all(abs(actual[k] - expected[k]) <= 1e-8 for k in ("pdr", "tracking_rmse_m", "mean_delay_s", "estimated_retry_fraction"))
    print("REFERENCE_MATCH" if same else "REFERENCE_DIFFERENCE (check build and environment)")
    return 0 if same else 1

if __name__ == "__main__":
    raise SystemExit(main())
