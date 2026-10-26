#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
binary="$repo_dir/build/p2_native_kf15"
test -x "$binary" || { echo "Build the programs first; see README.md." >&2; exit 1; }
mkdir -p "$repo_dir/outputs"
"$binary" --method=motion --mission=formation --nodes=24 --seed=4601 \
  --estimator=kf --input="$repo_dir/inputs" \
  --output="$repo_dir/outputs/formation_n24_s4601_motion_kf" \
  --summaryOnly=1
