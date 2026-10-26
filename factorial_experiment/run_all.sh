#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"

python3 -m factorial_experiment.generate_data --split all
python3 -m factorial_experiment.train_models
python3 -m factorial_experiment.run_experiment
python3 -m factorial_experiment.analyze_and_plot
