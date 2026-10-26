# Sensing Motion, Building Routes

This repository provides the ns-3 and Python code for *Sensing Motion,
Building Routes: Design Lessons for Embodied Low-Altitude IoT*. Mobile agents
use local motion states to score one-hop link persistence. An explicit
distributed route process then builds multihop paths and uses packet feedback
to refresh the state applied by the follower controller.

The current workflow covers five routing metrics:

- **Frozen:** one common 241-parameter persistence prior.
- **Scene FT:** one fine-tuned model for each motion family.
- **Speed FT:** one fine-tuned model for each measured relative-speed band.
- **Scene-speed FT:** one fine-tuned model for each family and speed band.
- **LET:** the analytical constant-velocity link-expiration metric.

Hop count is the paired reference. The final analysis retains complete paired
tasks whose delivered Hop route contains at least four hops.

## Code map

| Path | Purpose |
| --- | --- |
| src/p2_native_absolute_speed.cc | Main ns-3.32 PHY/MAC, mobility, routing, estimation, and following experiment |
| factorial_experiment/generate_data.py | Generate native mobility traces for disjoint training, validation, and evaluation seeds |
| factorial_experiment/train_models.py | Build link-holding examples and train scene, speed, and joint models |
| factorial_experiment/run_experiment.py | Run paired Hop, LET, Frozen, Scene FT, Speed FT, and Scene-speed FT trials |
| factorial_experiment/analyze_and_plot.py | Apply the multihop pairing rule, summarize results, and draw the two-panel violin figure |
| factorial_experiment/config.json | Motion families, speed ranges, seeds, model settings, and pairing rules |
| tests/ | Mission generation, split, task-matrix, speed-band, and endpoint-symmetry checks |
| python_training/ | Offline-prior workflow using public flight tracks |
| src/p2_native_density.cc | Node-density extension |
| src/p2_native_kf15.cc | Timestamp-aware estimation study |

The main C++ source uses the [ns-3](https://www.nsnam.org/) 802.11a PHY,
propagation model, ad hoc DCF MAC, ACK/retry behavior, and IP stack. It also
uses the native Random Waypoint and Gauss-Markov mobility models. Formation
motion uses constant-velocity agents with packet-driven follower updates.

## Requirements

- Linux with a C++17 compiler
- A built ns-3.32 tree
- Python 3
- NumPy for code checks and trace processing
- PyTorch for fine-tuning
- Matplotlib for the result figure

Install the Python packages for the final workflow:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r factorial_experiment/requirements.txt
~~~

## Build the ns-3 programs

~~~bash
bash scripts/build.sh /absolute/path/to/ns-3.32
~~~

This creates build/p2_native_absolute_speed, build/p2_native_density, and
build/p2_native_kf15.

## Check the code

The checks below use only the small deterministic fixtures included in the
repository.

~~~bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s python_training/tests -v
python3 -m factorial_experiment.generate_data --dry-run --limit 2
python3 -m factorial_experiment.train_models --dry-run
~~~

## Run the final workflow

The complete sequence is:

~~~bash
python3 -m factorial_experiment.generate_data --split all
python3 -m factorial_experiment.train_models
python3 -m factorial_experiment.run_experiment
python3 -m factorial_experiment.analyze_and_plot
~~~

The same sequence is available through factorial_experiment/run_all.sh.

The default design uses 20 whole-mission seeds for training, five for model
selection, and ten for final evaluation. The seed groups are disjoint. The
absolute-speed axis is 0--5 m/s. Link examples are assigned by measured
endpoint relative speed to 2--4, 5--7, or 8--10 m/s training bands. The figure
combines the first two bands as the lower-speed display group and keeps the
8--10 m/s group as the high-speed display group.

All generated traces, model files, logs, summaries, and figures are written
under factorial_experiment/artifacts/. That directory is ignored by Git. The
repository therefore contains generation and analysis code rather than a copy
of the experiment data. Frozen model parameters and small deterministic
fixtures remain tracked because the checks and examples use them.

## Offline prior

The separate python_training/ workflow forms the initial endpoint-symmetric
prior from public flight tracks. It reads an external checkout and writes
models and results into ignored directories. See python_training/README.md for
its commands.

## License

The ns-3-linked simulation sources retain their GPL-2.0-or-later notices. The
version 2 license text is provided in LICENSE-GPL-2.0.txt. ns-3 is a separate
dependency.
