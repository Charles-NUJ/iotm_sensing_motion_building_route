# Sensing Motion, Building Routes

This repository contains the ns-3 simulation core and Python model-training workflow for *Sensing Motion, Building Routes: Design Lessons for Embodied Low-Altitude IoT*. It studies how mobile agents use local motion information to establish multihop routes and deliver task data.

## Overview

Agents exchange local motion information. In the learning-based variants, a fixed predictor estimates a one-hop link-holding score. A distributed route builder uses link scores to discover a path and merges repeated route-request updates. Delivered packets update the state estimate used by a fixed follower controller. The resulting motion changes the network geometry.

The implementation uses the ns-3.32 802.11a PHY, propagation, DCF MAC, and IP stack. It is an AODV-style source-route overlay, not a complete RFC 3561 implementation.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/p2_native_density.cc` | Density-study routing, link scoring, task control, and ns-3 simulation |
| `src/p2_native_kf15.cc` | Final 15-seed study core with timestamp-aware state estimation |
| `inputs/` | Frozen predictor parameters and example missions for both cores |
| `scripts/` | Build, example, result-checking, and mission-generation scripts |
| `examples/` | Archived result for the included example |
| `tests/` | Mission-generator test |
| `python_training/` | Compact/Rich MLP and Ridge training, TFLite export, and tests |

The repository does not include the ns-3 distribution or the complete experiment results.

## Requirements

- Linux and a C++17 compiler.
- A separately built [ns-3.32](https://www.nsnam.org/releases/ns-3-32/) debug build, with headers in `build/ns3` and libraries in `build/lib`.
- Python 3 and NumPy for generating missions and running the test. NumPy is not required to run the included C++ example.

## Run the example

After building ns-3.32, clone this repository and run:

```bash
git clone https://github.com/Charles-NUJ/iotm_sensing_motion_building_route.git
cd iotm_sensing_motion_building_route

bash scripts/build.sh /absolute/path/to/ns-3.32
bash scripts/run_example.sh
python3 scripts/check_example.py outputs/formation_n24_s4501_motion.json
bash scripts/run_kf15_example.sh
```

The first example uses a 24-node formation mission, seed 4501, and the Motion MLP score. Its result is saved in `outputs/`. The check compares it with the archived result in `examples/` and prints `REFERENCE_MATCH` when the checked values agree. The second example uses the final timestamp-aware state estimator with seed 4601. Results may differ across build environments.

To check the mission generator:

```bash
python3 -m unittest discover -s tests -v
```

## Generate other missions

The generator supports waypoint, Gauss-Markov, and formation missions with 18, 24, or 30 nodes. The following command generates the mission inputs used in the density experiment:

```bash
python3 scripts/generate_missions.py --all --output-dir generated_inputs
cp inputs/ridge.dat inputs/motion_s*.dat inputs/context_s*.dat generated_inputs/
```

The generator does not overwrite existing mission files. It may produce small floating-point differences across environments. The density core accepts `hop`, `analytical`, `ridge`, `motion`, and `context`. The final core also accepts `base_motion` and uses `motion` and `context` for the corresponding fine-tuned priors. The repository retains the source, model workflow, frozen score parameters, and executable examples; the large raw result archive is stored separately.

## Train the link-holding models

The training workflow is independent of the ns-3 build. It downloads no data
automatically. Follow `python_training/README.md` to use the public Flight
Routes dataset, train the two endpoint-symmetric MLPs and Ridge baseline, and
export the neural models to TFLite.

## Scope and license

The system experiments use native ns-3.32 802.11a PHY/MAC discrete-event simulation. The simulation sources in `src/` retain their `GPL-2.0-or-later` notices; the version 2 license text is in `LICENSE-GPL-2.0.txt`. ns-3 is a separate dependency. No additional license is specified here for the other project files.
