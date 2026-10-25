# Sensing Motion, Building Routes

This repository contains the ns-3 simulation core for *Sensing Motion, Building Routes: Design Lessons for Embodied Low-Altitude IoT*. It studies how mobile agents use local motion information to establish multihop routes and deliver task data.

## Overview

Agents exchange local motion information. In the learning-based variants, a fixed predictor estimates a one-hop link-holding score. A distributed route builder uses link scores to discover a path and merges repeated route-request updates. Delivered packets update the state estimate used by a fixed follower controller. The resulting motion changes the network geometry.

The implementation uses the ns-3.32 802.11a PHY, propagation, DCF MAC, and IP stack. It is an AODV-style source-route overlay, not a complete RFC 3561 implementation.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/p2_native_density.cc` | Routing, link scoring, task control, and ns-3 simulation |
| `inputs/` | Frozen predictor parameters and one example mission |
| `scripts/` | Build, example, result-checking, and mission-generation scripts |
| `examples/` | Archived result for the included example |
| `tests/` | Mission-generator test |

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
```

The example uses a 24-node formation mission, seed 4501, and the Motion MLP score. Its result is saved in `outputs/`. The final command compares it with the archived result in `examples/` and prints `REFERENCE_MATCH` when the checked values agree. Results may differ across build environments.

To check the mission generator:

```bash
python3 -m unittest discover -s tests -v
```

## Generate other missions

The generator supports waypoint, Gauss-Markov, and formation missions with 18, 24, or 30 nodes. The following command generates the 108 mission inputs used in the density experiment:

```bash
python3 scripts/generate_missions.py --all --output-dir generated_inputs
cp inputs/ridge.dat inputs/motion_s*.dat inputs/context_s*.dat generated_inputs/
```

The generator does not overwrite existing mission files. It may produce small floating-point differences across environments. The simulator accepts `hop`, `analytical`, `ridge`, `motion`, and `context` as routing metrics; see `scripts/run_example.sh` for the command format. The full result archive and analysis pipeline are not included here.

## Scope and license

These are packet-level simulations, not UAV flight tests or measurements from a calibrated wireless channel. The simulation source `src/p2_native_density.cc` retains its `GPL-2.0-or-later` notice; the version 2 license text is in `LICENSE-GPL-2.0.txt`. ns-3 is a separate dependency. No additional license is specified here for the other project files.
