# Sensing Motion, Building Routes: ns-3 simulation core

This repository contains the compact, native ns-3 experiment core for the article *Sensing Motion, Building Routes: Design Lessons for Embodied Low-Altitude IoT*. It includes the source-route protocol and task simulation, frozen predictor parameters, and one small scenario with its recorded result. It does **not** contain the ns-3 distribution, the paper, reviewer correspondence, private infrastructure details, or all raw experiment outputs.

## What is modeled

- `src/p2_native_density.cc` implements HELLO exchange, improving-label route discovery and merging, reply and error handling, link scores, source-routed task packets, and a fixed follower controller. It calls ns-3.32 for ad hoc 802.11a PHY, propagation, DCF MAC, IP, ARP, and acknowledgments.
- `inputs/` contains the fixed Ridge, Motion MLP, and Context MLP weights and one synthetic formation mission (`formation_n24_s4501.dat`).
- `scripts/generate_missions.py` regenerates the synthetic mission inputs used by the density experiment. The included fixture permits a numerical generator check with an absolute tolerance of `1e-10`.
- `examples/formation_n24_s4501_motion.reference.json` is the archived aggregate result for the included input and Motion MLP. It is a reference for one run, not an independent measurement.

The shared-channel density experiment used three synthetic mission families, 18/24/30 nodes, seeds 4501–4512, and five metrics: hop count, analytical link expiration, Ridge, Motion MLP, and Context MLP. The resulting 540 configurations share the same protocol interface and frozen parameters. The route builder is an AODV-style source-route overlay, **not** a complete RFC 3561 implementation. The channel is a packet-level simulation; it is not a measured UAV channel, waveform-level study, onboard benchmark, or flight experiment. Model weights do not learn online. Results do not establish a general neural advantage.

## Requirements

- Linux, a C++17 compiler, and a separately installed and built **ns-3.32**. The original validation used the debug build and g++ 14.2.0. The bundled script links against the debug libraries; use `NS3_BUILD_PROFILE=optimized` only when matching ns-3.32 optimized libraries are present.
- Python 3 and NumPy for mission generation and its test. NumPy is not needed to run the C++ simulator with the included input.

Build ns-3.32 using its own documentation first. Then, from this repository:

```bash
bash scripts/build.sh /absolute/path/to/ns-3.32
bash scripts/run_example.sh
python3 scripts/check_example.py outputs/formation_n24_s4501_motion.json
python3 -m unittest discover -s tests -v
```

The first command uses ns-3.32's generated headers under `build/ns3` and its shared libraries under `build/lib`. It does not copy files into or overwrite the ns-3 source tree. The example creates ignored files under `outputs/`. The comparison script reports the difference from the archived result; matching it is an empirical check, not guaranteed for different builds or platforms.

To generate the complete fixed mission grid in a new directory:

```bash
python3 scripts/generate_missions.py --all --output-dir generated_inputs
```

Copy the seven frozen model `.dat` files from `inputs/` into that directory before running the simulator with `--input=generated_inputs`. The generator refuses to overwrite existing mission files. In a local NumPy 2.3.5 check against the 108 archived missions, the largest absolute numeric difference was below `4e-13`; small floating-point differences can still affect a discrete-event replay. To run a case directly, use the parameters shown in `scripts/run_example.sh`; `--method` accepts `hop`, `analytical`, `ridge`, `motion`, and `context`. Set `--mission`, `--nodes`, and `--seed` to the generated file's values. This repository supplies the experiment core, a generator, and a reference case; it does not supply the complete 540-run result archive or the analysis pipeline.

## Code and licensing

The experiment source retains its original `GPL-2.0-or-later` notice; the version 2 license text is supplied in `LICENSE-GPL-2.0.txt`. ns-3 is a separately obtained dependency with its own licensing terms. This repository does not grant additional permissions for the article or third-party materials. The small generated input and numeric parameter files are provided only as part of this simulation artifact; no additional license is specified for them here.
