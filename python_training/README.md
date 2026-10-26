# Python training workflow

This folder trains the endpoint-symmetric link-holding models used by the
paper. The learned model produces a one-link persistence score. Route
construction remains in the explicit distributed protocol.

## Data

Clone the public Flight Routes dataset and check out the recorded revision:

```bash
git clone https://github.com/Charles-NUJ/flightroutes.git data/flightroutes
git -C data/flightroutes checkout 81a09f48
```

The dataset is not copied into this repository. You may instead set
`P2_FLIGHTROUTES_DIR` to another checkout of the same revision.

## Environment

The validated reference environment uses Python 3.8.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Train and export

```bash
python train_link_hold_model.py
```

The default seeds are `20260721,20260722,20260723`. Override them with
`P2_LINK_HOLD_SEEDS`. Outputs are written to `models/` and `results/`.
The workflow trains the Compact MLP, the neighbor-context Rich MLP, and the
same-input Ridge baseline, then exports the neural models to TFLite.

## Tests

The data-only tests do not require TensorFlow:

```bash
python -m unittest tests.test_link_hold_model -v
```

They check the temporal embargo, endpoint symmetry, future-window labels,
and the static fallback.
