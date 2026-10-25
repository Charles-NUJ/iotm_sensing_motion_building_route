"""Generate the density-extension exogenous missions from the archived task rules.

The C++ simulator replaces follower positions with packet-driven control. The
input generator uses the same initial, unrefreshed leader estimate as the
original preparation code; no network events run during input generation.
"""
import argparse
import io
import itertools
import math
from pathlib import Path

import numpy as np

FAMILIES = ("waypoint", "gauss_markov", "formation")
NODE_COUNTS = (18, 24, 30)
SEEDS = range(4501, 4513)

def generate(family, n, seed):
    if family not in FAMILIES or n not in NODE_COUNTS:
        raise ValueError("unsupported family or node count")
    rng = np.random.default_rng(seed)
    rows = n // 6
    p = np.array([[120.0 * (i % 6), 300.0 * (i // 6) / (rows - 1), 60.0] for i in range(n)])
    p[:, :2] += rng.uniform(-8, 8, (n, 2))
    p[0] = [0, 0, 60]
    offsets = p.copy() - p[0]
    v = np.zeros_like(p)
    destinations = (n - 1, n - 2)
    targets = rng.uniform([0, 0, 60], [600, 300, 60], (n, 3))
    phases = rng.uniform(0, 2 * np.pi, n)
    frequency = rng.uniform(0.10, 0.16)
    phase = rng.uniform(0, 2 * np.pi)
    leader_initial = p[0].copy()
    out = io.StringIO()
    out.write(f"{n} 400\n")
    def line(values):
        out.write(" ".join(format(float(x), ".17g") for x in values) + "\n")
    for position, velocity in zip(p, v):
        line(list(position) + list(velocity))
    for step in range(400):
        t, dt = step * 100.0 / 1000, 0.1
        leader_v = np.array([3 * math.cos(frequency * t + phase), 3 * math.sin(frequency * t + phase), 0.0])
        desired = np.zeros_like(v)
        desired[0] = leader_v
        for i in range(1, n):
            if i in destinations:
                desired[i] = 0.6 * (leader_initial + offsets[i] - p[i])
            elif family == "formation":
                wobble = np.array([20 * math.sin(0.2 * t + phases[i]), 20 * math.cos(0.2 * t + phases[i]), 0])
                desired[i] = leader_v + 0.4 * (p[0] + offsets[i] + wobble - p[i])
            elif family == "waypoint":
                delta = targets[i] - p[i]
                if np.linalg.norm(delta) < 6:
                    targets[i] = rng.uniform([0, 0, 60], [600, 300, 60])
                    delta = targets[i] - p[i]
                desired[i] = 4.0 * delta / max(1.0, np.linalg.norm(delta))
            else:
                desired[i] = 0.98 * v[i] + rng.normal(0, 0.6, 3)
                desired[i, 2] = 0
                for j, bound in ((0, 600), (1, 300)):
                    if p[i, j] < 0: desired[i, j] = abs(desired[i, j])
                    if p[i, j] > bound: desired[i, j] = -abs(desired[i, j])
        speed = np.linalg.norm(desired, axis=1)
        desired *= np.minimum(1, 8.0 / np.maximum(speed, 1e-9))[:, None]
        dv = desired - v
        dv *= np.minimum(1, 2.0 * dt / np.maximum(np.linalg.norm(dv, axis=1), 1e-9))[:, None]
        v += dv
        p += dt * v
        values = [step * 0.1]
        values.extend(x for position, velocity in zip(p, v) for x in list(position) + list(velocity))
        line(values)
    return out.getvalue().encode("ascii")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="generate all 108 density missions")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--nodes", type=int, choices=NODE_COUNTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.all:
        cases = list(itertools.product(FAMILIES, NODE_COUNTS, SEEDS))
    elif args.family and args.nodes and args.seed is not None:
        cases = [(args.family, args.nodes, args.seed)]
    else:
        parser.error("provide --all or all of --family, --nodes, and --seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for family, n, seed in cases:
        target = args.output_dir / f"{family}_n{n}_s{seed}.dat"
        # Exclusive creation protects an existing fixture or earlier run.
        with target.open("xb") as f:
            f.write(generate(family, n, seed))
        print(target)

if __name__ == "__main__":
    main()
