"""Supplementary "rail" samples for the four transistor cases.

Inside a static CMOS gate every transistor source sits exactly on a rail
(0 or VDD) or on an internal stack node, and almost every drain sits within
a few microvolts of a rail. Gate-level leakage is therefore set by the device
currents at the four (Vd, Vs) rail corners, e.g. the ON PMOS gate-tunnelling
current at Vd = Vs = VDD, Vg = 0. Under the uniform specification only a
vanishing fraction of samples fall there, and even the rail-edge set
(generate_edge_samples.py) puts only ~6% of its samples exactly on a corner.

This script draws extra samples with every parameter distributed exactly as
in the project spec EXCEPT Vd and Vs, which are each drawn independently from:
  45%  exactly on a rail:   0 or VDD (equal odds)
  35%  next to a rail:      VDD * 10^U(-9, -3) away from 0 or VDD
  20%  uniform:             U(0, VDD)  (covers internal stack nodes)
so ~64% of samples have both terminals at/next to a rail and ~32% have one
terminal on a rail and the other anywhere (a stacked device).

The files are written separately (data/samples/{case}_rail_samples.csv), so
the required 10,000-row uniform datasets are unchanged. Like the edge samples,
this deviates from the uniform Vd/Vs specification and must be stated in the
report. Gate-level SPICE results are never used to build these samples.

    python src/data_generation/generate_rail_samples.py
    python src/spice/run_dataset.py --suffix _rail
"""

from pathlib import Path

import numpy as np
import pandas as pd

from generate_samples import LMIN, PARAMETERS, VDD_NOMINAL

SEED = 2029            # independent of the uniform (2026) and edge (2028) seeds
N_SAMPLES = 10000
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

rng = np.random.default_rng(SEED)


def rail_terminal(vdd):
    n = len(vdd)
    choice = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])
    high = rng.random(n) < 0.5
    rail = np.where(high, vdd, 0.0)
    offset = vdd * 10.0 ** rng.uniform(-9, -3, n)
    near = np.where(high, vdd - offset, offset)
    uniform = rng.uniform(0, 1, n) * vdd
    return np.select([choice == 0, choice == 1], [rail, near], uniform)


def generate(device, state):
    nominal = PARAMETERS[device]
    vdd = rng.uniform(0.9 * VDD_NOMINAL, 1.1 * VDD_NOMINAL, N_SAMPLES)
    vd = rail_terminal(vdd)
    vs = rail_terminal(vdd)
    w = rng.uniform(LMIN, 8 * LMIN, N_SAMPLES)
    temp = rng.uniform(-40, 125, N_SAMPLES)

    on = state == "ON"
    if device == "nmos":
        vg = vdd.copy() if on else np.zeros(N_SAMPLES)
        vb = np.zeros(N_SAMPLES)
    else:
        vg = np.zeros(N_SAMPLES) if on else vdd.copy()
        vb = vdd.copy()

    process = {
        p: rng.uniform(min(0.9 * v, 1.1 * v), max(0.9 * v, 1.1 * v), N_SAMPLES)
        for p, v in nominal.items()
    }

    df = pd.DataFrame({
        "vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs, "vg": vg, "vb": vb,
        **process,
    })
    path = OUTPUT_DIR / f"{device}_{state.lower()}_rail_samples.csv"
    df.to_csv(path, index=False)
    print(f"Generated {path.name}: {len(df)} rows")


if __name__ == "__main__":
    for device, state in [("nmos", "ON"), ("nmos", "OFF"), ("pmos", "ON"), ("pmos", "OFF")]:
        generate(device, state)
