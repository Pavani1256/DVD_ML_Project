"""Supplementary "stack" samples for the four transistor cases.

In a series stack (NAND2 pull-down, NOR2 pull-up, and the same stages inside
AND2/OR2) one terminal of each transistor sits on a rail while the other sits
on the internal stack node. That node settles at an intermediate voltage set
by the leakage balance of the stack, e.g. y ~ 0.9 VDD in NOR2 with both inputs
high. The rail and edge sets mostly put both terminals at a rail, and the
uniform set rarely puts one terminal exactly on a rail, so this bias region is
thinly covered.

Every parameter is distributed exactly as in the project spec EXCEPT Vd and
Vs. For each sample one of them (drain or source, equal odds) is the rail
terminal and the other is the stack-node terminal:
  rail terminal:        70% exactly 0 or VDD, 30% VDD * 10^U(-9, -3) from a rail
  stack-node terminal:  VDD * 10^U(-3, log10(0.5)) away from 0 or VDD, i.e.
                        anywhere from 0.1% to 50% of VDD away from a rail,
                        log-uniform so the region near each rail is dense
The rail and the stack-node side are chosen independently (equal odds).

The files are written separately (data/samples/{case}_stack_samples.csv), so
the required 10,000-row uniform datasets are unchanged. Like the edge and rail
samples, this deviates from the uniform Vd/Vs specification and must be stated
in the report. Gate-level SPICE results are never used to build these samples.

    python src/data_generation/generate_stack_samples.py
    python src/spice/run_dataset.py --suffix _stack
"""

from pathlib import Path

import numpy as np
import pandas as pd

from generate_samples import LMIN, PARAMETERS, VDD_NOMINAL

SEED = 2030            # independent of the uniform (2026), edge (2028) and rail (2029) seeds
N_SAMPLES = 10000
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

rng = np.random.default_rng(SEED)


def near_rail(vdd, low_exp, high_exp):
    """A voltage VDD * 10^U(low_exp, high_exp) away from 0 or VDD (equal odds)."""
    n = len(vdd)
    offset = vdd * 10.0 ** rng.uniform(low_exp, high_exp, n)
    return np.where(rng.random(n) < 0.5, vdd - offset, offset)


def rail_terminal(vdd):
    n = len(vdd)
    exact = np.where(rng.random(n) < 0.5, vdd, 0.0)
    return np.where(rng.random(n) < 0.7, exact, near_rail(vdd, -9, -3))


def generate(device, state):
    nominal = PARAMETERS[device]
    vdd = rng.uniform(0.9 * VDD_NOMINAL, 1.1 * VDD_NOMINAL, N_SAMPLES)
    rail = rail_terminal(vdd)
    stack = near_rail(vdd, -3, np.log10(0.5))
    drain_on_rail = rng.random(N_SAMPLES) < 0.5
    vd = np.where(drain_on_rail, rail, stack)
    vs = np.where(drain_on_rail, stack, rail)
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
    path = OUTPUT_DIR / f"{device}_{state.lower()}_stack_samples.csv"
    df.to_csv(path, index=False)
    print(f"Generated {path.name}: {len(df)} rows")


if __name__ == "__main__":
    for device, state in [("nmos", "ON"), ("nmos", "OFF"), ("pmos", "ON"), ("pmos", "OFF")]:
        generate(device, state)
