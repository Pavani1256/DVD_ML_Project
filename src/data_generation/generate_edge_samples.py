"""Supplementary "rail-edge" samples for the four transistor cases.

The required datasets (generate_samples.py) draw Vd and Vs uniformly in
[0, VDD]. Inside a logic gate, however, transistors operate with Vd and Vs at
(or within microvolts of) the rails 0 and VDD. Under uniform sampling the
probability that both Vd and Vs fall within 1% of VDD is ~1e-4, so the
models had to extrapolate exactly where gate leakage is evaluated.

This script draws extra samples with every parameter distributed as in the
project spec EXCEPT Vd and Vs, which are each drawn from a mixture:
  40%  near 0:    VDD * 10^U(-7, -1)
  40%  near VDD:  VDD * (1 - 10^U(-7, -1))
  20%  uniform:   U(0, VDD)
They are stored separately (data/samples/{case}_edge_samples.csv) so the
required 10,000-row uniform datasets stay unchanged.

    python .\\src\\data_generation\\generate_edge_samples.py
    python .\\src\\spice\\run_dataset.py --suffix _edge
"""

from pathlib import Path

import numpy as np
import pandas as pd

from generate_samples import LMIN, PARAMETERS, VDD_NOMINAL

SEED = 2028            # independent of the uniform dataset seed (2026)
N_SAMPLES = 6000
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

rng = np.random.default_rng(SEED)


def rail_biased(vdd):
    """Bias Vd/Vs directly toward the logic rails that dominate gate leakage.

    The original edge generator already concentrated samples near 0 and VDD, but
    the leakage-sensitive operating points in complete gates are even more
    concentrated at the exact rails. This distribution adds direct samples at
    0 V and VDD, plus near-rail samples, so the model sees the true gate-bias
    regime before the validation pass.
    """
    n = len(vdd)
    choice = rng.choice(5, size=n, p=[0.25, 0.25, 0.25, 0.15, 0.10])
    exact_low = np.zeros(n)
    exact_high = vdd.copy()
    near_low = vdd * 10.0 ** rng.uniform(-9, -2, n)
    near_high = vdd - vdd * 10.0 ** rng.uniform(-9, -2, n)
    uniform = rng.uniform(0, 1, n) * vdd
    sample = np.select(
        [choice == 0, choice == 1, choice == 2, choice == 3],
        [exact_low, exact_high, near_low, near_high],
        default=uniform,
    )
    return np.clip(sample, 0.0, vdd)


def generate(device, state):
    nominal = PARAMETERS[device]
    vdd = rng.uniform(0.9 * VDD_NOMINAL, 1.1 * VDD_NOMINAL, N_SAMPLES)
    vd = rail_biased(vdd)
    vs = rail_biased(vdd)
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
    path = OUTPUT_DIR / f"{device}_{state.lower()}_edge_samples.csv"
    df.to_csv(path, index=False)
    print(f"Generated {path.name}: {len(df)} rows")


if __name__ == "__main__":
    for device, state in [("nmos", "ON"), ("nmos", "OFF"), ("pmos", "ON"), ("pmos", "OFF")]:
        generate(device, state)
