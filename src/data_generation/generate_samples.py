
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------

SEED = 2026
N_SAMPLES = 10000

VDD_NOMINAL = 0.9
LMIN = 16e-9

OUTPUT_DIR = Path("data/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# -------------------------------------------------
# Nominal model parameters from 16nm_LP.pm
# -------------------------------------------------

PARAMETERS = {
    "nmos": {
        "vth0": 0.68191,
        "toxe": 1.20e-9,
        "ndep": 7.0e18,
        "u0": 0.028,
        "xj": 5.0e-9
    },

    "pmos": {
        "vth0": -0.6862,
        "toxe": 1.22e-9,
        "ndep": 4.4e18,
        "u0": 0.0075,
        "xj": 7.2e-9
    }
}

# -------------------------------------------------
# Generate one dataset
# -------------------------------------------------

def generate_samples(device, state):

    nominal = PARAMETERS[device]

    # Draw VDD first
    vdd = rng.uniform(
        0.9 * VDD_NOMINAL,
        1.1 * VDD_NOMINAL,
        N_SAMPLES
    )

    # Independent drain and source voltages
    vd = rng.uniform(0, 1, N_SAMPLES) * vdd
    vs = rng.uniform(0, 1, N_SAMPLES) * vdd

    # Width
    w = rng.uniform(LMIN, 8 * LMIN, N_SAMPLES)

    # Temperature
    temp = rng.uniform(-40, 125, N_SAMPLES)

    # Fixed gate and body voltages
    if device == "nmos":
        if state == "ON":
            vg = vdd.copy()
            vb = np.zeros(N_SAMPLES)
        else:
            vg = np.zeros(N_SAMPLES)
            vb = np.zeros(N_SAMPLES)

    else:
        if state == "ON":
            vg = np.zeros(N_SAMPLES)
            vb = vdd.copy()
        else:
            vg = vdd.copy()
            vb = vdd.copy()

    # Independently sample process parameters
    process = {}

    for parameter, value in nominal.items():

        lower = min(0.9 * value, 1.1 * value)
        upper = max(0.9 * value, 1.1 * value)

        process[parameter] = rng.uniform(
            lower, upper, N_SAMPLES
        )

    # Construct dataset
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": temp,
        "w": w,
        "vd": vd,
        "vs": vs,
        "vg": vg,
        "vb": vb,
        "vth0": process["vth0"],
        "toxe": process["toxe"],
        "ndep": process["ndep"],
        "u0": process["u0"],
        "xj": process["xj"]
    })

    filename = f"{device}_{state.lower()}_samples.csv"

    df.to_csv(OUTPUT_DIR / filename, index=False)

    print(f"Generated: {filename}")
    print(f"Rows: {len(df)}")
    print()

    return df


# -------------------------------------------------
# Generate all four cases
# -------------------------------------------------

if __name__ == "__main__":

    cases = [
        ("nmos", "ON"),
        ("nmos", "OFF"),
        ("pmos", "ON"),
        ("pmos", "OFF")
    ]

    for device, state in cases:
        generate_samples(device, state)

    print("All four parameter datasets generated.")
