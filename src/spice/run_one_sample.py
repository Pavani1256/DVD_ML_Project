import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = ROOT / "models" / "16nm_LP.pm"
INPUT_CSV = ROOT / "data" / "samples" / "nmos_on_samples.csv"

WORK_DIR = ROOT / "data" / "spice_test"
WORK_DIR.mkdir(parents=True, exist_ok=True)

NGSPICE = (
    os.environ.get("NGSPICE")
    or shutil.which("ngspice")
    or shutil.which("ngspice_con.exe")
    or "ngspice_con.exe"
)

LMIN = 16e-9

PROCESS_PARAMETERS = ["vth0", "toxe", "ndep", "u0", "xj"]


# --------------------------------------------------
# Modify one transistor model section
# --------------------------------------------------

def make_modified_model(original_text, transistor, row):
    """
    Make a temporary copy of the model file and replace
    the five process parameters for the selected transistor.
    """

    # Locate the selected model section.
    model_pattern = re.compile(
        rf"(?im)^\.model\s+{transistor}\b"
    )

    match = model_pattern.search(original_text)

    if not match:
        raise ValueError(f"Could not find {transistor} model.")

    start = match.start()

    # Find the next .model declaration, if one exists.
    next_model = re.search(
        r"(?im)^\.model\s+",
        original_text[match.end():]
    )

    if next_model:
        end = match.end() + next_model.start()
    else:
        end = len(original_text)

    section = original_text[start:end]

    # Replace each sampled parameter in this section.
    for parameter in PROCESS_PARAMETERS:

        value = float(row[parameter])

        pattern = re.compile(
            rf"(\b{parameter}\s*=\s*)"
            rf"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
            re.IGNORECASE
        )

        section, count = pattern.subn(
            lambda m: m.group(1) + f"{value:.12g}",
            section,
            count=1
        )

        if count != 1:
            raise ValueError(
                f"Could not replace {parameter} in {transistor} model."
            )

    return original_text[:start] + section + original_text[end:]


# --------------------------------------------------
# Generate a SPICE netlist for one sample
# --------------------------------------------------

def make_netlist(row, transistor, work_dir=None):
    """Write the modified model and netlist into work_dir (one per worker
    process when simulations run in parallel) and return the netlist path."""

    work_dir = Path(work_dir) if work_dir is not None else WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    original_model = MODEL_FILE.read_text()

    modified_model = make_modified_model(
        original_model,
        transistor,
        row
    )

    # Save the temporary model file.
    modified_model_path = work_dir / "sample_model.pm"
    modified_model_path.write_text(modified_model)

    # Use forward slashes in the SPICE include path.
    model_path = modified_model_path.resolve().as_posix()

    netlist = f"""
* One-sample transistor simulation

.include "{model_path}"

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9

.temp {row['temp']:.12g}

VDS d 0 DC {row['vd']:.12g}
VGS g 0 DC {row['vg']:.12g}
VSS s 0 DC {row['vs']:.12g}
VBS b 0 DC {row['vb']:.12g}

M1 d g s b {transistor} L={LMIN:.12g} W={row['w']:.12g}

.control
op

let Id = -i(VDS)
let Ig = -i(VGS)
let Is = -i(VSS)
let Ib = -i(VBS)

print Id
print Ig
print Is
print Ib

quit
.endc

.end
"""

    netlist_path = work_dir / "one_sample.cir"
    netlist_path.write_text(netlist)

    return netlist_path


# --------------------------------------------------
# Run ngspice and extract currents
# --------------------------------------------------

def run_ngspice(netlist_path):

    result = subprocess.run(
        [NGSPICE, "-b", str(netlist_path)],
        capture_output=True,
        text=True,
        timeout=60
    )

    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            "ngspice failed:\n" + output
        )

    currents = {}

    for name in ["Id", "Ig", "Is", "Ib"]:

        pattern = rf"(?im)^\s*{name}\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)"

        match = re.search(pattern, output)

        if not match:
            raise RuntimeError(
                f"Could not find {name} in ngspice output:\n{output}"
            )

        currents[name.lower()] = float(match.group(1))

    return currents, output


# --------------------------------------------------
# Main: test the first NMOS ON sample
# --------------------------------------------------

def main():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(MODEL_FILE)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    # Read the first row from the input CSV.
    df = pd.read_csv(INPUT_CSV)
    sample = df.iloc[0]

    print("Input sample:")
    print(sample.to_string())

    # Generate the netlist and modified model.
    netlist_path = make_netlist(sample, "nmos")

    print("\nGenerated netlist:")
    print(netlist_path)

    # Run ngspice.
    currents, spice_output = run_ngspice(netlist_path)

    # Calculate KCL residual.
    kcl = sum(currents.values())

    print("\nSPICE terminal currents:")
    for name, value in currents.items():
        print(f"{name} = {value:.12e} A")

    print(f"\nKCL residual = {kcl:.12e} A")

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()