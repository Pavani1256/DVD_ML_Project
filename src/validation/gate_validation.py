"""Validate the saved transistor models against SPICE CMOS gate simulations.

Part B of the project. For every gate (INV, NAND2, NOR2, AND2, OR2), input
combination and random PVT corner:

1. SPICE reference: the complete static-CMOS gate is simulated (.op) and the
   supply current |I(VDD)| plus all internal node voltages are recorded.
2. ML prediction: the SPICE node voltages give each transistor's terminal
   voltages. The gate voltage is snapped to its logic level (0 or VDD), as the
   transistor models were trained only at those gate voltages. Each device is
   classified ON/OFF from that logic level and passed to the matching model.
3. Leakage from KCL: an ON transistor inside a gate sits at |Vds| ~ 1 uV, where
   its channel current is a tiny residual that a regression model cannot
   resolve. So ON devices contribute only their predicted gate and body
   currents; their channel currents are solved from Kirchhoff's current law
   at the internal nodes, using the OFF devices' predicted currents. The
   supply current is then the sum of all terminal currents at the VDD node.

    python .\\src\\validation\\gate_validation.py --corners 200 \\
        --ngspice "C:\\Ngspice\\Spice64\\bin\\ngspice_con.exe"

Add --reuse-spice to re-score new models against the cached SPICE results.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transistor_ml import FEATURES, relative_error_percent  # noqa: E402

MODEL_FILE = ROOT / "models" / "16nm_LP.pm"
SAVED_MODEL_DIR = ROOT / "saved_models"
RESULT_DIR = ROOT / "results" / "gate_validation"
WORK_DIR = ROOT / "data" / "gate_validation"
SPICE_CACHE = RESULT_DIR / "gate_spice_raw.csv"

# Separate from the transistor data seed (2026) so the gate corners are an
# independent draw from the training samples.
CORNER_SEED = 7
VDD_NOMINAL = 0.9
LMIN = 16e-9
# Mid-range widths: well inside the 1-8 x Lmin training range, PMOS wider to
# offset its lower mobility.
DEFAULT_WN = 2 * LMIN
DEFAULT_WP = 4 * LMIN
TEMPERATURE_RANGE = (-40.0, 125.0)
PROCESS_PARAMETERS = ("vth0", "toxe", "ndep", "u0", "xj")
TERMINALS = ("id", "ig", "is", "ib")

GATES = {
    "INV": {
        "inputs": ("0", "1"),
        "nodes": ("out",),
        "transistors": (
            ("p1", "pmos", "out", "a", "vdd", "vdd"),
            ("n1", "nmos", "out", "a", "0", "0"),
        ),
    },
    "NAND2": {
        "inputs": ("00", "01", "10", "11"),
        "nodes": ("x", "out"),
        "transistors": (
            ("p1", "pmos", "out", "a", "vdd", "vdd"),
            ("p2", "pmos", "out", "b", "vdd", "vdd"),
            ("n1", "nmos", "out", "a", "x", "0"),
            ("n2", "nmos", "x", "b", "0", "0"),
        ),
    },
    "NOR2": {
        "inputs": ("00", "01", "10", "11"),
        "nodes": ("y", "out"),
        "transistors": (
            ("p1", "pmos", "y", "a", "vdd", "vdd"),
            ("p2", "pmos", "out", "b", "y", "vdd"),
            ("n1", "nmos", "out", "a", "0", "0"),
            ("n2", "nmos", "out", "b", "0", "0"),
        ),
    },
    "AND2": {
        "inputs": ("00", "01", "10", "11"),
        "nodes": ("x", "z", "out"),
        "transistors": (
            ("p1", "pmos", "z", "a", "vdd", "vdd"),
            ("p2", "pmos", "z", "b", "vdd", "vdd"),
            ("n1", "nmos", "z", "a", "x", "0"),
            ("n2", "nmos", "x", "b", "0", "0"),
            ("p3", "pmos", "out", "z", "vdd", "vdd"),
            ("n3", "nmos", "out", "z", "0", "0"),
        ),
    },
    "OR2": {
        "inputs": ("00", "01", "10", "11"),
        "nodes": ("y", "z", "out"),
        "transistors": (
            ("p1", "pmos", "y", "a", "vdd", "vdd"),
            ("p2", "pmos", "z", "b", "y", "vdd"),   # NOR2 output is z
            ("n1", "nmos", "z", "a", "0", "0"),
            ("n2", "nmos", "z", "b", "0", "0"),
            ("p3", "pmos", "out", "z", "vdd", "vdd"),
            ("n3", "nmos", "out", "z", "0", "0"),
        ),
    },
}

EXTERNAL_NODES = {"vdd", "0", "a", "b"}


# ============================================================
# MODEL FILE HANDLING
# ============================================================

def parse_model_parameters(model_text: str, transistor: str) -> dict[str, float]:
    """Read the nominal process parameters from a model section."""
    match = re.search(
        rf"(?ims)^\.model\s+{transistor}\b(.*?)(?=^\.model\s+|\Z)",
        model_text,
    )
    if not match:
        raise ValueError(f"Model section not found: {transistor}")
    section = match.group(1)
    values = {}
    for parameter in PROCESS_PARAMETERS:
        value_match = re.search(
            rf"\b{parameter}\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
            section,
            re.IGNORECASE,
        )
        if not value_match:
            raise ValueError(f"Parameter {parameter} not found for {transistor}")
        values[parameter] = float(value_match.group(1))
    return values


def modified_model(model_text: str, parameters: dict[str, dict[str, float]]) -> str:
    """Replace process parameters for both transistor model sections."""
    result = model_text
    for transistor, values in parameters.items():
        match = re.search(rf"(?im)^\.model\s+{transistor}\b", result)
        if not match:
            raise ValueError(f"Model section not found: {transistor}")
        next_model = re.search(r"(?im)^\.model\s+", result[match.end():])
        end = match.end() + next_model.start() if next_model else len(result)
        section = result[match.start():end]
        for parameter, value in values.items():
            section, count = re.subn(
                rf"(\b{parameter}\s*=\s*)"
                rf"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
                lambda item: item.group(1) + f"{value:.12g}",
                section,
                count=1,
                flags=re.IGNORECASE,
            )
            if count != 1:
                raise ValueError(f"Could not replace {parameter} for {transistor}")
        result = result[:match.start()] + section + result[end:]
    return result


def draw_corners(count: int, nominal: dict) -> list[dict]:
    rng = np.random.default_rng(CORNER_SEED)

    def draw(values):
        return {
            parameter: rng.uniform(min(0.9 * v, 1.1 * v), max(0.9 * v, 1.1 * v))
            for parameter, v in values.items()
        }

    corners = []
    for _ in range(count):
        corners.append({
            "vdd": rng.uniform(0.9 * VDD_NOMINAL, 1.1 * VDD_NOMINAL),
            "temp": rng.uniform(*TEMPERATURE_RANGE),
            "nmos": draw(nominal["nmos"]),
            "pmos": draw(nominal["pmos"]),
        })
    return corners


# ============================================================
# SPICE
# ============================================================

def make_netlist(gate: str, input_bits: str, corner: dict, widths: dict) -> str:
    """One complete static-CMOS gate netlist (no extra bias elements)."""
    spec = GATES[gate]
    lines = [
        f"* {gate} {input_bits}",
        '.include "{MODEL}"',
        ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
        f".temp {corner['temp']:.12g}",
        f"VDD vdd 0 DC {corner['vdd']:.12g}",
        f"VA a 0 DC {float(input_bits[0]) * corner['vdd']:.12g}",
    ]
    if len(input_bits) == 2:
        lines.append(f"VB b 0 DC {float(input_bits[1]) * corner['vdd']:.12g}")
    for name, transistor, drain, gate_node, source, body in spec["transistors"]:
        lines.append(
            f"M{name} {drain} {gate_node} {source} {body} {transistor} "
            f"L={LMIN:.12g} W={widths[transistor]:.12g}"
        )
    for node in spec["nodes"]:
        lines.append(f".nodeset v({node})={corner['vdd'] / 2:.12g}")
    lines.extend([".control", "op"])
    lines.append("print " + " ".join(f"v({node})" for node in spec["nodes"]))
    lines.append("print i(VDD)")
    lines.extend(["quit", ".endc", "", ".end", ""])
    return "\n".join(lines)


def run_spice(job: dict, ngspice: str) -> dict:
    gate, input_bits, index = job["gate"], job["inputs"], job["corner"]
    name = f"{gate.lower()}_{input_bits}_{index}"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    model_path = WORK_DIR / f"{name}.pm"
    netlist_path = WORK_DIR / f"{name}.cir"
    model_path.write_text(job["model_text"], encoding="utf-8")
    netlist_path.write_text(
        job["netlist"].replace("{MODEL}", model_path.resolve().as_posix()),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [ngspice, "-b", str(netlist_path)],
        # A few corners need ngspice's slower transient-op fallback.
        capture_output=True, text=True, timeout=900, check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"ngspice failed for {name}:\n{output}")
    row = {"gate": gate, "inputs": input_bits, "corner": index}
    for node in GATES[gate]["nodes"]:
        match = re.search(rf"(?im)v\({re.escape(node)}\)\s*=\s*([-+0-9.eE]+)", output)
        if not match:
            raise RuntimeError(f"Missing SPICE voltage v({node}) for {name}:\n{output}")
        row[f"spice_{node}_V"] = float(match.group(1))
    current = re.search(r"(?im)i\(vdd\)\s*=\s*([-+0-9.eE]+)", output)
    if not current:
        raise RuntimeError(f"Missing SPICE VDD current for {name}:\n{output}")
    row["spice_leakage"] = abs(float(current.group(1)))
    return row


def simulate_all(corners, widths, ngspice, workers) -> pd.DataFrame:
    nominal_text = MODEL_FILE.read_text(encoding="utf-8")
    jobs = []
    for index, corner in enumerate(corners):
        model_text = modified_model(
            nominal_text, {"nmos": corner["nmos"], "pmos": corner["pmos"]}
        )
        for gate, spec in GATES.items():
            for input_bits in spec["inputs"]:
                jobs.append({
                    "gate": gate, "inputs": input_bits, "corner": index,
                    "model_text": model_text,
                    "netlist": make_netlist(gate, input_bits, corner, widths),
                })
    print(f"Running {len(jobs)} gate simulations on {workers} threads...")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(lambda j: run_spice(j, ngspice), jobs), 1):
            rows.append(row)
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}")
    return pd.DataFrame(rows)


# ============================================================
# ML PREDICTION WITH KCL
# ============================================================

def load_packages(model_dir: Path = SAVED_MODEL_DIR) -> dict[str, dict]:
    packages = {}
    for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off"):
        path = Path(model_dir) / f"{case}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Saved model not found: {path}")
        packages[case] = joblib.load(path)
    return packages


def device_rows(spice: pd.DataFrame, corners, widths) -> pd.DataFrame:
    """One row per transistor instance with its model case and features."""
    rows = []
    for record in spice.itertuples(index=False):
        record = record._asdict()
        gate, input_bits = record["gate"], record["inputs"]
        corner = corners[record["corner"]]
        vdd = corner["vdd"]
        voltage = {
            "0": 0.0,
            "vdd": vdd,
            "a": float(input_bits[0]) * vdd,
            "b": float(input_bits[1]) * vdd if len(input_bits) == 2 else 0.0,
        }
        for node in GATES[gate]["nodes"]:
            voltage[node] = record[f"spice_{node}_V"]
        for name, transistor, drain, gate_node, source, body in GATES[gate]["transistors"]:
            gate_high = voltage[gate_node] > vdd / 2
            is_on = gate_high if transistor == "nmos" else not gate_high
            rows.append({
                "gate": gate, "inputs": input_bits, "corner": record["corner"],
                "device": name,
                "case": f"{transistor}_{'on' if is_on else 'off'}",
                "is_on": is_on,
                "vdd": vdd,
                "temp": corner["temp"],
                "w": widths[transistor],
                "vd": voltage[drain],
                "vs": voltage[source],
                "vg": vdd if gate_high else 0.0,   # snapped logic level
                "vb": voltage[body],
                **corner[transistor],
            })
    return pd.DataFrame(rows)


def predict_devices(devices: pd.DataFrame, packages) -> pd.DataFrame:
    devices = devices.copy()
    for terminal in TERMINALS:
        devices[terminal] = np.nan
    for case, group in devices.groupby("case"):
        prediction = packages[case]["model"].predict(
            group[FEATURES].to_numpy(dtype=float)
        )
        devices.loc[group.index, list(TERMINALS)] = prediction
    return devices


def kcl_supply_current(gate: str, devices: pd.DataFrame) -> float:
    """Supply current from predicted device currents plus KCL.

    OFF devices use all four predicted terminal currents. ON devices use the
    predicted ig and ib; their drain current u is unknown and their source
    current is -(u + ig + ib). The unknowns are solved (least squares,
    minimum norm) from KCL at every internal node.
    """
    spec = GATES[gate]["transistors"]
    devices = devices.set_index("device")
    internal = [n for n in GATES[gate]["nodes"]]
    on_devices = [s[0] for s in spec if devices.loc[s[0], "is_on"]]

    A = np.zeros((len(internal), len(on_devices)))
    rhs = np.zeros(len(internal))

    for name, _, drain, gate_node, source, body in spec:
        d = devices.loc[name]
        terminal_nodes = {"id": drain, "ig": gate_node, "is": source, "ib": body}
        for terminal, node in terminal_nodes.items():
            if node not in internal:
                continue
            row = internal.index(node)
            if not d["is_on"] or terminal in ("ig", "ib"):
                rhs[row] -= d[terminal]
            elif terminal == "id":
                A[row, on_devices.index(name)] += 1.0
            else:  # ON-device source: -(u + ig + ib)
                A[row, on_devices.index(name)] -= 1.0
                rhs[row] += d["ig"] + d["ib"]

    unknown = (
        np.linalg.lstsq(A, rhs, rcond=None)[0] if on_devices else np.zeros(0)
    )

    supply = 0.0
    for name, _, drain, gate_node, source, body in spec:
        d = devices.loc[name]
        if d["is_on"]:
            u = unknown[on_devices.index(name)]
            currents = {"id": u, "ig": d["ig"], "is": -(u + d["ig"] + d["ib"]), "ib": d["ib"]}
        else:
            currents = {t: d[t] for t in TERMINALS}
        for terminal, node in (("id", drain), ("ig", gate_node), ("is", source), ("ib", body)):
            if node == "vdd":
                supply += currents[terminal]
    return abs(float(supply))


def ml_leakage(spice: pd.DataFrame, corners, widths, packages) -> pd.Series:
    devices = predict_devices(device_rows(spice, corners, widths), packages)
    leakage = {
        key: kcl_supply_current(key[0], group)
        for key, group in devices.groupby(["gate", "inputs", "corner"], sort=False)
    }
    return pd.Series(
        [leakage[(g, i, c)] for g, i, c in spice[["gate", "inputs", "corner"]].itertuples(index=False)],
        index=spice.index,
    )


# ============================================================
# METRICS AND PLOTS
# ============================================================

def metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gate, input_bits), group in results.groupby(["gate", "inputs"], sort=False):
        error = relative_error_percent(group["spice_leakage"], group["ml_leakage"])
        rows.append({
            "gate": gate,
            "inputs": input_bits,
            "corners": len(group),
            "spice_median_A": group["spice_leakage"].median(),
            "ml_median_A": group["ml_leakage"].median(),
            "bias_A": (group["ml_leakage"] - group["spice_leakage"]).median(),
            "median_error_percent": np.median(error),
            "p95_error_percent": np.percentile(error, 95),
            "mean_accuracy_percent": np.clip(100 - error, 0, 100).mean(),
        })
    return pd.DataFrame(rows)


def parity_plot(raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for gate, group in raw.groupby("gate", sort=False):
        ax.scatter(group["spice_leakage"], group["ml_leakage"], alpha=0.35, s=10, label=gate)
    lo = max(min(raw["spice_leakage"].min(), raw["ml_leakage"].min()), 1e-18)
    hi = max(raw["spice_leakage"].max(), raw["ml_leakage"].max())
    ax.loglog([lo, hi], [lo, hi], "r--", label="Ideal")
    ax.set_xlabel("SPICE leakage (A)")
    ax.set_ylabel("ML leakage (A)")
    ax.set_title("Gate-level ML vs SPICE leakage")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "gate_validation_parity.png", dpi=300)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corners", type=int, default=200)
    parser.add_argument("--ngspice", default=os.environ.get("NGSPICE") or shutil.which("ngspice") or shutil.which("ngspice_con.exe") or "ngspice_con.exe")
    parser.add_argument("--wn", type=float, default=DEFAULT_WN, help="NMOS width (m)")
    parser.add_argument("--wp", type=float, default=DEFAULT_WP, help="PMOS width (m)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reuse-spice", action="store_true",
                        help=f"Load SPICE results from {SPICE_CACHE.name} instead of simulating")
    parser.add_argument("--models-dir", type=Path, default=SAVED_MODEL_DIR)
    parser.add_argument("--spice-only", action="store_true",
                        help="Only run and cache the SPICE reference")
    args = parser.parse_args()
    if args.corners < 1:
        raise ValueError("--corners must be positive")

    widths = {"nmos": args.wn, "pmos": args.wp}
    nominal_text = MODEL_FILE.read_text(encoding="utf-8")
    nominal = {t: parse_model_parameters(nominal_text, t) for t in ("nmos", "pmos")}
    corners = draw_corners(args.corners, nominal)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reuse_spice:
        spice = pd.read_csv(SPICE_CACHE, dtype={"inputs": str})
        settings = spice[["seed", "wn", "wp"]].drop_duplicates()
        expected = (CORNER_SEED, args.wn, args.wp)
        if len(settings) != 1 or not np.allclose(settings.iloc[0].to_numpy(float), expected):
            raise ValueError("Cached SPICE results were made with a different seed or widths")
        if spice["corner"].max() + 1 != args.corners:
            raise ValueError("Cached SPICE results have a different corner count")
        spice = spice.drop(columns=["seed", "wn", "wp"])
    else:
        spice = simulate_all(corners, widths, args.ngspice, args.workers)
        spice.assign(seed=CORNER_SEED, wn=args.wn, wp=args.wp).to_csv(SPICE_CACHE, index=False)

    corner_table = pd.DataFrame([{
        "corner": i, "vdd": c["vdd"], "temp": c["temp"],
        **{f"n_{k}": v for k, v in c["nmos"].items()},
        **{f"p_{k}": v for k, v in c["pmos"].items()},
    } for i, c in enumerate(corners)])
    corner_table.to_csv(RESULT_DIR / "gate_corners.csv", index=False)
    if args.spice_only:
        print(f"SPICE results saved to {SPICE_CACHE}")
        return

    packages = load_packages(args.models_dir)
    raw = spice.copy()
    raw["ml_leakage"] = ml_leakage(spice, corners, widths, packages)
    raw["error_percent"] = relative_error_percent(raw["spice_leakage"], raw["ml_leakage"])
    raw = raw.merge(corner_table[["corner", "vdd", "temp", "n_vth0", "p_vth0"]], on="corner")
    raw.to_csv(RESULT_DIR / "gate_validation_raw.csv", index=False)

    summary = metrics(raw)
    summary.to_csv(RESULT_DIR / "gate_validation_metrics.csv", index=False)
    parity_plot(raw)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    error = raw["error_percent"]
    accuracy = np.clip(100 - error, 0, 100)
    print("\nAGGREGATE ACCURACY")
    print(f"Mean accuracy (100 - error, clipped at 0%): {accuracy.mean():.2f}%")
    print(f"Median accuracy: {np.median(accuracy):.2f}%")
    print(f"Predictions within +/-1%:  {(error <= 1).mean() * 100:.2f}%")
    print(f"Predictions within +/-5%:  {(error <= 5).mean() * 100:.2f}%")
    print(f"Predictions within +/-10%: {(error <= 10).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
