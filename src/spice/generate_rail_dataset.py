"""Supplementary single-transistor dataset at static-CMOS operating points.

Why: inside a static CMOS gate every transistor's drain and source sit exactly
at a rail (0 or VDD) or, in a series stack, one of them sits at a rail and the
other at an internal node. The uniform datasets almost never sample those
exact points, and the general models are 3-8 % off there, which is most of
the gate-level error. This dataset trains the operating-point sub-models
(train_rail_models.py). It contains single-transistor SPICE results only: no
gate or circuit simulation is used for training.

Two sets, each over all four cases (NMOS/PMOS, ON/OFF; vg and vb as in the
main datasets):
  exact     (vd, vs) in {0, VDD}^2                -> configs 00 01 10 11
  one_rail  one of vd, vs at a rail, the other    -> configs m0 m1 0m 1m
            uniform in [0, VDD] ("m" = free)
VDD, temperature, W and the five process parameters follow the main dataset
distributions (generate_samples.py), with their own fixed seeds. One draw of
VDD/T/process ("corner") is shared by the 16 devices (4 cases x 4 configs)
simulated together in one ngspice run; every device has its own four voltage
sources, as in run_one_sample.py. Corners whose operating point needed
ngspice's transient-op fallback are dropped.

    python .\\src\\spice\\generate_rail_dataset.py --ngspice "C:\\Ngspice\\Spice64\\bin\\ngspice_con.exe" --workers 8

Output: data/processed/rail_samples.csv.gz
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_one_sample import LMIN, MODEL_FILE, ROOT, make_modified_model  # noqa: E402

OUTPUT = ROOT / "data" / "processed" / "rail_samples.csv.gz"
WORK_DIR = ROOT / "data" / "spice_work" / "rail"
VDD_NOMINAL = 0.9
SETS = {
    # name: (seed, number of corners, configs)
    "exact": (4041, 4000, [(0, 0), (0, 1), (1, 0), (1, 1)]),
    "one_rail": (5051, 6000, [("m", 0), ("m", 1), (0, "m"), (1, "m")]),
}
NOMINAL = {
    "nmos": {"vth0": 0.68191, "toxe": 1.20e-9, "ndep": 7.0e18, "u0": 0.028, "xj": 5.0e-9},
    "pmos": {"vth0": -0.6862, "toxe": 1.22e-9, "ndep": 4.4e18, "u0": 0.0075, "xj": 7.2e-9},
}
# case: (transistor, vg / VDD, vb / VDD), as in generate_samples.py
CASES = {
    "nmos_on": ("nmos", 1, 0), "nmos_off": ("nmos", 0, 0),
    "pmos_on": ("pmos", 0, 1), "pmos_off": ("pmos", 1, 1),
}
NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def draw_corners(seed: int, n_corners: int, configs) -> list[dict]:
    rng = np.random.default_rng(seed)
    corners = []
    for index in range(n_corners):
        vdd = rng.uniform(0.9 * VDD_NOMINAL, 1.1 * VDD_NOMINAL)
        temp = rng.uniform(-40, 125)
        process = {
            transistor: {name: rng.uniform(min(0.9 * v, 1.1 * v), max(0.9 * v, 1.1 * v))
                         for name, v in nominal.items()}
            for transistor, nominal in NOMINAL.items()
        }
        devices = []
        for case, (transistor, vg, vb) in CASES.items():
            for vd, vs in configs:
                if "m" in (vd, vs):
                    free = rng.uniform(0, 1)
                    vd = free if vd == "m" else vd
                    vs = free if vs == "m" else vs
                    config = ("m" if isinstance(vd, float) else str(vd)) + \
                             ("m" if isinstance(vs, float) else str(vs))
                else:
                    config = f"{vd}{vs}"
                devices.append({
                    "case": case, "transistor": transistor, "config": config,
                    "vdd": vdd, "temp": temp, "w": rng.uniform(LMIN, 8 * LMIN),
                    "vd": vd * vdd, "vs": vs * vdd, "vg": vg * vdd, "vb": vb * vdd,
                    **process[transistor],
                })
        corners.append({"index": index, "temp": temp, "process": process, "devices": devices})
    return corners


def simulate_corner(corner: dict, set_name: str, ngspice: str) -> list[dict]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    stem = WORK_DIR / f"{set_name}_{corner['index']}"
    model_text = MODEL_FILE.read_text()
    for transistor in ("nmos", "pmos"):
        model_text = make_modified_model(model_text, transistor, corner["process"][transistor])
    model_path = stem.with_suffix(".pm")
    model_path.write_text(model_text)
    lines = [
        f"* Operating-point samples, {set_name} corner {corner['index']}",
        f'.include "{model_path.resolve().as_posix()}"',
        ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
        f".temp {corner['temp']:.12g}",
    ]
    for u, d in enumerate(corner["devices"]):
        lines += [
            f"VD{u} d{u} 0 DC {d['vd']:.12g}",
            f"VG{u} g{u} 0 DC {d['vg']:.12g}",
            f"VS{u} s{u} 0 DC {d['vs']:.12g}",
            f"VB{u} b{u} 0 DC {d['vb']:.12g}",
            f"M{u} d{u} g{u} s{u} b{u} {d['transistor']} L={LMIN:.12g} W={d['w']:.12g}",
        ]
    lines += [".control", "set numdgt=12", "op"]
    lines += [f"print i(VD{u}) i(VG{u}) i(VS{u}) i(VB{u})" for u in range(len(corner["devices"]))]
    lines += ["quit", ".endc", ".end", ""]
    netlist_path = stem.with_suffix(".cir")
    netlist_path.write_text("\n".join(lines))
    done = subprocess.run([ngspice, "-b", str(netlist_path)], capture_output=True,
                          text=True, timeout=600, check=False)
    log = (done.stdout + done.stderr).lower()
    if done.returncode != 0 or "transient op started" in log or "dc solution failed" in log:
        return []
    rows = []
    for u, d in enumerate(corner["devices"]):
        row = {"set": set_name, "corner": corner["index"],
               **{k: v for k, v in d.items() if k != "transistor"}}
        for terminal, source in (("id", "d"), ("ig", "g"), ("is", "s"), ("ib", "b")):
            match = re.search(rf"(?im)^i\(v{source}{u}\)\s*=\s*{NUMBER}", done.stdout)
            if not match:
                return []
            row[terminal] = -float(match.group(1))   # same convention as run_one_sample.py
        rows.append(row)
    model_path.unlink()
    netlist_path.unlink()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ngspice", default="ngspice_con.exe")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frames = []
    for set_name, (seed, n_corners, configs) in SETS.items():
        corners = draw_corners(seed, n_corners, configs)
        start = time.time()
        print(f"{set_name}: {n_corners} corners x 16 devices (seed {seed})...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda c: simulate_corner(c, set_name, args.ngspice), corners))
        rows = [r for result in results for r in result]
        dropped = sum(1 for result in results if not result)
        print(f"  {len(rows)} samples, {dropped} corners dropped (convergence), "
              f"{(time.time() - start) / 60:.1f} min")
        frames.append(pd.DataFrame(rows))
    data = pd.concat(frames, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data)} samples -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
