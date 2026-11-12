"""Part C: test-circuit validation of the hierarchical ML leakage model.

Assigned test circuit: 8-bit parity checker (odd parity, P = d0 ^ d1 ^ ... ^ d7),
built as a balanced tree of seven XOR2 cells. Each XOR2 is composed only of
Part B gates, so the ML side can be assembled from the Part B gate models:

    --xor-style nand  (default)  textbook 4 x NAND2 XOR
        n1 = NAND(a, b); n2 = NAND(a, n1); n3 = NAND(b, n1); y = NAND(n2, n3)
        -> 28 NAND2 gates, 112 transistors
    --xor-style andor            y = AND2(OR2(a, b), NAND2(a, b))
        -> 7 x (OR2 + NAND2 + AND2) = 21 gates, 112 transistors

For every one of the 256 input vectors at each of the SAME 200 PVT corners as
Part B (corner seed 7, Wn = 2*Lmin, Wp = 4*Lmin):

1. SPICE reference: the complete transistor-level netlist is simulated (.op)
   and the circuit leakage is |I(VDD)|. One ngspice run per vector and corner,
   with the per-corner modified model card and a .nodeset at each net's logic
   level (Newton starting point only). Every net is checked against its
   expected logic value, and runs that needed ngspice's transient-op fallback
   or whose supply absorbs current are flagged.
2. ML prediction (hierarchical): a logic simulation gives the input bits of
   every gate; each gate model (gate_validation's transistor models + KCL,
   using the Part B pre-recorded node voltages for that gate/inputs/corner)
   returns the gate's supply leakage and the current drawn into its input
   pins (sum of the predicted ig of the transistors on each pin). Two
   compositions are reported:
     plain sum      sum of gate supply leakages (literal wording of the spec)
     with loading   plain sum + the input-pin currents of gates whose input
                    sits on an internal net at logic 1. In the circuit that
                    current (gate-oxide tunnelling, large in this 16nm LP card)
                    is supplied from VDD by the driving gate's pull-up; in an
                    isolated gate it comes from the ideal input source and so
                    is not part of the gate's |I(VDD)|.
3. Composition check: both sums are also formed from SPICE gate results
   (Part B cache + a re-simulation of the same gate netlists that records the
   input-source currents). Their error against the full netlist is the
   composition error alone, so total error = gate-model error + composition.

    python .\\src\\validation\\circuit_validation.py --corners 200 \\
        --ngspice "C:\\Ngspice\\Spice64\\bin\\ngspice_con.exe"

Requires results/gate_validation/gate_spice_raw.csv from gate_validation.py
(same corner count and widths). Add --reuse-spice to re-score new models
against the cached circuit SPICE results without re-simulating.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "validation"))

import gate_validation as gv  # noqa: E402
from transistor_ml import relative_error_percent  # noqa: E402

RESULT_DIR = ROOT / "results" / "circuit_validation"
WORK_DIR = ROOT / "data" / "circuit_validation"
NETLIST_DIR = ROOT / "netlists"
N_BITS = 8
INPUTS = tuple(f"d{i}" for i in range(N_BITS))
OUTPUT_NET = "par"


# ============================================================
# CIRCUIT DESCRIPTION (gate instances)
# ============================================================

def xor_cell(name: str, a: str, b: str, y: str, style: str) -> list[tuple]:
    """Gate instances (instance, gate type, input nets, output net) of one XOR2."""
    if style == "nand":
        n1 = f"{name}_n1"
        n2 = f"{name}_n2"
        n3 = f"{name}_n3"
        return [
            (f"{name}_g1", "NAND2", (a, b), n1),
            (f"{name}_g2", "NAND2", (a, n1), n2),
            (f"{name}_g3", "NAND2", (b, n1), n3),
            (f"{name}_g4", "NAND2", (n2, n3), y),
        ]
    if style == "andor":
        o, n = f"{name}_or", f"{name}_nd"
        return [
            (f"{name}_g1", "OR2", (a, b), o),
            (f"{name}_g2", "NAND2", (a, b), n),
            (f"{name}_g3", "AND2", (o, n), y),
        ]
    raise ValueError(f"Unknown XOR style: {style}")


def parity_circuit(style: str) -> list[tuple]:
    """Balanced XOR tree: ((d0^d1)^(d2^d3)) ^ ((d4^d5)^(d6^d7))."""
    tree = [
        ("xa", "d0", "d1", "s01"),
        ("xb", "d2", "d3", "s23"),
        ("xc", "d4", "d5", "s45"),
        ("xd", "d6", "d7", "s67"),
        ("xe", "s01", "s23", "s0123"),
        ("xf", "s45", "s67", "s4567"),
        ("xg", "s0123", "s4567", OUTPUT_NET),
    ]
    gates = []
    for name, a, b, y in tree:
        gates += xor_cell(name, a, b, y, style)
    return gates


GATE_FUNCTIONS = {
    "INV": lambda a: 1 - a[0],
    "NAND2": lambda a: 1 - (a[0] & a[1]),
    "NOR2": lambda a: 1 - (a[0] | a[1]),
    "AND2": lambda a: a[0] & a[1],
    "OR2": lambda a: a[0] | a[1],
}


def vector_bits(k: int) -> dict[str, int]:
    """Input vector k -> {d0: LSB, ..., d7: MSB}."""
    return {f"d{i}": (k >> i) & 1 for i in range(N_BITS)}


def vector_label(k: int) -> str:
    """Bit string written d7..d0 (MSB first), e.g. k=5 -> '00000101'."""
    return format(k, f"0{N_BITS}b")


def logic_simulate(gates, inputs: dict[str, int]):
    """Net values and the input-bit string seen by every gate instance.

    Gates are listed in topological order, so one pass is enough.
    """
    net = dict(inputs)
    gate_inputs = {}
    for inst, gtype, ins, out in gates:
        bits = tuple(net[n] for n in ins)
        gate_inputs[inst] = "".join(str(b) for b in bits)
        net[out] = GATE_FUNCTIONS[gtype](bits)
    return net, gate_inputs


# ============================================================
# SPICE
# ============================================================

def subckt_definitions(gate_types) -> list[str]:
    """Subcircuits generated from gate_validation.GATES, so each cell is
    transistor-for-transistor identical to the Part B gate netlist."""
    lines = []
    for gtype in sorted(gate_types):
        spec = gv.GATES[gtype]
        n_in = len(spec["inputs"][0])
        ports = ["a", "b"][:n_in] + ["out", "vdd"]
        lines.append(f".subckt {gtype} {' '.join(ports)}")
        for name, transistor, drain, gate_node, source, body in spec["transistors"]:
            lines.append(
                f"M{name} {drain} {gate_node} {source} {body} {transistor} "
                "L={L} W={W_" + transistor + "}"
            )
        lines.append(f".ends {gtype}")
    return lines


def make_netlist(gates, corner: dict, widths: dict, vectors, model_include: str,
                 title: str) -> str:
    """Complete transistor-level parity-checker netlist.

    One vector: the input sources are set to that vector and every net gets a
    .nodeset at its expected logic level. This is only the Newton starting
    point (the converged solution is the same to within reltol); without it
    nearly every operating point needs slow dynamic gmin stepping, and a few
    Part B gates fell through to ngspice's transient-op fallback.
    Several vectors: one session that steps through them with `alter` (used
    for the example deliverable netlist, which needs no per-vector nodeset).
    """
    vdd = corner["vdd"]
    nets = [g[3] for g in gates]
    lines = [
        f"* {title}",
        f'.include "{model_include}"',
        ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
        f".temp {corner['temp']:.12g}",
        "",
    ]
    for line in subckt_definitions({g[1] for g in gates}):
        lines.append(
            line.replace("{L}", f"{gv.LMIN:.12g}")
                .replace("{W_nmos}", f"{widths['nmos']:.12g}")
                .replace("{W_pmos}", f"{widths['pmos']:.12g}")
        )
    lines += ["", f"VDD vdd 0 DC {vdd:.12g}"]
    first = vector_bits(vectors[0])
    for d in INPUTS:
        lines.append(f"V{d} {d} 0 DC {first[d] * vdd:.12g}")
    lines.append("")
    for inst, gtype, ins, out in gates:
        lines.append(f"X{inst} {' '.join(ins)} {out} vdd {gtype}")
    if len(vectors) == 1:
        expected, _ = logic_simulate(gates, first)
        lines.append("")
        lines += [f".nodeset v({n})={expected[n] * vdd:.12g}" for n in nets]

    lines += ["", ".control", "set numdgt=10"]
    for position, k in enumerate(vectors):
        if position:
            bits = vector_bits(k)
            lines += [f"alter V{d} dc = {bits[d] * vdd:.12g}" for d in INPUTS]
        lines += [
            "op",
            f"echo @@VEC {k}",
            "print i(VDD)",
            "print " + " ".join(f"i(V{d})" for d in INPUTS),
            "print " + " ".join(f"v({n})" for n in nets),
            "destroy all",
        ]
    lines += ["quit", ".endc", "", ".end", ""]
    return "\n".join(lines)


NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def parse_value(block: str, name: str) -> float:
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*=\s*{NUMBER}", block)
    if not match:
        raise RuntimeError(f"Missing {name} in ngspice output block:\n{block[:2000]}")
    return float(match.group(1))


SOLVE_PATHS = ("direct", "gmin", "source", "transient_op", "failed")


def _solve_path(output: str) -> str:
    text = output.lower()
    if "dc solution failed" in text or "@@vec" not in text:
        return "failed"
    if "transient op started" in text:
        return "transient_op"
    if "source stepping completed" in text:
        return "source"
    if "gmin stepping completed" in text:
        return "gmin"
    return "direct"


def run_vector(gates, corner, widths, index, k, model_path, ngspice) -> dict:
    """Simulate one input vector at one corner (its own ngspice session).

    Attempts, in order, stopping at the first that converges without the
    transient-op fallback (whose result need not be a true DC point):
      1. logic-level nodesets
      2. no nodesets
      3. no nodesets, higher DC iteration limits (itl1/itl6)
      4. nodesets, higher DC iteration limits
    The attempt that converged by the most direct path is kept. If every
    attempt fails (seen once with ngspice-47), the point is recorded with
    solve_path = "failed" and NaN results, and left out of the metrics,
    instead of stopping the whole run.
    """
    netlist_path = WORK_DIR / f"parity_corner{index}_vec{k}.cir"
    netlist = make_netlist(gates, corner, widths, [k], model_path.resolve().as_posix(),
                           f"8-bit parity checker, corner {index}, vector {vector_label(k)}")
    plain = re.sub(r"(?m)^\.nodeset.*\n", "", netlist)
    more_iterations = ".options itl1=1000 itl6=1000\n.options gmin"
    attempts = {
        "nodeset": netlist,
        "no_nodeset": plain,
        "no_nodeset_itl": plain.replace(".options gmin", more_iterations, 1),
        "nodeset_itl": netlist.replace(".options gmin", more_iterations, 1),
    }
    best = None
    for name, text in attempts.items():
        netlist_path.write_text(text, encoding="utf-8")
        try:
            completed = subprocess.run(
                [ngspice, "-b", str(netlist_path)],
                capture_output=True, text=True, timeout=900, check=False,
            )
            output = completed.stdout + "\n" + completed.stderr
            path = "failed" if completed.returncode != 0 else _solve_path(output)
        except subprocess.TimeoutExpired:
            output, path = "timeout", "failed"
        if path != "failed":
            try:  # a run that printed no usable values counts as failed
                parse_value(output, "i(vdd)")
                parse_value(output, f"v({OUTPUT_NET})")
            except RuntimeError:
                path = "failed"
        if best is None or SOLVE_PATHS.index(path) < SOLVE_PATHS.index(best[1]):
            best = (name, path, output)
        if path in ("direct", "gmin", "source"):
            break
    attempt, path, output = best
    if path == "failed":
        print(f"WARNING: ngspice could not solve corner {index} vector {vector_label(k)} "
              f"with any attempt; recorded as failed and excluded from the metrics "
              f"(netlist kept: {netlist_path})", flush=True)
        return {"corner": index, "vector": k, "spice_leakage": np.nan,
                "spice_ivdd_signed": np.nan, "spice_input_current": np.nan,
                "spice_out_V": np.nan, "logic_ok": False,
                "max_rail_deviation_V": np.nan, "max_beyond_rail_V": np.nan,
                "solve_attempt": "all", "solve_path": "failed",
                "transient_op_fallback": False, "gmin_stepping": False}
    netlist_path.unlink()

    vdd = corner["vdd"]
    expected, _ = logic_simulate(gates, vector_bits(k))
    deviation, logic_ok, outside = 0.0, True, 0.0
    for _, _, _, net in gates:
        v = parse_value(output, f"v({net})")
        deviation = max(deviation, abs(v - expected[net] * vdd))
        outside = max(outside, v - vdd, -v)
        logic_ok &= (v > vdd / 2) == bool(expected[net])
    supply = parse_value(output, "i(vdd)")
    return {
        "corner": index,
        "vector": k,
        "spice_leakage": abs(supply),
        # ngspice reports a delivering supply as negative current; a positive
        # value means the point is not a physical DC solution.
        "spice_ivdd_signed": supply,
        "spice_input_current": sum(abs(parse_value(output, f"i(v{d})")) for d in INPUTS),
        "spice_out_V": parse_value(output, f"v({OUTPUT_NET})"),
        "logic_ok": bool(logic_ok),
        "max_rail_deviation_V": deviation,
        "max_beyond_rail_V": outside,
        "solve_attempt": attempt,
        "solve_path": path,
        "transient_op_fallback": path == "transient_op",
        "gmin_stepping": path in ("gmin", "source"),
    }


def run_corner(job: dict, ngspice: str) -> list[dict]:
    """All vectors of one corner; results are saved per corner so an
    interrupted run resumes where it stopped."""
    index = job["corner"]
    done_file = WORK_DIR / "corners" / f"corner_{index:03d}.csv"
    if done_file.exists():
        return pd.read_csv(done_file).to_dict("records")
    done_file.parent.mkdir(parents=True, exist_ok=True)
    model_path = WORK_DIR / f"parity_corner{index}.pm"
    model_path.write_text(job["model_text"], encoding="utf-8")
    rows = [run_vector(job["gates"], job["corner_values"], job["widths"], index, k,
                       model_path, ngspice) for k in job["vectors"]]
    model_path.unlink()
    pd.DataFrame(rows).to_csv(done_file, index=False)
    return rows


def simulate_all(gates, corners, widths, ngspice, workers) -> pd.DataFrame:
    nominal_text = gv.MODEL_FILE.read_text(encoding="utf-8")
    vectors = list(range(2 ** N_BITS))
    jobs = [{
        "corner": i,
        "corner_values": c,
        "gates": gates,
        "widths": widths,
        "vectors": vectors,
        "model_text": gv.modified_model(nominal_text, {"nmos": c["nmos"], "pmos": c["pmos"]}),
    } for i, c in enumerate(corners)]
    print(f"Running {len(jobs)} corners x {len(vectors)} vectors "
          f"({len(jobs) * len(vectors)} SPICE runs) on {workers} threads...")
    start = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(lambda j: run_corner(j, ngspice), jobs), 1):
            rows += result
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} corners  ({(time.time() - start) / 60:.1f} min)",
                      flush=True)
    return pd.DataFrame(rows)


def write_example_netlist(gates, widths, style: str) -> Path:
    """Deliverable netlist: nominal model card, VDD = 0.9 V, 27 C, all 256
    vectors. Run from the project root:  ngspice_con.exe -b netlists/parity8.cir"""
    NETLIST_DIR.mkdir(parents=True, exist_ok=True)
    path = NETLIST_DIR / f"parity8{'' if style == 'nand' else '_' + style}.cir"
    nominal = {"vdd": gv.VDD_NOMINAL, "temp": 27.0}
    path.write_text(
        make_netlist(gates, nominal, widths, list(range(2 ** N_BITS)),
                     "models/16nm_LP.pm",
                     f"8-bit parity checker (XOR tree, {style} XOR cells), "
                     "16nm LP nominal corner, all 256 input vectors"),
        encoding="utf-8",
    )
    return path


# ============================================================
# GATE MODELS: SUPPLY LEAKAGE AND INPUT-PIN CURRENTS
# ============================================================
#
# In an isolated Part B gate, |I(VDD)| is the gate leakage. Inside a circuit,
# the current drawn by a gate input that sits on an internal net at logic 1
# (mostly gate-oxide tunnelling into ON NMOS / OFF PMOS gates, large in this
# 16nm LP card) is supplied from VDD through the driving gate's ON pull-up.
# In the isolated gate it came from the ideal input source and so is missing
# from |I(VDD)|. A gate model therefore returns two things:
#   * supply leakage |I(VDD)|                         (Part B quantity)
#   * the current drawn into each input pin           (from the predicted ig
#                                                      of the devices on it)
# Two hierarchical predictions are reported:
#   plain_sum     sum of gate supply leakages (literal Part C spec)
#   with_loading  plain_sum + pin currents of gate inputs on internal nets at 1
# Primary inputs are ideal sources in the full netlist, so their pin currents
# are not added (they are not part of |I(VDD)| there either).

PIN_CACHE_NAME = "gate_pin_currents_spice.csv"


def read_part_b_spice(corners, widths, gate_types) -> pd.DataFrame:
    spice = pd.read_csv(gv.SPICE_CACHE, dtype={"inputs": str})
    settings = spice[["seed", "wn", "wp"]].drop_duplicates()
    expected = (gv.CORNER_SEED, widths["nmos"], widths["pmos"])
    if len(settings) != 1 or not np.allclose(settings.iloc[0].to_numpy(float), expected):
        raise ValueError("Part B SPICE cache was made with a different seed or widths")
    if spice["corner"].max() + 1 != len(corners):
        raise ValueError("Part B SPICE cache has a different corner count; "
                         "re-run gate_validation.py with the same --corners")
    spice = spice[spice["gate"].isin(gate_types)].drop(columns=["seed", "wn", "wp"])
    return spice.reset_index(drop=True)


def simulate_gate_pin_currents(corners, widths, gate_types, ngspice, workers) -> pd.DataFrame:
    """Re-simulate the Part B gate netlists (identical, from gate_validation)
    and also record the input-source currents. Used only for the SPICE
    composition check; the ML prediction never uses these values."""
    nominal_text = gv.MODEL_FILE.read_text(encoding="utf-8")
    work = WORK_DIR / "gate_pins"
    work.mkdir(parents=True, exist_ok=True)

    def run(index):
        corner = corners[index]
        model = work / f"corner{index}.pm"
        model.write_text(gv.modified_model(
            nominal_text, {"nmos": corner["nmos"], "pmos": corner["pmos"]}), encoding="utf-8")
        rows = []
        for gate in sorted(gate_types):
            for bits in gv.GATES[gate]["inputs"]:
                netlist = gv.make_netlist(gate, bits, corner, widths).replace(
                    "{MODEL}", model.resolve().as_posix())
                sources = ["VA"] + (["VB"] if len(bits) == 2 else [])
                netlist = netlist.replace(
                    "print i(VDD)", "print i(VDD)\n" + "\n".join(f"print i({v})" for v in sources))
                path = work / f"{gate}_{bits}_{index}.cir"
                path.write_text(netlist, encoding="utf-8")
                done = subprocess.run([ngspice, "-b", str(path)], capture_output=True,
                                      text=True, timeout=900, check=False)
                output = done.stdout + "\n" + done.stderr
                path.unlink()
                row = {"gate": gate, "inputs": bits, "corner": index,
                       "spice_leakage_rerun": abs(parse_value(output, "i(vdd)")),
                       "transient_op_fallback": "transient op started" in output.lower()}
                for pin, source in zip(("a", "b"), sources):
                    # current drawn INTO the gate's input pin from the net
                    row[f"spice_pin_{pin}"] = -parse_value(output, f"i({source.lower()})")
                rows.append(row)
        model.unlink()
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for result in pool.map(run, range(len(corners))) for r in result]
    return pd.DataFrame(rows)


def ml_pin_currents(spice: pd.DataFrame, corners, widths, packages) -> pd.DataFrame:
    """Predicted current into each input pin = sum of the predicted ig of the
    transistors whose gate is that pin (same device rows as Part B)."""
    devices = gv.predict_devices(gv.device_rows(spice, corners, widths), packages)
    gate_node = {(g, t[0]): t[3] for g, spec in gv.GATES.items() for t in spec["transistors"]}
    devices["pin"] = [gate_node[(g, d)] for g, d in zip(devices["gate"], devices["device"])]
    devices = devices[devices["pin"].isin(["a", "b"])]
    pins = devices.pivot_table(index=["gate", "inputs", "corner"], columns="pin",
                               values="ig", aggfunc="sum").reset_index()
    return pins.rename(columns={"a": "ml_pin_a", "b": "ml_pin_b"})


def gate_model_tables(spice_gates: pd.DataFrame, pins_spice: pd.DataFrame,
                      corners, widths, packages, models_dir):
    """Per (gate, inputs): arrays over corners of supply leakage and pin
    currents, for the ML gate models and for the SPICE gates."""
    table = spice_gates.copy()
    # Gate models = Part B procedure (transistor models + KCL).
    table["ml_leakage"] = gv.ml_leakage(table, corners, widths, packages).to_numpy()

    part_b = gv.RESULT_DIR / "gate_validation_raw.csv"
    if part_b.exists() and Path(models_dir).resolve() == gv.SAVED_MODEL_DIR.resolve():
        ref = pd.read_csv(part_b, dtype={"inputs": str})
        merged = table.merge(ref[["gate", "inputs", "corner", "ml_leakage"]],
                             on=["gate", "inputs", "corner"], suffixes=("", "_partb"))
        diff = relative_error_percent(merged["ml_leakage_partb"], merged["ml_leakage"])
        print(f"Gate models vs Part B results: max difference {diff.max():.2e} % "
              f"over {len(merged)} gate cases")

    table = table.merge(ml_pin_currents(table, corners, widths, packages),
                        on=["gate", "inputs", "corner"], how="left")
    table = table.merge(pins_spice, on=["gate", "inputs", "corner"], how="left")
    rerun = relative_error_percent(table["spice_leakage"], table["spice_leakage_rerun"])
    print(f"Gate pin-current re-simulation reproduces Part B |I(VDD)| to "
          f"{np.median(rerun):.2e} % median, {rerun.max():.2e} % max")
    table.to_csv(RESULT_DIR / "gate_model_table.csv", index=False)

    out = {}
    for (gate, bits), group in table.groupby(["gate", "inputs"]):
        group = group.sort_values("corner")
        entry = {}
        for source in ("ml", "spice"):
            entry[f"{source}_supply"] = group[f"{source}_leakage"].to_numpy()
            for pin in ("a", "b"):
                column = f"{source}_pin_{pin}"
                entry[f"{source}_pin_{pin}"] = (group[column].fillna(0.0).to_numpy()
                                                if column in group else np.zeros(len(group)))
        out[(gate, bits)] = entry
    return out


def hierarchical_leakage(gates, tables, n_corners):
    """Compose the circuit from gate models; arrays of shape (256, corners)."""
    keys = ("ml_plain", "ml_loaded", "spice_plain", "spice_loaded")
    out = {key: np.zeros((2 ** N_BITS, n_corners)) for key in keys}
    usage = []
    for k in range(2 ** N_BITS):
        net, gate_inputs = logic_simulate(gates, vector_bits(k))
        for inst, gtype, ins, _ in gates:
            entry = tables[(gtype, gate_inputs[inst])]
            usage.append((gtype, gate_inputs[inst]))
            for source in ("ml", "spice"):
                out[f"{source}_plain"][k] += entry[f"{source}_supply"]
                out[f"{source}_loaded"][k] += entry[f"{source}_supply"]
                for pin, node in zip(("a", "b"), ins):
                    if node not in INPUTS and net[node] == 1:
                        out[f"{source}_loaded"][k] += entry[f"{source}_pin_{pin}"]
    counts = (pd.DataFrame(usage, columns=["gate", "inputs"])
              .value_counts().rename("gate_instances_over_256_vectors").reset_index())
    return out, counts


# ============================================================
# METRICS AND PLOTS
# ============================================================

PREDICTIONS = {
    "ml_with_loading": "ML hierarchical, gate leakage + input loading",
    "ml_plain_sum": "ML hierarchical, plain sum of gate leakages (literal spec)",
    "spice_gates_with_loading": "SPICE gates + input loading (composition check)",
    "spice_gates_plain_sum": "SPICE gates, plain sum (composition check)",
}


def summarise(group: pd.DataFrame, prediction: str, reference: str = "spice_leakage") -> dict:
    error = relative_error_percent(group[reference], group[prediction])
    return {
        "spice_median_A": group[reference].median(),
        "ml_median_A": group[prediction].median(),
        "bias_A": (group[prediction] - group[reference]).median(),
        "median_error_percent": np.median(error),
        "p95_error_percent": np.percentile(error, 95),
        "mean_accuracy_percent": np.clip(100 - error, 0, 100).mean(),
    }


def per_vector_metrics(raw: pd.DataFrame, prediction: str) -> pd.DataFrame:
    rows = []
    for k, group in raw.groupby("vector", sort=True):
        rows.append({
            "vector_d7_d0": vector_label(k),
            "vector": k,
            "ones": bin(k).count("1"),
            "parity": bin(k).count("1") % 2,
            "corners": len(group),
            **summarise(group, prediction),
            "spice_flagged_points": int(group["spice_flagged"].sum()),
        })
    return pd.DataFrame(rows)


def overall_row(label: str, raw: pd.DataFrame, prediction: str,
                reference: str = "spice_leakage") -> dict:
    error = relative_error_percent(raw[reference], raw[prediction])
    accuracy = np.clip(100 - error, 0, 100)
    return {
        "prediction": label,
        "reference": reference,
        "cases": len(raw),
        **summarise(raw, prediction, reference),
        "median_accuracy_percent": np.median(accuracy),
        "within_1_percent": (error <= 1).mean() * 100,
        "within_5_percent": (error <= 5).mean() * 100,
        "within_10_percent": (error <= 10).mean() * 100,
        "max_error_percent": error.max(),
    }


def overall_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = [overall_row(label, raw, column) for column, label in PREDICTIONS.items()]
    clean = raw[~raw["spice_flagged"]]
    if len(clean) < len(raw):
        rows.append(overall_row(PREDICTIONS["ml_with_loading"]
                                + ", excluding flagged SPICE points", clean, "ml_with_loading"))
    # ML against the composed SPICE gates isolates the gate-model error.
    rows.append(overall_row("ML gate models vs SPICE gates (model error only, with loading)",
                            raw, "ml_with_loading", "spice_gates_with_loading"))
    return pd.DataFrame(rows)


BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def parity_plot(raw: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    series = (("ml_plain_sum", ORANGE, "ML, plain sum of gate leakages (spec)"),
              ("ml_with_loading", BLUE, "ML, gate leakage + input loading"))
    for column, color, label in series:
        error = np.median(relative_error_percent(raw["spice_leakage"], raw[column]))
        ax.scatter(raw["spice_leakage"], raw[column], s=3, alpha=0.2, color=color,
                   edgecolors="none", rasterized=True,
                   label=f"{label} (median error {error:.1f} %)")
    flagged = raw[raw["spice_flagged"]]
    if len(flagged):
        ax.scatter(flagged["spice_leakage"], flagged["ml_with_loading"], s=30,
                   facecolors="none", edgecolors=INK, linewidths=0.8,
                   label=f"SPICE transient-op fallback ({len(flagged)} points)")
    values = raw[["spice_leakage", "ml_plain_sum", "ml_with_loading"]].to_numpy()
    lo, hi = values.min() / 1.3, values.max() * 1.3
    line = np.array([lo, hi])
    ax.plot(line, line, "--", color=INK, lw=1.2, label="Ideal")
    ax.plot(line, line * 1.1, ":", color=MUTED, lw=1, label="±10 %")
    ax.plot(line, line / 1.1, ":", color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("SPICE circuit leakage |I(VDD)| (A)")
    ax.set_ylabel("ML hierarchical leakage (A)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", markerscale=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "circuit_validation_parity.png", dpi=300)
    plt.close(fig)


def error_analysis_plot(raw: pd.DataFrame) -> None:
    """Per-corner median error (over the 256 vectors) vs the corner variables,
    for the ML prediction with input loading and for its gate-model part
    alone (ML vs composed SPICE gates)."""
    raw = raw.assign(
        e_ml=relative_error_percent(raw["spice_leakage"], raw["ml_with_loading"]),
        e_model=relative_error_percent(raw["spice_gates_with_loading"], raw["ml_with_loading"]),
    )
    per_corner = raw.groupby("corner").agg(
        ml=("e_ml", "median"), model=("e_model", "median"),
        vdd=("vdd", "first"), temp=("temp", "first"),
        n_vth0=("n_vth0", "first"), p_vth0=("p_vth0", "first"),
    )
    panels = [("temp", "Temperature (°C)"), ("vdd", "VDD (V)"),
              ("n_vth0", "NMOS vth0 (V)"), ("p_vth0", "PMOS vth0 (V)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (column, label) in zip(axes.flat, panels):
        ax.scatter(per_corner[column], per_corner["ml"], s=16, color=BLUE,
                   label="ML (with loading) vs full SPICE")
        ax.scatter(per_corner[column], per_corner["model"], s=10, color=ORANGE,
                   marker="s", label="ML vs composed SPICE gates (gate-model error)")
        ax.set_xlabel(label)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Median error over 256 vectors (%, log)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("8-bit parity checker: per-corner median leakage error")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "circuit_error_analysis.png", dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corners", type=int, default=200)
    parser.add_argument("--ngspice", default="ngspice_con.exe")
    parser.add_argument("--wn", type=float, default=gv.DEFAULT_WN, help="NMOS width (m)")
    parser.add_argument("--wp", type=float, default=gv.DEFAULT_WP, help="PMOS width (m)")
    parser.add_argument("--xor-style", choices=("nand", "andor"), default="nand")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-spice", action="store_true",
                        help="Load circuit SPICE results from the cache instead of simulating")
    parser.add_argument("--models-dir", type=Path, default=gv.SAVED_MODEL_DIR)
    parser.add_argument("--spice-only", action="store_true")
    args = parser.parse_args()

    global RESULT_DIR
    if args.xor_style != "nand":
        RESULT_DIR = RESULT_DIR.parent / f"circuit_validation_{args.xor_style}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cache = RESULT_DIR / "circuit_spice_raw.csv"
    pin_cache = RESULT_DIR / PIN_CACHE_NAME

    widths = {"nmos": args.wn, "pmos": args.wp}
    gates = parity_circuit(args.xor_style)
    gate_types = {g[1] for g in gates}
    nominal_text = gv.MODEL_FILE.read_text(encoding="utf-8")
    nominal = {t: gv.parse_model_parameters(nominal_text, t) for t in ("nmos", "pmos")}
    corners = gv.draw_corners(args.corners, nominal)   # same corners as Part B
    example = write_example_netlist(gates, widths, args.xor_style)
    print(f"Circuit: {len(gates)} gates "
          f"({', '.join(f'{n} x {t}' for t, n in pd.Series([g[1] for g in gates]).value_counts().items())}), "
          f"{sum(len(gv.GATES[g[1]]['transistors']) for g in gates)} transistors. "
          f"Example netlist: {example.relative_to(ROOT)}")

    if args.reuse_spice:
        spice = pd.read_csv(cache)
        settings = spice[["seed", "wn", "wp"]].drop_duplicates()
        if len(settings) != 1 or not np.allclose(
                settings.iloc[0].to_numpy(float), (gv.CORNER_SEED, args.wn, args.wp)):
            raise ValueError("Cached circuit SPICE results use a different seed or widths")
        if spice["corner"].max() + 1 != args.corners:
            raise ValueError("Cached circuit SPICE results have a different corner count")
        spice = spice.drop(columns=["seed", "wn", "wp"])
    else:
        spice = simulate_all(gates, corners, widths, args.ngspice, args.workers)
        spice.assign(seed=gv.CORNER_SEED, wn=args.wn, wp=args.wp).to_csv(cache, index=False)
    spice = spice.sort_values(["corner", "vector"]).reset_index(drop=True)
    failed = spice[spice["solve_path"] == "failed"]
    if len(failed):
        print(f"WARNING: {len(failed)} SPICE point(s) could not be solved by ngspice and are "
              "excluded from all metrics:\n"
              + failed[["corner", "vector"]].to_string(index=False))
        spice = spice[spice["solve_path"] != "failed"].reset_index(drop=True)

    if args.reuse_spice and pin_cache.exists():
        pins_spice = pd.read_csv(pin_cache, dtype={"inputs": str})
    else:
        print("Re-simulating the Part B gates with input-pin currents (composition check)...")
        pins_spice = simulate_gate_pin_currents(corners, widths, gate_types,
                                                args.ngspice, args.workers)
        pins_spice.to_csv(pin_cache, index=False)

    spice["logic_ok"] = spice["logic_ok"].astype(bool)
    spice["transient_op_fallback"] = spice["transient_op_fallback"].astype(bool)
    bad = spice[~spice["logic_ok"]]
    print(f"SPICE logic check: {len(spice) - len(bad)}/{len(spice)} operating points "
          f"have every net at the expected logic level; worst deviation from a rail "
          f"{spice['max_rail_deviation_V'].max() * 1e3:.3f} mV")
    if len(bad):
        print("WARNING: logic failures at\n", bad.head(20).to_string(index=False))
    spice["spice_flagged"] = spice["transient_op_fallback"] | (spice["spice_ivdd_signed"] > 0)
    print("SPICE solve paths: " + ", ".join(
        f"{p} {n}" for p, n in spice["solve_path"].value_counts().items())
        + f"; retried without nodeset: {int((spice['solve_attempt'] == 'no_nodeset').sum())}")
    print(f"SPICE convergence check: {int(spice['transient_op_fallback'].sum())} used the "
          f"transient-op fallback, {int((spice['spice_ivdd_signed'] > 0).sum())} have a "
          f"supply absorbing current; largest excursion beyond a rail "
          f"{spice['max_beyond_rail_V'].max() * 1e3:.4f} mV")
    if args.spice_only:
        return

    packages = gv.load_packages(args.models_dir)
    spice_gates = read_part_b_spice(corners, widths, gate_types)
    tables = gate_model_tables(spice_gates, pins_spice, corners, widths, packages,
                               args.models_dir)
    sums, usage = hierarchical_leakage(gates, tables, args.corners)
    usage.to_csv(RESULT_DIR / "gate_input_usage.csv", index=False)

    raw = spice.copy()
    raw["vector_d7_d0"] = raw["vector"].map(vector_label)
    index = (raw["vector"].to_numpy(), raw["corner"].to_numpy())
    raw["ml_with_loading"] = sums["ml_loaded"][index]
    raw["ml_plain_sum"] = sums["ml_plain"][index]
    raw["spice_gates_with_loading"] = sums["spice_loaded"][index]
    raw["spice_gates_plain_sum"] = sums["spice_plain"][index]
    raw["error_percent"] = relative_error_percent(raw["spice_leakage"], raw["ml_with_loading"])
    raw["error_percent_plain_sum"] = relative_error_percent(
        raw["spice_leakage"], raw["ml_plain_sum"])
    corner_table = pd.read_csv(gv.RESULT_DIR / "gate_corners.csv")
    check = np.array([[c["vdd"], c["temp"]] for c in corners])
    if not np.allclose(corner_table[["vdd", "temp"]].to_numpy()[:len(check)], check):
        raise ValueError("Corners differ from Part B gate_corners.csv")
    raw = raw.merge(corner_table[["corner", "vdd", "temp", "n_vth0", "p_vth0"]], on="corner")
    raw.to_csv(RESULT_DIR / "circuit_validation_raw.csv", index=False)

    tables_out = {}
    for column, name in (("ml_with_loading", "with_loading"), ("ml_plain_sum", "plain_sum")):
        per_vector = per_vector_metrics(raw, column)
        per_vector.to_csv(RESULT_DIR / f"circuit_metrics_per_vector_{name}.csv", index=False)
        tables_out[name] = per_vector
    by_ones = []
    for name, per_vector in tables_out.items():
        grouped = per_vector.groupby("ones").agg(
            vectors=("vector", "size"),
            spice_median_A=("spice_median_A", "median"),
            ml_median_A=("ml_median_A", "median"),
            median_error_percent=("median_error_percent", "median"),
            worst_vector_p95_error_percent=("p95_error_percent", "max"),
            mean_accuracy_percent=("mean_accuracy_percent", "mean"),
        ).reset_index()
        by_ones.append(grouped.assign(prediction=name))
    by_ones = pd.concat(by_ones, ignore_index=True)
    by_ones.to_csv(RESULT_DIR / "circuit_metrics_by_ones.csv", index=False)
    summary = overall_summary(raw)
    summary.to_csv(RESULT_DIR / "circuit_validation_summary.csv", index=False)

    parity_plot(raw, f"8-bit parity checker ({args.xor_style} XOR): ML vs SPICE leakage")
    error_analysis_plot(raw)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    ranked = tables_out["with_loading"].sort_values("median_error_percent")
    print("\nPer-vector metrics, ML with input loading (5 best and 5 worst):")
    print(pd.concat([ranked.head(5), ranked.tail(5)]).to_string(index=False))
    print("\nBy number of ones in the input vector:")
    print(by_ones.to_string(index=False))
    print("\nOVERALL")
    print(summary.set_index("prediction").T.to_string())


if __name__ == "__main__":
    main()
