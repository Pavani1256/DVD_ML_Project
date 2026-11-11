"""Simulate every sampled transistor in data/samples with ngspice.

Writes data/processed/{case}.csv (inputs + id, ig, is, ib), a Kirchhoff
residual log and a failure log per case. Simulations run in parallel worker
processes, each with its own scratch directory, and rows are written in the
original sample order, so the output does not depend on the worker count.

    python .\\src\\spice\\run_dataset.py [--workers N] [--suffix _edge]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from run_one_sample import make_netlist, run_ngspice


ROOT = Path(__file__).resolve().parents[2]

CASES = {
    "nmos_on": "nmos",
    "nmos_off": "nmos",
    "pmos_on": "pmos",
    "pmos_off": "pmos",
}

INPUT_DIR = ROOT / "data" / "samples"
OUTPUT_DIR = ROOT / "data" / "processed"
SCRATCH_DIR = ROOT / "data" / "spice_work"

N_EXPECTED = 10_000
CHUNK_SIZE = 50

INPUT_COLUMNS = [
    "vdd", "temp", "w", "vd", "vs", "vg", "vb",
    "vth0", "toxe", "ndep", "u0", "xj"
]

CURRENT_COLUMNS = ["id", "ig", "is", "ib"]
OUTPUT_COLUMNS = INPUT_COLUMNS + CURRENT_COLUMNS


def simulate(job):
    """Run one sample in this worker's own scratch directory."""

    index, row, transistor = job
    work_dir = SCRATCH_DIR / f"worker_{os.getpid()}"

    try:
        netlist = make_netlist(pd.Series(row), transistor, work_dir=work_dir)
        currents, _ = run_ngspice(netlist)
        return index, currents, None
    except Exception as error:  # recorded in the failure log
        return index, None, str(error)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument(
        "--suffix", default="",
        help="Dataset suffix, e.g. _edge for the supplementary rail-edge samples"
    )
    args = parser.parse_args()
    suffix = args.suffix

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite existing full-run outputs.
    for case in CASES:
        files = [
            OUTPUT_DIR / f"{case}{suffix}.csv",
            OUTPUT_DIR / f"{case}{suffix}_kcl_checks.csv",
            OUTPUT_DIR / f"{case}{suffix}_failures.csv",
        ]

        existing = [path.name for path in files if path.exists()]

        if existing:
            raise FileExistsError(
                f"Existing output files for {case}: {existing}. "
                "Inspect or move them before starting a new run."
            )

    total_success = 0
    total_failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for case, transistor in CASES.items():

            df = pd.read_csv(INPUT_DIR / f"{case}{suffix}_samples.csv")

            if not suffix and len(df) != N_EXPECTED:
                raise ValueError(
                    f"{case}: expected {N_EXPECTED} samples, found {len(df)}"
                )

            print(f"\n========== {case.upper()} ==========")
            print(f"Starting {len(df)} simulations on {args.workers} workers...")

            jobs = [
                (index, row[INPUT_COLUMNS].to_dict(), transistor)
                for index, row in df.iterrows()
            ]

            results = {}
            failures = []
            start_time = time.time()

            for done, (index, currents, error) in enumerate(
                pool.map(simulate, jobs, chunksize=CHUNK_SIZE), start=1
            ):
                if error is None:
                    results[index] = currents
                else:
                    failures.append({"sample_index": index, "error": error})

                if done % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"{done}/{len(df)} | failed: {len(failures)} | "
                        f"{elapsed / 60:.1f} min"
                    )

            rows = []
            kcl_rows = []

            for index in sorted(results):
                currents = results[index]
                rows.append({
                    **df.loc[index, INPUT_COLUMNS].to_dict(),
                    **currents,
                })
                kcl_rows.append({
                    "sample_index": index,
                    "kcl_residual": sum(currents.values()),
                })

            pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
                OUTPUT_DIR / f"{case}{suffix}.csv", index=False
            )
            pd.DataFrame(kcl_rows).to_csv(
                OUTPUT_DIR / f"{case}{suffix}_kcl_checks.csv", index=False
            )
            if failures:
                pd.DataFrame(failures).to_csv(
                    OUTPUT_DIR / f"{case}{suffix}_failures.csv", index=False
                )

            print(f"Finished {case}: {len(rows)} ok, {len(failures)} failed, "
                  f"{(time.time() - start_time) / 60:.2f} min")

            total_success += len(rows)
            total_failed += len(failures)

    print("\n========== COMPLETE SUMMARY ==========")
    print(f"Total successful simulations: {total_success}")
    print(f"Total failed simulations: {total_failed}")


if __name__ == "__main__":
    main()
