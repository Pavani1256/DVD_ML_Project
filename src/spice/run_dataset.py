
from pathlib import Path
import time

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

BATCH_SIZE = 100
PROGRESS_INTERVAL = 100

INPUT_COLUMNS = [
    "vdd", "temp", "w", "vd", "vs", "vg", "vb",
    "vth0", "toxe", "ndep", "u0", "xj"
]

CURRENT_COLUMNS = ["id", "ig", "is", "ib"]
OUTPUT_COLUMNS = INPUT_COLUMNS + CURRENT_COLUMNS


def append_batch(rows, path):
    """Append successful simulation results, writing a header only once."""

    if not rows:
        return

    batch_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    batch_df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False
    )


def append_log(rows, path):
    """Append KCL or failure records, writing a header only once."""

    if not rows:
        return

    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False
    )


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite existing full-run outputs.
    for case in CASES:
        files = [
            OUTPUT_DIR / f"{case}.csv",
            OUTPUT_DIR / f"{case}.partial.csv",
            OUTPUT_DIR / f"{case}_kcl_checks.csv",
            OUTPUT_DIR / f"{case}_failures.csv",
        ]

        existing = [str(path.name) for path in files if path.exists()]

        if existing:
            raise FileExistsError(
                f"Existing output files for {case}: {existing}. "
                "Inspect or move them before starting a new run."
            )

    total_success = 0
    total_failed = 0

    for case, transistor in CASES.items():

        input_file = INPUT_DIR / f"{case}_samples.csv"
        partial_file = OUTPUT_DIR / f"{case}.partial.csv"
        final_file = OUTPUT_DIR / f"{case}.csv"

        kcl_file = OUTPUT_DIR / f"{case}_kcl_checks.csv"
        failure_file = OUTPUT_DIR / f"{case}_failures.csv"

        df = pd.read_csv(input_file)

        if len(df) != 10_000:
            raise ValueError(
                f"{case}: expected 10,000 samples, found {len(df)}"
            )

        print(f"\n========== {case.upper()} ==========")
        print(f"Starting {len(df)} simulations...")

        successful_batch = []
        kcl_batch = []
        failure_batch = []

        success_count = 0
        failure_count = 0

        start_time = time.time()

        for index, row in df.iterrows():

            try:
                netlist = make_netlist(row, transistor)
                currents, _ = run_ngspice(netlist)

                kcl = sum(currents.values())

                result = {
                    column: row[column]
                    for column in INPUT_COLUMNS
                }
                result.update(currents)

                successful_batch.append(result)

                kcl_batch.append({
                    "sample_index": index,
                    "kcl_residual": kcl
                })

                success_count += 1

            except Exception as error:

                failure_batch.append({
                    "sample_index": index,
                    "error": str(error)
                })

                failure_count += 1

            # Write every 100 attempted samples.
            if (index + 1) % BATCH_SIZE == 0:

                append_batch(successful_batch, partial_file)
                append_log(kcl_batch, kcl_file)
                append_log(failure_batch, failure_file)

                successful_batch.clear()
                kcl_batch.clear()
                failure_batch.clear()

            if (index + 1) % PROGRESS_INTERVAL == 0:

                elapsed = time.time() - start_time

                print(
                    f"{index + 1}/{len(df)} attempted | "
                    f"Success: {success_count} | "
                    f"Failed: {failure_count} | "
                    f"Elapsed: {elapsed / 60:.1f} min"
                )

        # Save the remaining records after the loop.
        append_batch(successful_batch, partial_file)
        append_log(kcl_batch, kcl_file)
        append_log(failure_batch, failure_file)

        # Rename the partial file only after this case finishes.
        if partial_file.exists():
            partial_file.replace(final_file)

        elapsed = time.time() - start_time

        print(f"\nFinished {case}.")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")
        print(f"Time: {elapsed / 60:.2f} minutes")

        if final_file.exists():
            print(f"Dataset saved: {final_file}")

        total_success += success_count
        total_failed += failure_count

    print("\n========== COMPLETE SUMMARY ==========")
    print(f"Total successful simulations: {total_success}")
    print(f"Total failed simulations: {total_failed}")
    print(f"Total attempted: {total_success + total_failed}")


if __name__ == "__main__":
    main()
