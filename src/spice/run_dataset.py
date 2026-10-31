from pathlib import Path

import pandas as pd

from run_one_sample import make_netlist, run_ngspice


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "samples" / "nmos_on_samples.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

CASE = "nmos_on"
TRANSISTOR = "nmos"

# First test only 10 samples.
LIMIT = 10


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE).head(LIMIT)

    results = []
    failures = []

    print(f"Running {len(df)} samples for {CASE}...\n")

    for index, row in df.iterrows():

        try:
            # Generate a netlist for this sample.
            netlist = make_netlist(row, TRANSISTOR)

            # Run ngspice and collect terminal currents.
            currents, _ = run_ngspice(netlist)

            # Calculate KCL residual.
            kcl = sum(currents.values())

            # Combine inputs and outputs.
            result = row.to_dict()
            result.update(currents)
            result["kcl_residual"] = kcl

            results.append(result)

            print(
                f"Sample {index + 1}/{len(df)} completed | "
                f"Id = {currents['id']:.6e} A | "
                f"KCL = {kcl:.3e} A"
            )

        except Exception as error:
            failures.append({
                "sample_index": index,
                "error": str(error)
            })

            print(f"Sample {index + 1} FAILED: {error}")

    # Save successful results.
    output_file = OUTPUT_DIR / f"{CASE}_test_10.csv"

    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False)

    # Save failures separately, if any.
    if failures:
        failure_file = OUTPUT_DIR / f"{CASE}_failures.csv"
        pd.DataFrame(failures).to_csv(failure_file, index=False)

    print("\n---------- SUMMARY ----------")
    print(f"Total samples attempted: {len(df)}")
    print(f"Successful simulations: {len(results)}")
    print(f"Failed simulations: {len(failures)}")

    if results:
        print(f"Output saved to: {output_file}")

        max_kcl = max(abs(item["kcl_residual"]) for item in results)
        print(f"Maximum absolute KCL residual: {max_kcl:.6e} A")


if __name__ == "__main__":
    main()