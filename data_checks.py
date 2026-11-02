
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

DATASETS = {
    "NMOS ON": "data/processed/nmos_on.csv",
    "NMOS OFF": "data/processed/nmos_off.csv",
    "PMOS ON": "data/processed/pmos_on.csv",
    "PMOS OFF": "data/processed/pmos_off.csv"
}

PLOT_DIR = "results/plots"
METRICS_DIR = "results/metrics"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

CURRENT_COLUMNS = ["id", "ig", "is", "ib"]

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

datasets = {}

for case, filepath in DATASETS.items():

    print(f"\nLoading {case}: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    required_columns = (
        CURRENT_COLUMNS + ["temp", "vth0", "toxe"]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{case} is missing columns: {missing}"
        )

    # Convert relevant columns to numeric
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[required_columns].isna().any().any():
        raise ValueError(
            f"{case} contains missing or non-numeric values "
            "in required columns."
        )

    datasets[case] = df

    print(f"Loaded {len(df)} samples.")


# --------------------------------------------------
# 3. KIRCHHOFF CURRENT LAW CHECK
# --------------------------------------------------

# KCL requires the algebraic sum of all four
# terminal currents to be approximately zero.

kcl_summaries = []

for case, df in datasets.items():

    print(f"\nKCL check: {case}")

    # Algebraic sum of terminal currents
    kcl_sum = df[CURRENT_COLUMNS].sum(axis=1)

    # Absolute KCL error
    abs_error = np.abs(kcl_sum)

    # Sum of magnitudes of terminal currents
    total_current = df[CURRENT_COLUMNS].abs().sum(axis=1)

    # Relative KCL error
    relative_error = np.divide(
        abs_error,
        total_current,
        out=np.zeros(len(df), dtype=float),
        where=total_current.to_numpy() != 0
    )

    # Save row-wise KCL results
    row_results = pd.DataFrame({
        "kcl_sum_A": kcl_sum,
        "absolute_kcl_error_A": abs_error,
        "relative_kcl_error": relative_error
    })

    row_results.to_csv(
        f"{METRICS_DIR}/kcl_{case.lower().replace(' ', '_')}.csv",
        index=False
    )

    # Summary statistics
    summary = {
        "case": case,
        "samples": len(df),
        "mean_absolute_error_A": abs_error.mean(),
        "median_absolute_error_A": abs_error.median(),
        "p95_absolute_error_A": abs_error.quantile(0.95),
        "maximum_absolute_error_A": abs_error.max(),
        "median_relative_error": np.median(relative_error),
        "p95_relative_error": np.quantile(relative_error, 0.95)
    }

    kcl_summaries.append(summary)

    print(f"Mean absolute KCL error: {abs_error.mean():.6e} A")
    print(f"Median absolute KCL error: {abs_error.median():.6e} A")
    print(f"95th percentile error: {abs_error.quantile(0.95):.6e} A")
    print(f"Maximum absolute error: {abs_error.max():.6e} A")


# Save combined KCL summary
kcl_summary_df = pd.DataFrame(kcl_summaries)

kcl_summary_df.to_csv(
    f"{METRICS_DIR}/kirchhoff_summary.csv",
    index=False
)

print("\nKCL summary:")
print(kcl_summary_df.to_string(index=False))


# --------------------------------------------------
# 4. HELPER FUNCTION FOR TREND PLOTS
# --------------------------------------------------

def plot_trend(ax, df, x_column, y_column, label):

    x = df[x_column].to_numpy()
    y = np.abs(df[y_column].to_numpy())

    # Scatter plot of actual samples
    ax.scatter(
        x, y,
        s=8,
        alpha=0.18,
        label=f"{label} samples"
    )

    # Divide x-axis into bins and calculate median current
    temp_df = pd.DataFrame({
        "x": x,
        "y": y
    })

    temp_df = temp_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Keep positive currents for logarithmic plotting
    temp_df = temp_df[temp_df["y"] > 0]

    if len(temp_df) > 1:

        temp_df["bin"] = pd.qcut(
            temp_df["x"],
            q=20,
            duplicates="drop"
        )

        trend = temp_df.groupby(
            "bin", observed=True
        ).agg(
            x_median=("x", "median"),
            y_median=("y", "median")
        )

        ax.plot(
            trend["x_median"],
            trend["y_median"],
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{label} binned median"
        )


# --------------------------------------------------
# 5. PLOT 1: OFF CURRENT VS TEMPERATURE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

for case in ["NMOS OFF", "PMOS OFF"]:

    df = datasets[case]

    plot_trend(
        ax,
        df,
        x_column="temp",
        y_column="id",
        label=case
    )

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Absolute drain current |Id| (A)")
ax.set_title("OFF Current vs Temperature")

ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/off_current_vs_temperature.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 6. PLOT 2: OFF CURRENT VS VTH0
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

for case in ["NMOS OFF", "PMOS OFF"]:

    df = datasets[case]

    plot_trend(
        ax,
        df,
        x_column="vth0",
        y_column="id",
        label=case
    )

ax.set_xlabel("Threshold voltage vth0 (V)")
ax.set_ylabel("Absolute drain current |Id| (A)")
ax.set_title("OFF Current vs Threshold Voltage")

ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/off_current_vs_vth0.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 7. PLOT 3: GATE CURRENT VS TOXE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

for case, df in datasets.items():

    plot_trend(
        ax,
        df,
        x_column="toxe",
        y_column="ig",
        label=case
    )

ax.set_xlabel("Oxide thickness toxe")
ax.set_ylabel("Absolute gate current |Ig| (A)")
ax.set_title("Gate Current vs Oxide Thickness")

ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/gate_current_vs_toxe.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 8. FINISHED
# --------------------------------------------------

print("\nAll checks and plots generated successfully!")

print(f"\nPlots saved in: {PLOT_DIR}")
print(f"KCL reports saved in: {METRICS_DIR}")
