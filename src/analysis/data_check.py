"""Data check deliverable: Kirchhoff check per case and three trend plots.

    python .\\src\\analysis\\data_check.py

Outputs in results/data_check/:
  kcl_summary.csv              |id + ig + is + ib| per case, absolute and
                               relative to the largest terminal current
  trend_off_current_vs_temp.png
  trend_off_current_vs_vth0.png
  trend_gate_current_vs_toxe.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "data_check"
CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
TARGETS = ["id", "ig", "is", "ib"]


def kcl_summary(data):
    rows = []
    for case, df in data.items():
        residual = df[TARGETS].sum(axis=1).abs()
        largest = df[TARGETS].abs().max(axis=1)
        relative = residual / largest.clip(lower=1e-30)
        rows.append({
            "case": case,
            "samples": len(df),
            "max_abs_residual_A": residual.max(),
            "median_abs_residual_A": residual.median(),
            "max_relative_residual": relative.max(),
            "median_relative_residual": relative.median(),
        })
    return pd.DataFrame(rows)


def off_leakage(df):
    """OFF-state channel leakage: the larger of |id| and |is|."""
    return np.maximum(df["id"].abs(), df["is"].abs())


def trend(data, cases, x, y_func, xlabel, ylabel, title, filename):
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 5))
    for ax, case in zip(np.atleast_1d(axes), cases):
        df = data[case]
        # Colour by |Vds| so the dependence on the swept parameter is visible.
        points = ax.scatter(df[x], y_func(df), c=(df["vd"] - df["vs"]).abs(),
                            s=4, alpha=0.5, cmap="viridis")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{case.upper()}: {title}")
        ax.grid(True, which="both", alpha=0.3)
        fig.colorbar(points, ax=ax, label="|Vds| (V)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {case: pd.read_csv(DATA_DIR / f"{case}.csv") for case in CASES}

    summary = kcl_summary(data)
    summary.to_csv(OUT_DIR / "kcl_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("Kirchhoff check (sum of the four terminal currents):")
    print(summary.to_string(index=False))

    trend(data, ("nmos_off", "pmos_off"), "temp", off_leakage,
          "Temperature (C)", "OFF current max(|Id|,|Is|) (A)",
          "OFF current vs temperature", "trend_off_current_vs_temp.png")
    trend(data, ("nmos_off", "pmos_off"), "vth0", off_leakage,
          "vth0 (V)", "OFF current max(|Id|,|Is|) (A)",
          "OFF current vs vth0", "trend_off_current_vs_vth0.png")
    trend(data, ("nmos_on", "pmos_on"), "toxe", lambda df: df["ig"].abs(),
          "toxe (m)", "|Ig| (A)",
          "gate current vs toxe", "trend_gate_current_vs_toxe.png")
    print(f"\nPlots saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
