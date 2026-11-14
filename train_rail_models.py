"""Train the operating-point sub-models and add them to the four saved models.

Run after train_models.py and src/spice/generate_rail_dataset.py:

    python .\\train_rail_models.py --jobs 8

For each case (nmos_on/off, pmos_on/off), configuration (exact-rail 00 01 10
11, one-rail m0 m1 0m 1m) and terminal, an MLP regresses log10|I| on VDD, T,
log W and the process parameters (plus the free node voltage for one-rail
configs); the sign is the training data's (unanimous) sign. 20 % of the
corners of every (case, config) are held out. A sub-model is kept only if,
on its held-out corners, the median error is < 10 % and the P95 error is
<= 50 %; otherwise the general model stays in charge of that terminal.
Skipped entirely:
  * terminals whose current is numerical noise (median |I| < 1e-16 A),
  * ON-device id/is: the gate procedure never uses them (Kirchhoff's law
    solves the ON-device channel currents).

The kept sub-models are stored in each saved package as
``package["model"].rail_models`` (used by TransistorModel.predict), so there
are still exactly four model files. Re-running train_models.py rewrites the
packages without them: run this script again afterwards.

Outputs:
  saved_models/{case}.joblib                        (updated in place)
  results/metrics/rail_submodels_metrics.csv        held-out error per sub-model
  results/metrics/rail_submodels_test_effect.csv    original test set, before/after
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from transistor_ml import (FEATURES, LOG_FLOOR, TARGETS, rail_config,  # noqa: E402
                           rail_features, relative_error_percent)

DATA = ROOT / "data" / "processed" / "rail_samples.csv.gz"
MODEL_DIR = ROOT / "saved_models"
METRIC_DIR = ROOT / "results" / "metrics"
CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
SPLIT_SEED = 1
NOISE = 1e-16          # A; below this SPICE terminal currents are rounding noise
MAX_MEDIAN = 10.0      # % held-out median error to keep a sub-model
MAX_P95 = 50.0         # % held-out P95 error to keep a sub-model


def fit_one(case, config, terminal, Z_train, y_train, Z_val, y_val, sign):
    hidden = (96, 96) if "m" in config else (64, 64)
    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=hidden, activation="tanh", alpha=1e-6, max_iter=3000,
        learning_rate_init=3e-3, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=60, tol=1e-7, random_state=0))
    model.fit(Z_train, np.log10(np.abs(y_train) + LOG_FLOOR))
    error = relative_error_percent(y_val, sign * 10.0 ** model.predict(Z_val))
    return case, config, terminal, model, sign, np.median(error), np.percentile(error, 95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=4, help="parallel MLP fits")
    args = parser.parse_args()

    data = pd.read_csv(DATA, dtype={"config": str})
    rng = np.random.default_rng(SPLIT_SEED)
    tasks, sizes = [], {}
    for (case, config), group in data.groupby(["case", "config"]):
        corners = group["corner"].unique()
        held_out = set(rng.choice(corners, size=int(0.2 * len(corners)), replace=False))
        val = group["corner"].isin(held_out).to_numpy()
        X = group[FEATURES].to_numpy(float)
        Z = rail_features(X, config)
        sizes[(case, config)] = (int((~val).sum()), int(val.sum()))
        for terminal in TARGETS:
            if case.endswith("_on") and terminal in ("id", "is"):
                continue
            y = group[terminal].to_numpy()
            if np.median(np.abs(y)) < NOISE:
                continue
            sign = 1.0 if (y[~val] > 0).mean() >= 0.5 else -1.0
            tasks.append((case, config, terminal, Z[~val], y[~val], Z[val], y[val], sign))

    print(f"Training {len(tasks)} sub-models on {len(data)} samples with {args.jobs} jobs...")
    start = time.time()
    results = Parallel(n_jobs=args.jobs)(delayed(fit_one)(*task) for task in tasks)
    print(f"  done in {(time.time() - start) / 60:.1f} min")

    rows, kept = [], {case: {} for case in CASES}
    for case, config, terminal, model, sign, median, p95 in results:
        keep = median < MAX_MEDIAN and p95 <= MAX_P95
        if keep:
            kept[case][(config, terminal)] = (model, sign)
        rows.append({"case": case, "config": config, "terminal": terminal,
                     "train_samples": sizes[(case, config)][0],
                     "heldout_samples": sizes[(case, config)][1],
                     "heldout_median_error_percent": median,
                     "heldout_p95_error_percent": p95, "kept": keep})
    metrics = pd.DataFrame(rows).sort_values(["case", "config", "terminal"])
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRIC_DIR / "rail_submodels_metrics.csv", index=False)
    print(f"Kept {int(metrics['kept'].sum())} of {len(metrics)} sub-models "
          f"(held-out median < {MAX_MEDIAN:.0f} % and P95 <= {MAX_P95:.0f} %)")
    print(metrics[~metrics["kept"]][["case", "config", "terminal",
                                     "heldout_median_error_percent",
                                     "heldout_p95_error_percent"]].round(1).to_string(index=False))

    effect = []
    for case in CASES:
        path = MODEL_DIR / f"{case}.joblib"
        package = joblib.load(path)
        model = package["model"]
        test = pd.read_csv(METRIC_DIR / f"{case}_test_predictions.csv")
        X = test[FEATURES].to_numpy(float)
        model.rail_models = {}
        before = model.predict(X)
        model.rail_models = kept[case]
        after = model.predict(X)
        # rows whose operating-point config has at least one kept sub-model
        changed = np.isin(rail_config(X), sorted({c for c, _ in kept[case]}))
        for source, subset in test.groupby("source"):
            rows_ = subset.index.to_numpy()
            for t, terminal in enumerate(TARGETS):
                e0 = relative_error_percent(subset[f"actual_{terminal}"], before[rows_, t])
                e1 = relative_error_percent(subset[f"actual_{terminal}"], after[rows_, t])
                effect.append({"case": case, "test_set": source, "terminal": terminal,
                               "rows_using_submodels": int(changed[rows_].sum()),
                               "median_error_before": np.median(e0),
                               "median_error_after": np.median(e1),
                               "p95_error_before": np.percentile(e0, 95),
                               "p95_error_after": np.percentile(e1, 95)})
        package["rail_submodels"] = {
            "configs": sorted({c for c, _ in kept[case]}),
            "terminals": sorted(f"{c}:{t}" for c, t in kept[case]),
            "data": DATA.name,
        }
        joblib.dump(package, path, compress=3)
        print(f"  {case}: {len(kept[case])} sub-models saved into {path.relative_to(ROOT)}")
    effect = pd.DataFrame(effect)
    effect.to_csv(METRIC_DIR / "rail_submodels_test_effect.csv", index=False)
    print("\nOriginal test set (median / P95 % error), before -> after sub-models:")
    for _, r in effect.iterrows():
        print(f"  {r.case:9s} {r.test_set:7s} {r.terminal}: "
              f"{r.median_error_before:5.2f} / {r.p95_error_before:6.1f}  ->  "
              f"{r.median_error_after:5.2f} / {r.p95_error_after:6.1f}"
              f"   [{r.rows_using_submodels} rows]")


if __name__ == "__main__":
    main()
