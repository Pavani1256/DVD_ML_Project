"""Train the operating-point sub-models and add them to the four saved models.

Run after train_models.py and src/spice/generate_rail_dataset.py:

    python ./train_rail_models.py --jobs 8

For each case (nmos_on/off, pmos_on/off), configuration (exact-rail 00 01 10
11, one-rail m0 m1 0m 1m) and terminal, several candidate regressors of
log10|I| are compared:

  * "mlp":  the original sub-model, an MLP on VDD, T, log W and the process
            parameters (plus the free node voltage for one-rail configs);
  * "poly": polynomial ridge regressions (degree 2-4, several ridge strengths)
            on the same inputs plus BSIM4-style physics estimates computed at
            the operating point (effective threshold, EKV channel current,
            gate-tunnelling exponents, 1/T, ...; see rail_phys_features in
            src/transistor_ml.py), with log10|I| normalised by W or by
            Weff = W - 2*wint (junction currents scale with W, channel and
            gate currents with Weff).

The sign is the training data's (unanimous) sign. 20 % of the corners of
every (case, config) are held out (SPLIT_SEED = 1, as before). The candidate
is chosen on an inner 20 % split of the *training* corners, refitted on all
training corners and then scored on the held-out corners. A sub-model is kept
only if, on its held-out corners, (1) the median error is < 10 % and the P95
error is <= 50 %, and (2) its mean error (each sample capped at 100 %, i.e.
100 - mean accuracy) is lower than the general model's on the same held-out
samples. Otherwise the general model stays in charge of that terminal.

Kirchhoff's law (OFF devices only): a single transistor obeys
id + ig + is + ib = 0. For each OFF-device configuration, the drain or source
current may instead be computed as minus the sum of the other three predicted
terminals (useful when it is a sum of two mechanisms, e.g. subthreshold plus
gate-drain tunnelling). This is used if it lowers the held-out mean error of
that terminal.

Only single-transistor data is used for every choice, never gate or circuit
results. Skipped entirely:
  * terminals whose current is numerical noise (median |I| < 1e-16 A),
  * ON-device id/is: the gate procedure never uses them (Kirchhoff's law
    solves the ON-device channel currents).

The kept sub-models are stored in each saved package as
``package["model"].rail_models`` and the KCL choices as
``package["model"].rail_kcl`` (both used by TransistorModel.predict), so there
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
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from transistor_ml import (FEATURES, LOG_FLOOR, TARGETS, RailPolyModel,  # noqa: E402
                           device_of, rail_config, rail_features,
                           rail_phys_features, rail_width_log10,
                           relative_error_percent)

DATA = ROOT / "data" / "processed" / "rail_samples.csv.gz"
MODEL_DIR = ROOT / "saved_models"
METRIC_DIR = ROOT / "results" / "metrics"
CASES = ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
SPLIT_SEED = 1         # held-out corners (same split as the original script)
INNER_SEED = 2         # candidate selection inside the training corners
NOISE = 1e-16          # A; below this SPICE terminal currents are rounding noise
MAX_MEDIAN = 10.0      # % held-out median error to keep a sub-model
MAX_P95 = 50.0         # % held-out P95 error to keep a sub-model
ALPHAS = (1e-8, 1e-5, 1e-3, 1e-1)


def poly_candidates(config):
    """(degree, ridge alpha, width normalisation) to try. Degree 4 only for
    exact-rail configs: with the extra free-node inputs it has more terms than
    there are training samples."""
    degrees = (2, 3) if "m" in config else (2, 3, 4)
    return [(d, a, n) for d in degrees for a in ALPHAS for n in ("weff", "w")]


def mean_error(y, prediction):
    """Mean relative error with each sample capped at 100 % (= 100 - mean accuracy)."""
    return float(np.mean(np.minimum(relative_error_percent(y, prediction), 100.0)))


def make_mlp(config):
    hidden = (96, 96) if "m" in config else (64, 64)
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=hidden, activation="tanh", alpha=1e-6, max_iter=3000,
        learning_rate_init=3e-3, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=60, tol=1e-7, random_state=0))


def fit_candidate(spec, config, device, X, y):
    """Fit one candidate on raw inputs X; return an object with predict_log10(X)."""
    L = np.log10(np.abs(y) + LOG_FLOOR)
    if spec == "mlp":
        model = make_mlp(config).fit(rail_features(X, config), L)
        return _MLPWrapper(model, config)
    degree, alpha, norm = spec
    pipeline = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge(alpha=alpha))
    pipeline.fit(rail_phys_features(X, config, device), L - rail_width_log10(X, norm))
    return RailPolyModel(pipeline, config, device, norm)


class _MLPWrapper:
    """Training-time adapter so MLP and polynomial candidates share predict_log10.
    Only the inner MLP pipeline is saved (the original sub-model format)."""

    def __init__(self, model, config):
        self.model, self.config = model, config

    def predict_log10(self, X):
        return self.model.predict(rail_features(X, self.config))


def label(spec):
    return "mlp" if spec == "mlp" else f"poly{spec[0]}_a{spec[1]:g}_{spec[2]}"


def train_group(case, config, group, general_model, held_out):
    """All terminals of one (case, config): select, refit, score."""
    warnings.filterwarnings("ignore")   # ridge ill-conditioning warnings at alpha=1e-8
    device = device_of(case)
    corners = group["corner"].unique()
    val = group["corner"].isin(held_out).to_numpy()
    train_corners = np.array(sorted(set(corners) - held_out))
    inner = set(np.random.default_rng(INNER_SEED).choice(
        train_corners, size=int(0.2 * len(train_corners)), replace=False))
    inner_val = group["corner"].isin(inner).to_numpy() & ~val
    inner_fit = ~val & ~inner_val
    X = group[FEATURES].to_numpy(float)
    general_val = general_model.predict(X[val])

    rows, kept, final_val = [], {}, {}
    for t, terminal in enumerate(TARGETS):
        if case.endswith("_on") and terminal in ("id", "is"):
            continue
        y = group[terminal].to_numpy()
        if np.median(np.abs(y)) < NOISE:
            continue
        sign = 1.0 if (y[~val] > 0).mean() >= 0.5 else -1.0
        best = None
        for spec in ["mlp"] + poly_candidates(config):
            model = fit_candidate(spec, config, device, X[inner_fit], y[inner_fit])
            score = mean_error(y[inner_val], sign * 10.0 ** model.predict_log10(X[inner_val]))
            if best is None or score < best[0]:
                best = (score, spec)
        model = fit_candidate(best[1], config, device, X[~val], y[~val])
        prediction = sign * 10.0 ** model.predict_log10(X[val])
        error = relative_error_percent(y[val], prediction)
        median, p95, mean = np.median(error), np.percentile(error, 95), mean_error(y[val], prediction)
        general_mean = mean_error(y[val], general_val[:, t])
        passes = median < MAX_MEDIAN and p95 <= MAX_P95
        keep = passes and mean < general_mean
        if keep:
            saved = model.model if isinstance(model, _MLPWrapper) else model
            kept[(config, terminal)] = (saved, sign)
        final_val[terminal] = prediction if keep else general_val[:, t]
        rows.append({"case": case, "config": config, "terminal": terminal,
                     "regressor": label(best[1]),
                     "inner_mean_error_percent": best[0],
                     "train_samples": int((~val).sum()), "heldout_samples": int(val.sum()),
                     "heldout_median_error_percent": median,
                     "heldout_p95_error_percent": p95,
                     "heldout_mean_error_percent": mean,
                     "general_heldout_mean_error_percent": general_mean,
                     "passes_thresholds": passes, "kept": keep, "kcl_derived": False})

    # KCL-derived drain or source current for OFF devices.
    kcl = None
    if case.endswith("_off"):
        full = general_val.copy()
        for terminal, prediction in final_val.items():
            full[:, TARGETS.index(terminal)] = prediction
        gate_drop = np.max(np.abs(X[val][:, [3, 4, 6]] - X[val][:, [5]]), axis=1)
        full[gate_drop < 1e-3, TARGETS.index("ig")] = 0.0
        best_gain = 0.0
        for terminal in ("id", "is"):
            t = TARGETS.index(terminal)
            y = group[terminal].to_numpy()[val]
            if np.median(np.abs(group[terminal].to_numpy())) < NOISE:
                continue   # numerical-noise current: nothing to improve
            direct = mean_error(y, full[:, t])
            derived = mean_error(y, -np.delete(full, t, axis=1).sum(axis=1))
            if direct - derived > best_gain:
                best_gain, kcl = direct - derived, (terminal, direct, derived)
        if kcl is not None:
            terminal, direct, derived = kcl
            for row in rows:
                if row["terminal"] == terminal:
                    row["kcl_derived"] = True
                    row["kcl_heldout_mean_error_percent"] = derived
            if not any(r["terminal"] == terminal for r in rows):
                rows.append({"case": case, "config": config, "terminal": terminal,
                             "regressor": "kcl", "kept": False, "kcl_derived": True,
                             "general_heldout_mean_error_percent": direct,
                             "kcl_heldout_mean_error_percent": derived})
            kcl = terminal
    return case, config, rows, kept, kcl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=4, help="parallel (case, config) groups")
    args = parser.parse_args()

    data = pd.read_csv(DATA, dtype={"config": str})
    general = {}
    for case in CASES:
        model = joblib.load(MODEL_DIR / f"{case}.joblib")["model"]
        model.rail_models, model.rail_kcl = {}, {}
        general[case] = model
    # Held-out corners: one generator drawn in groupby order, exactly as in the
    # original version of this script, so the held-out sets are unchanged.
    rng = np.random.default_rng(SPLIT_SEED)
    groups = []
    for (case, config), group in data.groupby(["case", "config"]):
        corners = group["corner"].unique()
        held_out = set(rng.choice(corners, size=int(0.2 * len(corners)), replace=False))
        groups.append((case, config, group.reset_index(drop=True), general[case], held_out))

    print(f"Selecting and training sub-models for {len(groups)} (case, config) groups "
          f"on {len(data)} samples with {args.jobs} jobs...")
    start = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(train_group)(*g) for g in groups)
    print(f"  done in {(time.time() - start) / 60:.1f} min")

    rows = [r for _, _, group_rows, _, _ in results for r in group_rows]
    kept = {case: {} for case in CASES}
    kcl = {case: {} for case in CASES}
    for case, config, _, group_kept, group_kcl in results:
        kept[case].update(group_kept)
        if group_kcl:
            kcl[case][config] = group_kcl
    metrics = pd.DataFrame(rows).sort_values(["case", "config", "terminal"])
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRIC_DIR / "rail_submodels_metrics.csv", index=False)
    print(f"{int(metrics['kept'].sum())} sub-models kept "
          f"({int((metrics['kept'] & metrics['regressor'].str.startswith('poly')).sum())} polynomial, "
          f"{int((metrics['kept'] & (metrics['regressor'] == 'mlp')).sum())} MLP); "
          f"KCL-derived terminals: {sum(len(v) for v in kcl.values())}")
    columns = ["case", "config", "terminal", "regressor", "heldout_median_error_percent",
               "heldout_p95_error_percent", "heldout_mean_error_percent",
               "general_heldout_mean_error_percent", "kept", "kcl_derived"]
    with pd.option_context("display.width", 200):
        print(metrics[columns].round(3).to_string(index=False))

    effect = []
    for case in CASES:
        path = MODEL_DIR / f"{case}.joblib"
        package = joblib.load(path)
        model = package["model"]
        test = pd.read_csv(METRIC_DIR / f"{case}_test_predictions.csv")
        X = test[FEATURES].to_numpy(float)
        model.rail_models, model.rail_kcl = {}, {}
        before = model.predict(X)
        model.rail_models, model.rail_kcl = kept[case], kcl[case]
        after = model.predict(X)
        changed = np.isin(rail_config(X), sorted({c for c, _ in kept[case]} | set(kcl[case])))
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
            "kcl": dict(kcl[case]),
            "data": DATA.name,
        }
        joblib.dump(package, path, compress=3)
        print(f"  {case}: {len(kept[case])} sub-models, KCL {dict(kcl[case])} "
              f"saved into {path.relative_to(ROOT)}")
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
