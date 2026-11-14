"""Train, compare and select the four transistor models (NMOS/PMOS, ON/OFF).

Run from the project root after src/spice/run_dataset.py:

    python .\\train_models.py

Method (see src/transistor_ml.py for details):
  * 70/15/15 train/validation/test split with a fixed seed.
  * Canonical source/drain orientation + physics features.
  * Targets are log10(|I| + 1e-30) per terminal; terminal signs come from a
    constant or a sign classifier fitted on the training set.
  * Four model families (Random Forest, Extra Trees, Histogram Gradient
    Boosting, MLP), each with a small hyperparameter grid.
  * For every terminal, the (family, config) with the lowest validation P95
    error is selected. The test set is only used after selection.

Errors are floor-adjusted relative errors |pred - true| / max(|true|, 1e-18 A);
1e-18 A is the SPICE abstol, below which currents are not resolved.
"""

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Avoids a noisy (harmless) physical-core probe in joblib/loky on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
# Eight training processes run in parallel; one BLAS thread each avoids
# oversubscribing the CPU (with default threading the MLP fits ran ~4x slower).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from transistor_ml import (  # noqa: E402
    ERROR_FLOOR,
    FEATURES,
    FRAMES,
    SIGN_CONSTANT_THRESHOLD,
    PerTerminalRegressor,
    TARGETS,
    SignModel,
    TransistorModel,
    device_of,
    frame_of,
    log_target,
    model_features,
    relative_error_percent,
    to_frame,
)

warnings.filterwarnings("ignore")

SEED = 42

DATASETS = {
    case: ROOT / "data" / "processed" / f"{case}.csv"
    for case in ("nmos_on", "nmos_off", "pmos_on", "pmos_off")
}
# Supplementary rail-edge samples (src/data_generation/generate_edge_samples.py).
# Set USE_EDGE_SAMPLES=0 in the environment to train on the uniform data only.
EDGE_DATASETS = {
    case: ROOT / "data" / "processed" / f"{case}_edge.csv" for case in DATASETS
}
USE_EDGE_SAMPLES = os.environ.get("USE_EDGE_SAMPLES", "1") != "0"
# Supplementary rail samples (src/data_generation/generate_rail_samples.py):
# Vd and Vs on/next to the rails, where transistors sit inside logic gates.
# Set USE_RAIL_SAMPLES=0 in the environment to leave them out.
RAIL_DATASETS = {
    case: ROOT / "data" / "processed" / f"{case}_rail.csv" for case in DATASETS
}
USE_RAIL_SAMPLES = os.environ.get("USE_RAIL_SAMPLES", "1") != "0"
# Supplementary stack samples (src/data_generation/generate_stack_samples.py):
# one terminal on a rail, the other at an internal stack-node voltage.
# Set USE_STACK_SAMPLES=0 in the environment to leave them out.
STACK_DATASETS = {
    case: ROOT / "data" / "processed" / f"{case}_stack.csv" for case in DATASETS
}
USE_STACK_SAMPLES = os.environ.get("USE_STACK_SAMPLES", "1") != "0"

MODEL_DIR = ROOT / "saved_models"
METRIC_DIR = ROOT / "results" / "metrics"
PLOT_DIR = ROOT / "results" / "plots"


# ============================================================
# CANDIDATE MODELS (family -> config name -> estimator)
# ============================================================

N_JOBS_PER_CASE = 1   # eight (case, frame) jobs train in parallel
MLP_SEEDS = tuple(range(int(os.environ.get("MLP_SEEDS", "1"))))


def get_candidates():
    """Model families and their configs.

    The grid was narrowed with a pilot run: max_features 0.5 vs 1.0 and
    learning rates 0.05 vs 0.1 changed validation P95 by only a few percent,
    while the MLP improved several-fold when trained with full-batch L-BFGS
    (sklearn's Adam has no learning-rate decay and stalls at ~20% error).
    """

    def forest(cls, max_features):
        # 100 trees, min leaf 2: with the larger combined dataset 300 fully
        # grown multi-output trees took ~1 GB per forest, more than eight
        # parallel jobs fit in memory. Validation error changed only slightly.
        return cls(
            n_estimators=100,
            min_samples_leaf=2,
            max_features=max_features,
            random_state=SEED,
            n_jobs=N_JOBS_PER_CASE,
        )

    def hgb(learning_rate, max_leaf_nodes):
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=3000,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=10,
            l2_regularization=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=SEED,
        ))

    def mlp(hidden):
        """One network per terminal current (PerTerminalRegressor).

        A shared 4-output network had to split its capacity between channel,
        gate-tunnelling and junction currents; separate networks reduced the
        validation error of the gate-leakage-critical terminals ~2x.
        """
        def make(seed):
            return TransformedTargetRegressor(
                regressor=Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=hidden,
                        activation="tanh",
                        solver="lbfgs",
                        alpha=1e-7,
                        max_iter=4000,
                        max_fun=40000,
                        tol=1e-12,
                        random_state=SEED + seed,
                    )),
                ]),
                transformer=StandardScaler(),
            )
        # Train each network only on currents the simulator resolves
        # (>= abstol); see PerTerminalRegressor. This halved the PMOS OFF
        # ib/is validation error in a pilot run.
        return PerTerminalRegressor(
            make, seeds=MLP_SEEDS, min_log=float(np.log10(ERROR_FLOOR))
        )

    return {
        "RandomForest": {"100 trees,mf=0.5": forest(RandomForestRegressor, 0.5)},
        "ExtraTrees": {"100 trees,mf=1.0": forest(ExtraTreesRegressor, 1.0)},
        "GradientBoosting": {"lr=0.05,leaves=31": hgb(0.05, 31)},
        "MLP": {"per-terminal 64x3,tanh,lbfgs": mlp((64, 64, 64))},
    }


# ============================================================
# METRICS
# ============================================================

# Model selection ignores currents below 100x the SPICE abstol (1e-18 A).
# The edge/rail samples put terminals exactly on the gate/body rail, where
# e.g. the PMOS OFF gate and body currents are exactly 0 A; any non-zero
# prediction then counts as a ~1000% error and these unresolved, physically
# irrelevant currents dominated the P95 and steered the selection.
RESOLVED_CURRENT = 1e-16


def selection_score(metrics):
    """Validation P95 on resolved currents; plain P95 if none are resolved
    (e.g. NMOS OFF ig, which is below 1e-16 A in every sample)."""
    score = metrics["p95_resolved_error_percent"]
    return metrics["p95_error_percent"] if np.isnan(score) else score


def terminal_metrics(y_true, y_pred):
    error = relative_error_percent(y_true, y_pred)
    log_true = np.log10(np.maximum(np.abs(y_true), ERROR_FLOOR))
    log_pred = np.log10(np.maximum(np.abs(y_pred), ERROR_FLOOR))
    resolved = np.abs(y_true) >= RESOLVED_CURRENT
    return {
        "R2": r2_score(y_true, y_pred),
        "R2_log": r2_score(log_true, log_pred),
        "median_error_percent": float(np.median(error)),
        "p95_error_percent": float(np.percentile(error, 95)),
        "p95_resolved_error_percent": (
            float(np.percentile(error[resolved], 95)) if resolved.any() else np.nan
        ),
    }


def pooled_metrics(Y_true, Y_pred):
    error = relative_error_percent(Y_true, Y_pred)
    resolved = np.abs(Y_true) >= RESOLVED_CURRENT
    return {
        "R2": r2_score(Y_true, Y_pred, multioutput="uniform_average"),
        "median_error_percent": float(np.median(error)),
        "p95_error_percent": float(np.percentile(error, 95)),
        "p95_resolved_error_percent": (
            float(np.percentile(error[resolved], 95)) if resolved.any() else np.nan
        ),
    }


# ============================================================
# SIGN MODELS
# ============================================================

def fit_sign_models(F_train, Y_train_canonical):
    signs = []
    info = []
    for t, target in enumerate(TARGETS):
        positive = Y_train_canonical[:, t] >= 0
        minority = min(positive.mean(), 1 - positive.mean())
        if minority < SIGN_CONSTANT_THRESHOLD or minority * len(positive) < 20:
            constant = 1.0 if positive.mean() >= 0.5 else -1.0
            signs.append(SignModel(constant=constant))
            info.append(f"{target}: constant {constant:+.0f} "
                        f"(minority {minority:.2%})")
        else:
            classifier = HistGradientBoostingClassifier(
                max_iter=1000,
                learning_rate=0.1,
                early_stopping=True,
                n_iter_no_change=30,
                random_state=SEED,
            ).fit(F_train, np.where(positive, 1, -1))
            signs.append(SignModel(classifier=classifier))
            info.append(f"{target}: classifier (minority {minority:.2%})")
    return signs, info


# ============================================================
# PARITY PLOT
# ============================================================

def make_parity_plot(case, y_true, y_pred, selection):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, ax in enumerate(axes.flat):
        actual = y_true[:, i]
        predicted = y_pred[:, i]
        ax.scatter(actual, predicted, alpha=0.5, s=10)
        ax.set_xscale("symlog", linthresh=ERROR_FLOOR)
        ax.set_yscale("symlog", linthresh=ERROR_FLOOR)
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], "r--", label="Ideal prediction")
        ax.set_xlabel(f"SPICE {TARGETS[i]} (A)")
        ax.set_ylabel(f"ML {TARGETS[i]} (A)")
        ax.set_title(f"{TARGETS[i].upper()} ({selection[i]})")
        ax.legend()
        ax.grid(True)

    fig.suptitle(f"{case.upper()} - Test Set Parity")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{case}_parity.png", dpi=300)
    plt.close(fig)


# ============================================================
# TRAIN ONE CASE
# ============================================================

def load_case_data(case):
    """Uniform (+ optional edge) data for one case, split 70/15/15 each."""

    def load_split(path):
        df = pd.read_csv(path)
        missing = [c for c in FEATURES + TARGETS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: Missing columns {missing}")
        df = df[FEATURES + TARGETS].dropna()
        X = df[FEATURES].to_numpy(dtype=float)
        Y = df[TARGETS].to_numpy(dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError(f"{path.name}: Dataset contains inf or NaN")
        X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(
            X, Y, test_size=0.30, random_state=SEED
        )
        X_va, X_te, Y_va, Y_te = train_test_split(
            X_tmp, Y_tmp, test_size=0.50, random_state=SEED
        )
        return X_tr, Y_tr, X_va, Y_va, X_te, Y_te

    # The uniform dataset keeps exactly the same split as before. The
    # supplementary edge dataset is split separately and appended.
    sources = {"uniform": load_split(DATASETS[case])}
    if USE_EDGE_SAMPLES and EDGE_DATASETS[case].exists():
        sources["edge"] = load_split(EDGE_DATASETS[case])
    if USE_RAIL_SAMPLES and RAIL_DATASETS[case].exists():
        sources["rail"] = load_split(RAIL_DATASETS[case])
    if USE_STACK_SAMPLES and STACK_DATASETS[case].exists():
        sources["stack"] = load_split(STACK_DATASETS[case])

    names = ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test")
    data = {
        name: np.concatenate([parts[i] for parts in sources.values()])
        for i, name in enumerate(names)
    }
    data["test_source"] = np.concatenate([
        np.full(len(parts[4]), name) for name, parts in sources.items()
    ])
    data["sources"] = list(sources)
    data["sizes"] = [
        f"{name:8s} train / val / test: "
        f"{len(parts[0])} / {len(parts[2])} / {len(parts[4])}"
        for name, parts in sources.items()
    ]
    return data


def fit_frame(job):
    """Fit every candidate family for one (case, target frame)."""

    case, frame = job
    device = device_of(case)
    data = load_case_data(case)

    F_train, mask_train = model_features(data["X_train"], device)
    F_val, mask_val = model_features(data["X_val"], device)
    Y_val = data["Y_val"]
    Y_frame = to_frame(data["Y_train"], mask_train, frame)

    signs, info = fit_sign_models(F_train, Y_frame)
    L_train = log_target(Y_frame)

    fitted, rows, log = {}, [], []
    best = [(np.inf, None)] * len(TARGETS)
    scorer = TransistorModel(case, fitted, [None] * len(TARGETS), {frame: signs})

    for family, configs in get_candidates().items():
        for config, estimator in configs.items():
            name = f"{frame}:{family}[{config}]"
            start = time.time()
            estimator.fit(F_train, L_train)
            fitted[name] = estimator

            Y_val_pred = scorer.predict_candidate(name, F_val, mask_val)
            pooled = pooled_metrics(Y_val, Y_val_pred)
            log.append(f"{name:48s} {time.time() - start:7.1f}s  "
                       f"median {pooled['median_error_percent']:8.3f}%  "
                       f"P95 {pooled['p95_error_percent']:9.3f}%")

            base = {"case": case, "model": family, "config": config, "frame": frame}
            for t, target in enumerate(TARGETS):
                metrics_t = terminal_metrics(Y_val[:, t], Y_val_pred[:, t])
                rows.append({**base, "current": target, **metrics_t})
                if selection_score(metrics_t) < best[t][0]:
                    best[t] = (selection_score(metrics_t), name)
            rows.append({**base, "current": "all", **pooled})

            # Selection (train_case) picks, per terminal, the lowest validation
            # P95 over all frames and families, so only a candidate that is
            # currently best in this frame for some terminal can be selected.
            # Dropping the others keeps memory bounded (forests are large).
            keep = {name for _, name in best}
            for other in [n for n in fitted if n not in keep]:
                del fitted[other]

    return {
        "case": case,
        "frame": frame,
        "fitted": fitted,
        "signs": signs,
        "sign_info": [f"{frame:9s} {line}" for line in info],
        "rows": rows,
        "log": log,
    }


# ============================================================
# SELECT AND TEST ONE CASE
# ============================================================

def train_case(case, frame_results):
    """Select per terminal, test and save one case. Returns results and log."""

    log = []

    def say(text=""):
        log.append(str(text))

    say("=" * 60)
    say(f"TRAINING: {case.upper()}")
    say("=" * 60)

    data = load_case_data(case)
    X_test, Y_test = data["X_test"], data["Y_test"]
    test_source = data["test_source"]
    sources = data["sources"]
    for line in data["sizes"]:
        say(line)

    fitted, signs, sign_info, comparison_rows = {}, {}, [], []
    for result in frame_results:
        fitted.update(result["fitted"])
        signs[result["frame"]] = result["signs"]
        sign_info += result["sign_info"]
        comparison_rows += result["rows"]
    say("Sign models:\n  " + "\n  ".join(sign_info))
    for result in frame_results:
        for line in result["log"]:
            say(line)

    comparison = pd.DataFrame(comparison_rows)

    # --------------------------------------------------------
    # Per-terminal selection on validation P95 (resolved currents, see
    # RESOLVED_CURRENT)
    # --------------------------------------------------------

    selection = []
    for target in TARGETS:
        rows = comparison[comparison["current"] == target]
        best = rows.loc[rows.apply(selection_score, axis=1).idxmin()]
        selection.append(f"{best['frame']}:{best['model']}[{best['config']}]")

    say("\nSelected per terminal (validation P95, resolved currents):")
    for target, name in zip(TARGETS, selection):
        say(f"  {target}: {name}")

    model = TransistorModel(
        case=case,
        regressors={name: fitted[name] for name in set(selection)},
        selection=selection,
        signs={frame: signs[frame] for frame in {frame_of(n) for n in selection}},
    )

    # --------------------------------------------------------
    # Evaluate the selected composite model on the untouched test set
    # --------------------------------------------------------

    Y_test_pred = model.predict(X_test)
    test_pooled = pooled_metrics(Y_test, Y_test_pred)
    say("\nTEST RESULTS (all terminals pooled)")
    say({k: round(v, 4) for k, v in test_pooled.items()})
    uniform = test_source == "uniform"
    test_pooled_uniform = pooled_metrics(Y_test[uniform], Y_test_pred[uniform])
    say("Uniform test set only:")
    say({k: round(v, 4) for k, v in test_pooled_uniform.items()})

    per_current = pd.DataFrame([
        {
            "case": case,
            "test_set": name,
            "current": target,
            "selected_model": selection[t],
            **terminal_metrics(Y_test[subset, t], Y_test_pred[subset, t]),
        }
        for name, subset in [("all", slice(None))] + [
            (src, test_source == src) for src in sources
        ]
        for t, target in enumerate(TARGETS)
    ])
    say(per_current.to_string(index=False))
    per_current.to_csv(METRIC_DIR / f"{case}_test_metrics.csv", index=False)

    pd.concat([
        pd.DataFrame({"source": test_source}),
        pd.DataFrame(X_test, columns=FEATURES),
        pd.DataFrame(Y_test, columns=[f"actual_{t}" for t in TARGETS]),
        pd.DataFrame(Y_test_pred, columns=[f"predicted_{t}" for t in TARGETS]),
    ], axis=1).to_csv(METRIC_DIR / f"{case}_test_predictions.csv", index=False)

    make_parity_plot(case, Y_test, Y_test_pred, selection)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "targets": TARGETS,
            "case": case,
            "selected_models": dict(zip(TARGETS, selection)),
            "sign_models": sign_info,
        },
        MODEL_DIR / f"{case}.joblib",
        compress=3,
    )

    final = {
        "case": case,
        "selected_models": "; ".join(f"{t}={s}" for t, s in zip(TARGETS, selection)),
        **test_pooled,
        **{f"uniform_{k}": v for k, v in test_pooled_uniform.items()},
    }
    return comparison, per_current, final, "\n".join(log)


# ============================================================
# MAIN
# ============================================================

def main():

    for directory in (MODEL_DIR, METRIC_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for path in DATASETS.values():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    # Every (case, target frame) is independent: fit all eight in parallel,
    # then select and test each case.
    # TRAIN_CASES=pmos_off (comma separated) retrains only those cases; the
    # combined metric tables keep the existing rows of the other cases.
    cases = [c for c in os.environ.get("TRAIN_CASES", ",".join(DATASETS)).split(",") if c]
    unknown = set(cases) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown TRAIN_CASES: {sorted(unknown)}")
    jobs = [(case, frame) for case in cases for frame in FRAMES]
    # The fits take ~2 h; checkpoint them so a failure in the selection /
    # test / save step below can be rerun with REUSE_FITS=1 without refitting.
    checkpoint = ROOT / "data" / "training_cache" / "frame_results.joblib"
    if os.environ.get("REUSE_FITS") == "1" and checkpoint.exists():
        frame_results = [r for r in joblib.load(checkpoint) if r["case"] in cases]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            frame_results = list(pool.map(fit_frame, jobs))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(frame_results, checkpoint, compress=3)
    results = [
        train_case(case, [r for r in frame_results if r["case"] == case])
        for case in cases
    ]

    def merged(new, name):
        """Keep rows of cases that were not retrained in this run."""
        path = METRIC_DIR / name
        if len(cases) < len(DATASETS) and path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old[~old["case"].isin(cases)], new], ignore_index=True)
            order = {c: i for i, c in enumerate(DATASETS)}
            new = new.sort_values("case", key=lambda c: c.map(order), kind="stable")
        new.to_csv(path, index=False)
        return new

    for *_, log in results:
        print(log + "\n")

    comparison = merged(
        pd.concat([r[0] for r in results], ignore_index=True),
        "model_comparison_detailed.csv",
    ).reset_index(drop=True)

    # Deliverable table: 4 model families x 4 cases on validation data,
    # each family at its best (frame, config) by pooled P95.
    pooled = comparison[comparison["current"] == "all"]
    summary = pooled.loc[
        pooled.groupby(["case", "model"], sort=False)["p95_error_percent"].idxmin()
    ].drop(columns=["current", "R2_log"])
    summary.to_csv(METRIC_DIR / "model_comparison.csv", index=False)

    merged(pd.concat([r[1] for r in results], ignore_index=True),
           "final_test_metrics_per_current.csv")
    merged(pd.DataFrame([r[2] for r in results]), "final_test_results.csv")

    pd.set_option("display.width", 200)
    print("=" * 60)
    print(f"TRAINED: {', '.join(cases)}")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
