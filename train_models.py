
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATASETS = {
    "nmos_on": "data/processed/nmos_on.csv",
    "nmos_off": "data/processed/nmos_off.csv",
    "pmos_on": "data/processed/pmos_on.csv",
    "pmos_off": "data/processed/pmos_off.csv",
}

FEATURES = [
    "vdd", "temp", "w", "vd", "vs", "vg", "vb",
    "vth0", "toxe", "ndep", "u0", "xj"
]

TARGETS = ["id", "ig", "is", "ib"]

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# ============================================================
# SIGNED LOG TRANSFORM FOR CURRENTS
# ============================================================

class SignedLogTransform:

    def fit(self, y):
        self.scale = np.median(np.abs(y), axis=0)
        self.scale = np.maximum(self.scale, 1e-30)
        return self

    def transform(self, y):
        return np.sign(y) * np.log1p(
            np.abs(y) / self.scale
        )

    def inverse_transform(self, y):
        return np.sign(y) * self.scale * np.expm1(
            np.abs(y)
        )


# ============================================================
# MODELS
# ============================================================

def get_models():

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        ),

        "GradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=SEED
            )
        ),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 128),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=SEED
            ))
        ])
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, error_floor):

    relative_error = (
        np.abs(y_pred - y_true)
        / np.maximum(np.abs(y_true), error_floor)
    ) * 100

    return {
        "R2": r2_score(y_true, y_pred, multioutput="uniform_average"),
        "median_error_percent": np.median(relative_error),
        "p95_error_percent": np.percentile(relative_error, 95)
    }


def calculate_per_current_metrics(y_true, y_pred, error_floor):

    rows = []

    for i, target in enumerate(TARGETS):

        error = (
            np.abs(y_pred[:, i] - y_true[:, i])
            / np.maximum(np.abs(y_true[:, i]), error_floor[i])
        ) * 100

        rows.append({
            "current": target,
            "R2": r2_score(y_true[:, i], y_pred[:, i]),
            "median_error_percent": np.median(error),
            "p95_error_percent": np.percentile(error, 95)
        })

    return rows


# ============================================================
# PARITY PLOT
# ============================================================

def make_parity_plot(case, y_true, y_pred):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, ax in enumerate(axes.flat):

        actual = y_true[:, i]
        predicted = y_pred[:, i]

        ax.scatter(
            actual, predicted,
            alpha=0.5,
            s=12
        )

        # Signed logarithmic scale handles negative currents
        ax.set_xscale("symlog", linthresh=1e-15)
        ax.set_yscale("symlog", linthresh=1e-15)

        minimum = min(actual.min(), predicted.min())
        maximum = max(actual.max(), predicted.max())

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            "r--",
            label="Ideal prediction"
        )

        ax.set_xlabel(f"SPICE {TARGETS[i]} (A)")
        ax.set_ylabel(f"ML {TARGETS[i]} (A)")
        ax.set_title(TARGETS[i].upper())
        ax.legend()
        ax.grid(True)

    fig.suptitle(f"{case.upper()} - Test Set Parity")
    fig.tight_layout()

    plt.savefig(
        f"results/plots/{case}_parity.png",
        dpi=300
    )

    plt.close()


# ============================================================
# TRAIN ONE CASE
# ============================================================

def train_case(case, csv_path):

    print("\n" + "=" * 60)
    print(f"TRAINING: {case.upper()}")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    required_columns = FEATURES + TARGETS

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{case}: Missing columns {missing}"
        )

    df = df[required_columns].dropna()

    X = df[FEATURES].to_numpy(dtype=float)
    Y = df[TARGETS].to_numpy(dtype=float)

    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"{case}: Dataset contains inf or NaN")

    # 70% training, 15% validation, 15% testing
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y,
        test_size=0.30,
        random_state=SEED
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=0.50,
        random_state=SEED
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))

    # Fit transformation using training targets only
    transform = SignedLogTransform().fit(Y_train)

    Y_train_transformed = transform.transform(Y_train)
    Y_val_transformed = transform.transform(Y_val)

    # Relative error denominator floor:
    # 0.1% of training median absolute current
    error_floor = np.maximum(
        1e-30,
        1e-3 * np.median(np.abs(Y_train), axis=0)
    )

    model_results = []

    best_model = None
    best_model_name = None
    best_score = float("inf")

    # --------------------------------------------------------
    # Compare four models using validation set
    # --------------------------------------------------------

    for model_name, model in get_models().items():

        print(f"\nFitting {model_name}...")

        model.fit(X_train, Y_train_transformed)

        val_prediction_transformed = model.predict(X_val)

        val_prediction = transform.inverse_transform(
            val_prediction_transformed
        )

        metrics = calculate_metrics(
            Y_val,
            val_prediction,
            error_floor
        )

        print("Validation R2:", metrics["R2"])
        print(
            "Validation median error:",
            metrics["median_error_percent"], "%"
        )
        print(
            "Validation P95 error:",
            metrics["p95_error_percent"], "%"
        )

        model_results.append({
            "case": case,
            "model": model_name,
            **metrics
        })

        # Choose model with lowest validation P95 error
        if metrics["p95_error_percent"] < best_score:

            best_score = metrics["p95_error_percent"]
            best_model = model
            best_model_name = model_name

    print("\nSelected model:", best_model_name)

    # --------------------------------------------------------
    # Evaluate selected model on untouched test set
    # --------------------------------------------------------

    test_prediction_transformed = best_model.predict(X_test)

    test_prediction = transform.inverse_transform(
        test_prediction_transformed
    )

    test_metrics = calculate_metrics(
        Y_test,
        test_prediction,
        error_floor
    )

    print("\nTEST RESULTS")
    print(test_metrics)

    # --------------------------------------------------------
    # Save model and target transformation
    # --------------------------------------------------------

    model_package = {
        "model": best_model,
        "target_transform": transform,
        "features": FEATURES,
        "targets": TARGETS,
        "case": case,
        "selected_model_name": best_model_name
    }

    joblib.dump(
        model_package,
        f"saved_models/{case}.joblib"
    )

    # --------------------------------------------------------
    # Save test metrics per current
    # --------------------------------------------------------

    per_current = calculate_per_current_metrics(
        Y_test,
        test_prediction,
        error_floor
    )

    for row in per_current:
        row["case"] = case
        row["selected_model"] = best_model_name

    pd.DataFrame(per_current).to_csv(
        f"results/metrics/{case}_test_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_df = pd.DataFrame(
        Y_test,
        columns=[f"actual_{t}" for t in TARGETS]
    )

    prediction_df = pd.DataFrame(
        test_prediction,
        columns=[f"predicted_{t}" for t in TARGETS]
    )

    pd.concat(
        [test_df, prediction_df],
        axis=1
    ).to_csv(
        f"results/metrics/{case}_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    make_parity_plot(
        case,
        Y_test,
        test_prediction
    )

    return model_results, {
        "case": case,
        "selected_model": best_model_name,
        **test_metrics
    }


# ============================================================
# MAIN
# ============================================================

all_validation_results = []
all_test_results = []

for case, path in DATASETS.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    validation_results, test_results = train_case(
        case,
        path
    )

    all_validation_results.extend(validation_results)
    all_test_results.append(test_results)


# Save validation comparison table
pd.DataFrame(all_validation_results).to_csv(
    "results/metrics/model_comparison.csv",
    index=False
)

# Save final selected model test results
pd.DataFrame(all_test_results).to_csv(
    "results/metrics/final_test_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("ALL FOUR CASES TRAINED SUCCESSFULLY")
print("=" * 60)

print("\nSaved models in saved_models/")
print("Metrics in results/metrics/")
print("Plots in results/plots/")
