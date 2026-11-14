"""Shared transistor-model code used by training and by gate validation.

Saved model packages pickle a ``TransistorModel`` from this module, so any
script that loads them must have ``src/`` on ``sys.path``.

Prediction pipeline for one transistor case (e.g. NMOS OFF):

1. Canonical orientation. Each sample is relabelled so that the "source" is
   the lower-voltage terminal for NMOS and the higher-voltage terminal for
   PMOS (Vds >= 0 for NMOS, Vsd >= 0 for PMOS). The physics features are
   computed in this frame, where the channel current always flows one way.
2. Physics features. Vgs, Vds, Vbs, Vgd, Vbd, a BSIM4-style effective
   threshold (body effect, temperature, DIBL), log10 of an EKV channel-current
   estimate, oxide-field proxies, log10(W), 1/T, plus the raw vd, vs and the
   swap flag (the model card is not perfectly drain/source symmetric).
3. Targets. Every terminal current is regressed as log10(|I| + 1e-21), in
   either the raw or the canonical terminal frame (both are tried). The sign
   of each terminal is constant if the training data agrees on it, otherwise
   predicted by a classifier.
4. Per-terminal selection. Each terminal uses whichever (frame, regressor)
   had the lowest validation P95 error for that terminal.
5. Operating-point sub-models (optional, added by train_rail_models.py).
   In static CMOS every transistor terminal sits exactly at a rail (0 or
   VDD) or, for series-stack devices, one terminal sits at a rail and the
   other at an internal node. The uniform datasets almost never sample those
   exact points, and there the general regressors are 3-8 % off. For a
   device whose drain and source are both within RAIL_TOLERANCE of a rail
   ("exact-rail" config, e.g. "01" = drain at 0, source at VDD), or whose
   drain or source is (the other being free, "m"), a small MLP trained on
   single-transistor SPICE samples at exactly that configuration replaces
   the general prediction for the terminals it was kept for. The models
   live in ``TransistorModel.rail_models`` {(config, terminal): (model, sign)}.
6. Zero-bias gate rule. ig is set to 0 when the gate is within 1 mV of every
   other terminal (no oxide field, so no tunnelling current).
"""

from __future__ import annotations

import numpy as np

FEATURES = [
    "vdd", "temp", "w", "vd", "vs", "vg", "vb",
    "vth0", "toxe", "ndep", "u0", "xj",
]
TARGETS = ["id", "ig", "is", "ib"]

# Floor inside log10(|I| + floor). 1e-21 A is 1000x below the SPICE abstol
# (1e-18 A): currents smaller than that are numerical noise, and a lower
# floor would make the regressors spend capacity fitting that noise.
LOG_FLOOR = 1e-21
# Relative-error denominator floor. Equal to the SPICE abstol (1e-18 A):
# currents below it are not resolved by the simulator.
ERROR_FLOOR = 1e-18
# A terminal whose minority sign is rarer than this uses a constant sign.
SIGN_CONSTANT_THRESHOLD = 1e-3

BOLTZMANN_OVER_Q = 8.617333262e-5  # V/K

_VD, _VS = FEATURES.index("vd"), FEATURES.index("vs")
_VG, _VB = FEATURES.index("vg"), FEATURES.index("vb")
_ID, _IS = TARGETS.index("id"), TARGETS.index("is")
_IG = TARGETS.index("ig")

# Gate-oxide tunnelling needs a voltage across the oxide. When the gate is
# within this many volts of every other terminal, the true gate current is
# zero (SPICE returns ~1e-16 A rounding noise), and the regressors are not
# reliable there: the PMOS OFF model predicted ~1e-7 A with all four
# terminals at VDD at high VDD. Such predictions are set to exactly 0.
ZERO_BIAS_GATE_TOLERANCE = 1e-3

# Operating-point sub-models (step 5 above).
# A drain/source within this many volts of 0 or VDD counts as "at the rail".
RAIL_TOLERANCE = 1e-3
RAIL_CONFIGS_EXACT = ("00", "01", "10", "11")     # (drain, source): 0 = 0 V, 1 = VDD
RAIL_CONFIGS_ONE = ("m0", "m1", "0m", "1m")       # m = free (internal node)
_RAIL_BASE = ["vdd", "temp", "w", "vth0", "toxe", "ndep", "u0", "xj"]


def rail_config(X) -> np.ndarray:
    """Operating-point configuration of every row: '00'..'11', 'm0'.., or 'mm'."""
    X = np.asarray(X, dtype=float)
    vdd = X[:, FEATURES.index("vdd")]
    levels = []
    for column in (_VD, _VS):
        v = X[:, column]
        levels.append(np.where(np.abs(v) < RAIL_TOLERANCE, "0",
                               np.where(np.abs(v - vdd) < RAIL_TOLERANCE, "1", "m")))
    return np.char.add(levels[0].astype(str), levels[1].astype(str))


def rail_features(X, config: str) -> np.ndarray:
    """Inputs of a sub-model: VDD, T, log W, process parameters (toxe in nm,
    log10 ndep) and, for one-rail configs, the free node voltage as a
    fraction of VDD plus its log distance to each rail."""
    X = np.asarray(X, dtype=float)
    Z = X[:, [FEATURES.index(c) for c in _RAIL_BASE]].copy()
    Z[:, 2] = np.log(Z[:, 2])
    Z[:, 4] = Z[:, 4] * 1e9
    Z[:, 5] = np.log10(Z[:, 5])
    if "m" in config:
        free = X[:, _VD if config[0] == "m" else _VS] / X[:, FEATURES.index("vdd")]
        Z = np.column_stack([Z, free,
                             np.log10(np.clip(free, 1e-7, None)),
                             np.log10(np.clip(1.0 - free, 1e-7, None))])
    return Z


def device_of(case: str) -> str:
    return case.split("_")[0]


def canonical_swap_mask(X: np.ndarray, device: str) -> np.ndarray:
    """Rows whose drain and source must be swapped to reach canonical form."""
    if device == "nmos":
        return X[:, _VD] < X[:, _VS]
    return X[:, _VD] > X[:, _VS]


def swap_drain_source(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=float, copy=True)
    X[mask, _VD], X[mask, _VS] = X[mask, _VS], X[mask, _VD].copy()
    return X


def swap_id_is(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Y = np.array(Y, dtype=float, copy=True)
    Y[mask, _ID], Y[mask, _IS] = Y[mask, _IS], Y[mask, _ID].copy()
    return Y


# Fixed BSIM4 parameters from models/16nm_LP.pm used by the physics features
# (they are not swept, so they are constants here).
BSIM_CONSTANTS = {
    "nmos": {"k1": 0.4, "kt1": -0.11, "eta0": 0.0095, "nfactor": 1.6, "voff": -0.1014},
    "pmos": {"k1": 0.4, "kt1": -0.11, "eta0": 0.0095, "nfactor": 1.8, "voff": -0.08},
}
TNOM_K = 27.0 + 273.15
LMIN = 16e-9
EPS_OX = 3.9 * 8.854e-12          # F/m
NI_CM3 = 1.45e10                  # intrinsic carrier density, cm^-3


def _log_softplus(x):
    """log(log(1 + exp(x))), stable for very negative x."""
    x = np.asarray(x, dtype=float)
    return np.where(x < -30.0, x, np.log(np.logaddexp(0.0, np.maximum(x, -30.0))))


def engineer_features(X_canonical: np.ndarray, device: str) -> np.ndarray:
    """Raw inputs plus physics-motivated features (canonical frame).

    Includes a BSIM4-style effective threshold (body effect, temperature,
    DIBL) and log10 of an EKV channel-current estimate, so the regressors
    only have to learn a correction to first-order device physics.
    """
    f = {name: X_canonical[:, i] for i, name in enumerate(FEATURES)}
    c = BSIM_CONSTANTS[device]
    polarity = 1.0 if device == "nmos" else -1.0

    kelvin = f["temp"] + 273.15
    vt = BOLTZMANN_OVER_Q * kelvin
    vgs = polarity * (f["vg"] - f["vs"])
    vds = polarity * (f["vd"] - f["vs"])          # >= 0 in canonical frame
    vbs = polarity * (f["vb"] - f["vs"])
    vgd = polarity * (f["vg"] - f["vd"])
    vbd = polarity * (f["vb"] - f["vd"])
    vgb = polarity * (f["vg"] - f["vb"])
    vsb = np.maximum(-vbs, 0.0)
    vdb = np.maximum(-vbd, 0.0)
    vth0 = np.abs(f["vth0"])

    # Surface potential (BSIM4: phis = 0.4 + Vt0 * ln(NDEP / ni)).
    phis = 0.4 + BOLTZMANN_OVER_Q * TNOM_K * np.log(f["ndep"] / NI_CM3)
    body = c["k1"] * (np.sqrt(phis + vsb) - np.sqrt(phis))
    vth_t = vth0 + c["kt1"] * (kelvin / TNOM_K - 1.0)
    vth_eff = vth_t + body - c["eta0"] * vds
    n = 1.0 + c["nfactor"] * 0.5 * c["k1"] / np.sqrt(phis + vsb)

    # EKV channel current, bulk referenced, forward minus reverse.
    cox = EPS_OX / f["toxe"]
    i_spec = 2.0 * n * f["u0"] * cox * (f["w"] / LMIN) * vt ** 2
    vp = (vgb - vth_t + c["voff"]) / n
    log_f_fwd = 2.0 * _log_softplus((vp - vsb) / (2.0 * vt))
    log_f_rev = 2.0 * _log_softplus((vp - vdb) / (2.0 * vt))
    log_diff = log_f_fwd + np.log(
        np.maximum(-np.expm1(np.minimum(log_f_rev - log_f_fwd, 0.0)), 1e-30)
    )
    log10_i_ekv = (np.log(i_spec) + log_diff) / np.log(10.0)

    drain_factor = np.log10(
        np.maximum(1.0 - np.exp(-np.maximum(vds, 0.0) / vt), 1e-12)
    )

    extra = np.column_stack([
        vgs, vds, vbs, vgd, vbd, vgb,
        (vgs - vth_eff) / (n * vt),
        vth_eff,
        n,
        np.log10(f["w"]),
        1.0 / kelvin,
        vt,
        drain_factor,
        np.log10(np.maximum(vds, 1e-9)),
        log10_i_ekv,
        np.log10(f["toxe"]),
        vgs / f["toxe"] * 1e-9,          # oxide field proxies (V/nm)
        vgd / f["toxe"] * 1e-9,
    ])
    return np.column_stack([X_canonical, extra])


ENGINEERED_FEATURES = FEATURES + [
    "vgs", "vds", "vbs", "vgd", "vbd", "vgb", "overdrive_over_nvt",
    "vth_eff", "n_factor", "log10_w", "inv_temp_K", "thermal_voltage",
    "log10_drain_factor", "log10_vds", "log10_i_ekv", "log10_toxe",
    "field_gs", "field_gd",
]


def log_target(Y: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(Y) + LOG_FLOOR)


def magnitude_from_log(L: np.ndarray) -> np.ndarray:
    return np.maximum(10.0 ** L - LOG_FLOOR, 0.0)


class SignModel:
    """Constant sign, or a classifier when the training signs disagree."""

    def __init__(self, constant: float | None = None, classifier=None):
        self.constant = constant
        self.classifier = classifier

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            return np.full(len(F), self.constant)
        return self.classifier.predict(F).astype(float)


FRAMES = ("raw", "canonical")


def model_features(X: np.ndarray, device: str):
    """Features shared by both target frames, plus the canonical swap mask.

    The physics features are computed in the canonical frame (they are
    symmetric), and the raw vd, vs and the swap flag are appended so a model
    can also learn drain/source asymmetric effects. The 16nm LP model card
    is not fully symmetric: it places a ~1e-14 A leakage term on the
    netlist drain terminal regardless of which side is at higher voltage.
    """
    X = np.asarray(X, dtype=float)
    mask = canonical_swap_mask(X, device)
    F = np.column_stack([
        engineer_features(swap_drain_source(X, mask), device),
        X[:, _VD], X[:, _VS], mask.astype(float),
    ])
    return F, mask


MODEL_FEATURE_NAMES = ENGINEERED_FEATURES + ["raw_vd", "raw_vs", "swapped"]


def to_frame(Y: np.ndarray, mask: np.ndarray, frame: str) -> np.ndarray:
    """Express raw-orientation currents in the given target frame."""
    return swap_id_is(Y, mask) if frame == "canonical" else np.asarray(Y, dtype=float)


def from_frame(Y: np.ndarray, mask: np.ndarray, frame: str) -> np.ndarray:
    """Map currents in a target frame back to the raw orientation."""
    return swap_id_is(Y, mask) if frame == "canonical" else Y


def frame_of(name: str) -> str:
    return name.split(":", 1)[0]


class TransistorModel:
    """Predict (id, ig, is, ib) in the original terminal orientation.

    regressors: name -> fitted multi-output regressor on the 4 log targets.
                Names are "<frame>:<family>[<config>]", where the frame is
                "raw" or "canonical" (see module docstring).
    selection:  per raw terminal, the regressor name chosen on validation
    signs:      frame -> list of 4 SignModels in that frame
    """

    def __init__(self, case, regressors, selection, signs):
        self.case = case
        self.device = device_of(case)
        self.regressors = regressors
        self.selection = selection
        self.signs = signs

    def predict_candidate(self, name, F, mask) -> np.ndarray:
        """Full 4-terminal raw-orientation prediction of one regressor."""
        frame = frame_of(name)
        magnitude = magnitude_from_log(self.regressors[name].predict(F))
        sign = np.column_stack([s.predict(F) for s in self.signs[frame]])
        return from_frame(sign * magnitude, mask, frame)

    def predict(self, X) -> np.ndarray:
        F, mask = model_features(X, self.device)
        outputs = {}
        Y = np.empty((len(F), len(TARGETS)))
        for t, name in enumerate(self.selection):
            if name not in outputs:
                outputs[name] = self.predict_candidate(name, F, mask)
            Y[:, t] = outputs[name][:, t]
        X = np.asarray(X, dtype=float)
        rail_models = getattr(self, "rail_models", None)
        if rail_models:
            configs = rail_config(X)
            for config in np.unique(configs):
                rows = configs == config
                Z = None
                for t, terminal in enumerate(TARGETS):
                    entry = rail_models.get((str(config), terminal))
                    if entry is None:
                        continue
                    if Z is None:
                        Z = rail_features(X[rows], str(config))
                    model, sign = entry
                    Y[rows, t] = sign * 10.0 ** model.predict(Z)
        gate_drop = np.max(np.abs(X[:, [_VD, _VS, _VB]] - X[:, [_VG]]), axis=1)
        Y[gate_drop < ZERO_BIAS_GATE_TOLERANCE, _IG] = 0.0
        return Y


def relative_error_percent(y_true, y_pred) -> np.ndarray:
    """Floor-adjusted relative error: |pred - true| / max(|true|, 1e-18 A)."""
    return (
        np.abs(np.asarray(y_pred) - np.asarray(y_true))
        / np.maximum(np.abs(np.asarray(y_true)), ERROR_FLOOR)
        * 100.0
    )
