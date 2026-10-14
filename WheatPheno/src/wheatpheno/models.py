"""Verified inference with two fixed, bundled Extra Trees estimators.

Pickle/joblib deserialization can execute code. This module never accepts an
uploaded pickle or a custom model path. Only the two bundled byte sequences
whose hashes are pinned below may be deserialized. A checksum is an integrity
check, not a signature: install this code and its artifacts from a trusted source.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from .compat import register_legacy_selector
from .schema import _asset_directory, feature_names


_TRUSTED = {
    "AGB": (
        "AGB_ExtraTrees_full_refit.joblib",
        "ee0af68ef25c9562adda8d75bca909fd93c5d78d4d16a8daa4b17fb2406ad062",
    ),
    "Yield": (
        "Yield_ExtraTrees_full_refit.joblib",
        "e0dc7a6709a17da44047e6f6380c20cf97fabeec98d080bee64cc23e0c92a882",
    ),
}
_SKLEARN_VERSION = "1.9.1"


def _check_target(target: str) -> None:
    if target not in _TRUSTED:
        raise ValueError("target must be exactly 'AGB' or 'Yield'.")


def model_manifest() -> dict:
    """Return model provenance, aggregate ranges, and scientific limitations."""
    return json.loads((_asset_directory() / "manifest.json").read_text("utf-8"))


def load_model(target: str = "AGB") -> Pipeline:
    """Load a SHA256-verified, frozen model from the packaged asset directory.

    No training or parameter search is performed. Each call loads a fresh
    estimator, so a caller modifying one instance cannot change later calls.
    Exact scikit-learn compatibility is required for historical pickle objects.
    """
    _check_target(target)
    filename, expected_hash = _TRUSTED[target]
    raw = (_asset_directory() / filename).read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Integrity check failed for the bundled {target} model; "
            "deserialization was not attempted."
        )
    if sklearn.__version__ != _SKLEARN_VERSION:
        raise RuntimeError(
            "These frozen artifacts require scikit-learn "
            f"{_SKLEARN_VERSION}; installed version is {sklearn.__version__}. "
            "Use the locked environment supplied with WheatPheno."
        )
    register_legacy_selector()
    model = joblib.load(BytesIO(raw))
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["prep", "model"]:
        raise TypeError("The verified artifact does not have the expected pipeline.")
    stored_names = list(model.named_steps["prep"].transformers_[0][2])
    if stored_names != feature_names():
        raise ValueError("The packaged schema disagrees with the frozen model inputs.")
    return model


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order the 72 inputs without learning any preprocessing.

    Numeric strings are accepted; nonnumeric values and complex numbers are
    rejected. Infinites are treated as missing, and all-missing rows are rejected.
    The frozen model applies its stored training medians to other missing values.
    Additional metadata columns are ignored, including any observation column.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame with named input columns.")
    names = feature_names()
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    duplicates = [name for name in names if (frame.columns == name).sum() != 1]
    if duplicates:
        raise ValueError("Duplicate feature columns: " + ", ".join(duplicates))
    selected = frame.loc[:, names]
    converted = {}
    for name in names:
        try:
            values = pd.to_numeric(selected[name], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature {name!r} contains a nonnumeric value.") from error
        if pd.api.types.is_complex_dtype(values.dtype):
            raise ValueError(f"Feature {name!r} contains complex values.")
        if pd.api.types.is_bool_dtype(values.dtype):
            raise ValueError(f"Feature {name!r} must contain numeric measurements, not booleans.")
        converted[name] = values.astype("float64")
    X = pd.DataFrame(converted, index=frame.index).replace([np.inf, -np.inf], np.nan)
    blank = X.isna().all(axis=1).to_numpy()
    if blank.any():
        rows = np.flatnonzero(blank)[:10].tolist()
        raise ValueError(
            "No finite feature information in row positions "
            f"{rows}; wholly missing rows cannot be predicted."
        )
    return X


def predict_frame(frame: pd.DataFrame, target: str = "AGB") -> np.ndarray:
    """Return fixed-model predictions in g m-2; never fit or tune a model.

    Each row must retain at least half of the candidate inputs and at least half
    of the inputs actually used by the selected estimator. Unused features
    cannot substitute for missing predictors in the fitted model.
    """
    _check_target(target)
    X = prepare_frame(frame)
    if X.empty:
        return np.empty(0, dtype=np.float64)
    selected = model_manifest()["models"][target]["selected_features"]
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(X.columns):
        raise ValueError("The model manifest has an invalid selected-feature contract.")
    candidate_minimum = int(np.ceil(len(X.columns) * 0.5))
    selected_minimum = int(np.ceil(len(selected) * 0.5))
    candidate_count = X.notna().sum(axis=1).to_numpy()
    selected_count = X.loc[:, selected].notna().sum(axis=1).to_numpy()
    unsupported = (candidate_count < candidate_minimum) | (selected_count < selected_minimum)
    if unsupported.any():
        rows = np.flatnonzero(unsupported)[:10].tolist()
        raise ValueError(
            f"Insufficient finite inputs in row positions {rows}: each row requires "
            f"at least {candidate_minimum}/{len(X.columns)} candidate features and "
            f"{selected_minimum}/{len(selected)} selected {target} features "
            "(50% of each group)."
        )
    predictions = np.asarray(load_model(target).predict(X), dtype=np.float64).reshape(-1)
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise RuntimeError("The fixed estimator returned invalid predictions.")
    return predictions
