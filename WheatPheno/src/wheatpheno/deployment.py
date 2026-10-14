"""Deploy explicitly trusted local refits with their exported input metadata."""
from pathlib import Path
from io import BytesIO
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn


def read_metadata(path):
    path = Path(path).resolve()
    if path.suffix == ".joblib":
        path = path.with_suffix(".json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    names = meta.get("feature_names")
    selected = meta.get("selected_features")
    if (meta.get("deployment_schema") != 1 or meta.get("target") not in ("AGB", "Yield")
            or not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)
            or not isinstance(selected, list) or not selected
            or len(set(selected)) != len(selected) or not set(selected).issubset(names)):
        raise ValueError("A version 1 exported refit metadata file is required.")
    if meta.get("feature_contract") not in ("wheatpheno-standard-v1", "wheatpheno-frozen-nir2-v1"):
        raise ValueError("Unsupported model feature contract.")
    return meta


class LocalModel:
    def __init__(self, model, metadata):
        self.model, self.metadata = model, metadata

    def predict(self, frame):
        names = self.metadata["feature_names"]
        if any((frame.columns == n).sum() != 1 for n in names):
            raise ValueError("Missing or duplicate model input columns.")
        if any(pd.api.types.is_complex_dtype(frame[n]) or pd.api.types.is_bool_dtype(frame[n]) for n in names):
            raise ValueError("Model inputs must be real numeric measurements.")
        values = frame[names].apply(pd.to_numeric, errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
        selected = self.metadata["selected_features"]
        if ((values.notna().sum(axis=1) < np.ceil(.5 * len(names))) |
                (values[selected].notna().sum(axis=1) < np.ceil(.5 * len(selected)))).any():
            raise ValueError("Insufficient finite inputs for the local model (50% required).")
        if values.empty:
            return np.empty(0)
        result = np.asarray(self.model.predict(values), dtype=float).reshape(-1)
        if len(result) != len(values) or not np.isfinite(result).all():
            raise RuntimeError("Local model returned invalid predictions.")
        return result


def load_local_model(path, *, trusted=False):
    # joblib executes Python during loading. A hash proves consistency only.
    if not trusted:
        raise ValueError("Loading a local joblib requires --trust-model; use only your own or trusted artifacts.")
    path = Path(path).resolve()
    meta = read_metadata(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta.get("sha256"):
        raise ValueError("Local model hash does not match its metadata.")
    if meta.get("scikit_learn") != sklearn.__version__:
        raise ValueError("Local model scikit-learn version differs from this environment.")
    model = joblib.load(BytesIO(raw))
    if list(getattr(model, "feature_names_in_", [])) != meta["feature_names"]:
        raise ValueError("Local model feature order disagrees with metadata.")
    selector = model.named_steps.get("select")
    actual = [meta["feature_names"][int(i)] for i in selector.indices_]
    if actual != meta["selected_features"]:
        raise ValueError("Local model selected predictors disagree with metadata.")
    return LocalModel(model, meta)
