"""Public, self-contained contract tests for the fixed model inference layer.

The tests use the small public example when present, otherwise deterministic
midpoints of published aggregate ranges. They never read private training data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

from wheatpheno import models
from wheatpheno.compat import RankSelect
from wheatpheno.schema import feature_names


@pytest.fixture
def frame():
    names = feature_names()
    example = Path(__file__).resolve().parents[1] / "examples" / "tables" / "agb_demo.csv"
    if example.is_file():
        data = pd.read_csv(example)
        assert not data.empty, "The distributed example must contain observations."
        return data.iloc[:3].loc[:, names].copy()
    ranges = models.model_manifest()["models"]["AGB"]["training_ranges"]
    row = {}
    for name in names:
        limits = ranges[name]
        finite = [x for x in (limits["min"], limits["max"]) if x is not None and np.isfinite(x)]
        row[name] = float(np.mean(finite)) if finite else 0.0
    return pd.DataFrame([row, dict(row), dict(row)], columns=names)


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_column_order_and_extra_metadata_do_not_change_predictions(frame, target):
    expected = models.predict_frame(frame, target)
    reordered = frame.loc[:, list(reversed(feature_names()))].copy()
    reordered["observed"] = -999999.0
    reordered["private_note"] = "extra columns are ignored"
    actual = models.predict_frame(reordered, target)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (len(frame),)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_partial_missing_uses_stored_preprocessing(frame, target):
    data = frame.copy()
    data.loc[data.index[0], "CHM_MEAN"] = np.nan
    actual = models.predict_frame(data, target)
    expected = models.load_model(target).predict(data)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("target", ["AGB", "Yield"])
@pytest.mark.parametrize("infinite", [np.inf, -np.inf])
def test_infinity_is_treated_as_missing(frame, target, infinite):
    data = frame.copy()
    data.loc[data.index[0], "CHM_MEAN"] = infinite
    missing = data.replace([np.inf, -np.inf], np.nan)
    np.testing.assert_array_equal(
        models.predict_frame(data, target), models.predict_frame(missing, target)
    )
    assert np.isnan(models.prepare_frame(data).iloc[0]["CHM_MEAN"])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_wholly_missing_rows_are_rejected(frame, value):
    data = frame.copy()
    data.loc[data.index[0], feature_names()] = value
    with pytest.raises(ValueError, match="No finite feature information"):
        models.predict_frame(data)


def test_missing_feature_is_rejected(frame):
    with pytest.raises(ValueError, match="Missing required feature columns"):
        models.predict_frame(frame.drop(columns="CHM_MEAN"))


def test_duplicate_feature_is_rejected(frame):
    data = pd.concat([frame, frame[["CHM_MEAN"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate feature columns"):
        models.predict_frame(data)


def test_nonnumeric_feature_is_rejected(frame):
    data = frame.copy()
    data["CHM_MEAN"] = "not a measurement"
    with pytest.raises(ValueError, match="nonnumeric"):
        models.predict_frame(data)


def test_complex_feature_is_rejected(frame):
    data = frame.copy()
    data["CHM_MEAN"] = 1.0 + 2.0j
    with pytest.raises(ValueError, match="complex"):
        models.predict_frame(data)


def test_boolean_feature_is_rejected(frame):
    data = frame.copy()
    data["CHM_MEAN"] = True
    with pytest.raises(ValueError, match="booleans"):
        models.predict_frame(data)


def test_input_must_be_dataframe(frame):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        models.predict_frame(frame.to_numpy())


@pytest.mark.parametrize("target", ["agb", "yield", "PH", "../../other.joblib"])
def test_target_is_a_fixed_allowlist(frame, target):
    with pytest.raises(ValueError, match="target must be exactly"):
        models.predict_frame(frame, target)


def test_empty_input_returns_empty_vector(frame):
    assert models.predict_frame(frame.iloc[:0]).shape == (0,)


def test_unused_candidates_cannot_replace_missing_selected_yield_features(frame):
    data = frame.copy()
    selected = models.model_manifest()["models"]["Yield"]["selected_features"]
    data.loc[:, selected] = np.nan
    with pytest.raises(ValueError, match="Insufficient finite inputs"):
        models.predict_frame(data, "Yield")


def test_one_finite_candidate_does_not_support_prediction(frame):
    data = frame.copy()
    data.loc[:, :] = np.nan
    data["EXR"] = 0.25
    with pytest.raises(ValueError, match="Insufficient finite inputs"):
        models.predict_frame(data, "Yield")


def test_less_than_half_selected_features_is_rejected_with_other_candidates_present(frame):
    data = frame.copy()
    selected = models.model_manifest()["models"]["Yield"]["selected_features"]
    data.loc[:, selected[7:]] = np.nan
    with pytest.raises(ValueError, match="Insufficient finite inputs"):
        models.predict_frame(data, "Yield")


def test_less_than_half_candidates_is_rejected_despite_complete_selected_inputs(frame):
    data = frame.copy()
    selected = models.model_manifest()["models"]["Yield"]["selected_features"]
    other = [name for name in feature_names() if name not in selected]
    data.loc[:, other[19:]] = np.nan  # 16 selected + 19 other = 35/72
    with pytest.raises(ValueError, match="Insufficient finite inputs"):
        models.predict_frame(data, "Yield")


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_exactly_half_candidate_and_selected_inputs_is_allowed(frame, target):
    names = feature_names()
    selected = models.model_manifest()["models"][target]["selected_features"]
    half_selected = selected[:int(np.ceil(len(selected) / 2))]
    other = [name for name in names if name not in selected]
    kept = half_selected + other[:36 - len(half_selected)]
    data = frame.iloc[:1].copy()
    data.loc[:, :] = np.nan
    data.loc[:, kept] = frame.iloc[:1].loc[:, kept].fillna(0.25).to_numpy()
    assert data.notna().sum(axis=1).iloc[0] == 36
    assert np.isfinite(models.predict_frame(data, target)).all()


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_predict_does_not_train(frame, target, monkeypatch):
    def forbidden_fit(*args, **kwargs):
        pytest.fail("An inference operation attempted to train a model.")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    monkeypatch.setattr(ExtraTreesRegressor, "fit", forbidden_fit)
    monkeypatch.setattr(RankSelect, "fit", forbidden_fit)
    assert np.isfinite(models.predict_frame(frame, target)).all()


def test_compatibility_selector_rejects_training():
    with pytest.raises(RuntimeError, match="frozen"):
        RankSelect().fit(np.ones((2, 2)), np.ones(2))


def test_tampered_artifact_fails_before_deserialization(tmp_path, monkeypatch):
    artifact = tmp_path / "AGB_ExtraTrees_full_refit.joblib"
    artifact.write_bytes(b"Not a trusted model artifact")
    monkeypatch.setattr(models, "_asset_directory", lambda: tmp_path)

    def forbidden_load(*args, **kwargs):
        pytest.fail("An untrusted artifact reached pickle deserialization.")

    monkeypatch.setattr(models.joblib, "load", forbidden_load)
    with pytest.raises(ValueError, match="Integrity check failed"):
        models.load_model("AGB")


def test_incompatible_sklearn_fails_before_deserialization(monkeypatch):
    monkeypatch.setattr(models.sklearn, "__version__", "incompatible-version")

    def forbidden_load(*args, **kwargs):
        pytest.fail("An incompatible environment reached deserialization.")

    monkeypatch.setattr(models.joblib, "load", forbidden_load)
    with pytest.raises(RuntimeError, match="require scikit-learn"):
        models.load_model("AGB")
