"""Public examples are numerical regression fixtures, not validation data."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from wheatpheno.models import predict_frame
from wheatpheno.pipeline import predict_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_public_ndre_is_conventional_and_nir2_is_separate(target):
    frame=pd.read_csv(ROOT/f"examples/tables/{target.lower()}_demo.csv")
    nir,rededge=frame.Reflectance_NIR,frame.Reflectance_RE
    np.testing.assert_allclose(frame.NDRE,(nir-rededge)/(nir+rededge),atol=1e-12)
    np.testing.assert_allclose(frame.NIR2,2*nir,atol=1e-12)


def test_synthetic_extraction_cli_input_and_standard_profile(tmp_path):
    from wheatpheno.pipeline import extract_archive
    report=extract_archive(ROOT/"examples/synthetic/feature_input.npz",tmp_path/"extracted")
    assert report["feature_contract"]=="wheatpheno-standard-v1"
    with np.load(tmp_path/"extracted/features.npz",allow_pickle=False) as archive:
        assert "NDRE" in archive.files and "NIR2" not in archive.files
        assert np.isfinite(archive["NDRE"]).all()


def example_pair(target):
    directory = ROOT / "examples" / "tables"
    source = directory / f"{target.lower()}_demo.csv"
    expected = pd.read_csv(directory / f"{target.lower()}_expected.csv")
    frame = pd.read_csv(source)
    assert len(frame) > 0
    assert frame["sample_id"].is_unique and expected["sample_id"].is_unique
    assert frame["sample_id"].tolist() == expected["sample_id"].tolist()
    assert np.isfinite(expected["prediction"]).all()
    return source, frame, expected


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_public_fixed_model_predictions_match_golden_values(target):
    _, frame, expected = example_pair(target)
    predicted = predict_frame(frame, target)
    # Small floating-point differences across supported platforms are allowed;
    # the expected file must not be regenerated to make a changed model pass.
    np.testing.assert_allclose(predicted, expected["prediction"], rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_public_csv_pipeline_preserves_golden_values_and_provenance(target, tmp_path):
    source, frame, expected = example_pair(target)
    output = tmp_path / target
    predict_csv(source, output, target)
    actual = pd.read_csv(output / "predictions.csv")
    assert actual["sample_id"].tolist() == expected["sample_id"].tolist()
    np.testing.assert_allclose(actual["prediction"], expected["prediction"], rtol=1e-10, atol=1e-8)
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert report["target"] == target
    assert report["rows"] == len(frame)
    assert report["training_performed"] is False
    assert report["observed_values_not_used_for_prediction"] is True
    assert "not independent validation" in report["validation_claim"]
