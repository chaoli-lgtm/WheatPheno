"""Scientific boundary tests for mapping, using synthetic public-contract scenes."""

import json

from affine import Affine
import numpy as np
import pandas as pd
import pytest
import rasterio

from wheatpheno import pipeline
from wheatpheno.models import model_manifest, predict_frame
from wheatpheno.schema import feature_names


def rectangle(x0, y0, x1, y1):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


@pytest.fixture
def scene_factory(tmp_path, monkeypatch):
    """Small local metre grids require neither private rasters nor original plots."""
    counter = 0
    ranges = model_manifest()["models"]["AGB"]["training_ranges"]
    names = feature_names()
    # These tests exercise numerical maps; separate UI tests cover image styling.
    monkeypatch.setattr(pipeline, "_render", lambda *args, **kwargs: None)

    def make(*, mutate=None, config_changes=None, geometries=None):
        nonlocal counter
        counter += 1
        folder = tmp_path / f"scene_{counter}"
        folder.mkdir()
        arrays = {}
        for name in names:
            bounds = ranges[name]
            numbers = [v for v in (bounds["min"], bounds["max"]) if v is not None and np.isfinite(v)]
            value = float(np.mean(numbers)) if numbers else 0.0
            arrays[name] = np.full((8, 12), value, dtype=np.float32)
        arrays["source_coverage"] = np.ones((8, 12), dtype=np.float32)
        if mutate is not None:
            mutate(arrays)
        np.savez_compressed(folder / "features.npz", **arrays)
        geoms = geometries if geometries is not None else [rectangle(1, 1, 11, 7)]
        boundary = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "LOCAL_METRE"}},
            "features": [{"type": "Feature", "properties": {}, "geometry": g} for g in geoms],
        }
        (folder / "boundary.geojson").write_text(json.dumps(boundary), encoding="utf-8")
        config = {
            "schema_version": 1,
            "scene_id": "synthetic_test",
            "feature_contract": "wheatpheno-frozen-nir2-v1",
            "feature_archive": "features.npz",
            "boundary": "boundary.geojson",
            "transform": [1, 0, 0, 0, -1, 8],
            "crs": None,
            "units": "m",
        }
        config.update(config_changes or {})
        path = folder / "scene.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path, arrays, folder / "output"

    return make


def read_map(path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).filled(np.nan)


@pytest.mark.parametrize("target", ["AGB", "Yield"])
def test_complete_low_osavi_zero_cover_cells_are_still_predicted(scene_factory, target):
    def modify(arrays):
        arrays["OSAVI"][:] = -0.4
        arrays["CC"][:] = 0.0

    scene, arrays, out = scene_factory(mutate=modify)
    report = pipeline.map_scene(scene, out, targets=[target])
    actual = read_map(out / f"{target}.tif")
    assert np.isfinite(actual[1:7, 1:11]).all()
    # Compare an interior cell with direct fixed-model inference, not a mock.
    row = pd.DataFrame([{name: arrays[name][3, 3] for name in feature_names()}])
    expected = predict_frame(row, target)[0]
    np.testing.assert_allclose(actual[3, 3], expected, rtol=1e-6)
    assert report["osavi_output_mask"] is False
    assert report["targets"][target]["predicted_cells"] == 60


def test_geometry_and_source_coverage_are_respected(scene_factory):
    def modify(arrays):
        arrays["source_coverage"][3, 3] = 0

    scene, _, out = scene_factory(mutate=modify)
    pipeline.map_scene(scene, out, targets=["PH", "AGB", "Yield"], fill_distance=100)
    for target in pipeline.TARGETS:
        actual = read_map(out / f"{target}.tif")
        assert np.isnan(actual[0, :]).all()
        assert np.isnan(actual[-1, :]).all()
        assert np.isnan(actual[:, 0]).all()
        assert np.isnan(actual[:, -1]).all()
        assert np.isnan(actual[3, 3])
        display_path = out / f"{target}_display.tif"
        if display_path.exists():
            assert np.isnan(read_map(display_path)[3, 3])


def test_no_predictions_from_wholly_missing_feature_cells(scene_factory):
    def modify(arrays):
        for name in feature_names():
            arrays[name][3, 3] = np.nan

    scene, _, out = scene_factory(mutate=modify)
    report = pipeline.map_scene(scene, out, targets=["AGB", "Yield"])
    for target in ["AGB", "Yield"]:
        actual = read_map(out / f"{target}.tif")
        assert np.isnan(actual[3, 3])
        assert np.isfinite(actual[3, 4])
        assert report["targets"][target]["predicted_cells"] == 59


def test_unused_features_cannot_substitute_for_all_missing_selected_inputs(scene_factory):
    selected = model_manifest()["models"]["Yield"]["selected_features"]

    def modify(arrays):
        for name in selected:
            arrays[name][3, 3] = np.nan

    scene, _, out = scene_factory(mutate=modify)
    pipeline.map_scene(scene, out, targets=["Yield"])
    actual = read_map(out / "Yield.tif")
    assert np.isnan(actual[3, 3]), (
        "All 16 inputs actually used by the yield estimator are missing. "
        "The other 56 unused candidates cannot establish prediction support."
    )
    assert np.isfinite(actual[3, 4])


def test_nonfinite_coverage_cannot_authorize_prediction(scene_factory):
    def modify(arrays):
        arrays["source_coverage"][3, 3] = np.inf

    scene, _, out = scene_factory(mutate=modify)
    pipeline.map_scene(scene, out, targets=["PH", "Yield"])
    for target in ["PH", "Yield"]:
        assert np.isnan(read_map(out / f"{target}.tif")[3, 3])


def test_source_coverage_cannot_be_assumed(scene_factory):
    scene, _, out = scene_factory(mutate=lambda arrays: arrays.pop("source_coverage"))
    with pytest.raises(ValueError, match="source_coverage"):
        pipeline.map_scene(scene, out, targets=["PH"])


def test_empty_target_cannot_return_success_or_partial_products(scene_factory):
    def modify(arrays):
        for name in feature_names():
            arrays[name][:] = np.nan
    scene, _, out = scene_factory(mutate=modify)
    with pytest.raises(ValueError, match="No eligible prediction cells"):
        pipeline.map_scene(scene, out, targets=["AGB", "Yield"])
    assert not out.exists()


def test_unused_feature_range_and_roundoff_do_not_inflate_selected_warning(scene_factory):
    ranges=model_manifest()["models"]["Yield"]["training_ranges"]
    selected=model_manifest()["models"]["Yield"]["selected_features"]
    unused=next(n for n in feature_names() if n not in selected)
    def modify(arrays):
        for name in feature_names():
            arrays[name][:]=ranges[name]["max"]
        arrays[unused][:]=ranges[unused]["max"]+100
        name=selected[0]
        arrays[name][:]=ranges[name]["max"]+1e-9
    scene, _, out = scene_factory(mutate=modify)
    rep=pipeline.map_scene(scene,out,targets=["Yield"])["targets"]["Yield"]
    assert rep["cells_with_out_of_training_range_features"]==60
    assert rep["cells_with_out_of_selected_training_range_features"]==0


def test_negative_surface_heights_are_reported_and_preserved(scene_factory):
    def modify(arrays):
        arrays["CHM_PCT95"][3, 3] = -.11
    scene, _, out = scene_factory(mutate=modify)
    report = pipeline.map_scene(scene, out, targets=["PH"])
    assert report["targets"]["PH"]["negative_height_cells"] == 1
    assert read_map(out / "PH.tif")[3, 3] == pytest.approx(-11)


def test_missing_ph_is_not_silently_imputed_as_a_model_prediction(scene_factory):
    def modify(arrays):
        arrays["CHM_PCT95"][3, 3] = np.nan

    scene, arrays, out = scene_factory(mutate=modify)
    report = pipeline.map_scene(scene, out, targets=["PH", "AGB"], fill_distance=1.1)
    ph_raw = read_map(out / "PH.tif")
    assert np.isnan(ph_raw[3, 3])
    np.testing.assert_allclose(ph_raw[3, 4], arrays["CHM_PCT95"][3, 4] * 100)
    assert np.isfinite(read_map(out / "PH_display.tif")[3, 3])
    assert report["targets"]["PH"]["display_filled_cells"] == 1
    assert report["targets"]["PH"]["predicted_cells"] == 59
    # Partial missingness is supported by the fixed AGB preprocessing pipeline.
    assert np.isfinite(read_map(out / "AGB.tif")[3, 3])
    with rasterio.open(out / "PH_quality.tif") as src:
        assert src.read(1)[3, 3] == 3  # explicitly display only


def test_display_fill_preserves_raw_and_cannot_borrow_from_disconnected_field(scene_factory):
    def modify(arrays):
        for name in feature_names():
            arrays[name][:, 7:11] = np.nan  # whole second field has no donors
            arrays[name][3, 2] = np.nan  # a supported gap in the first field

    polygons = [rectangle(1, 1, 4, 7), rectangle(7, 1, 11, 7)]
    scene, _, out = scene_factory(mutate=modify, geometries=polygons)
    report = pipeline.map_scene(scene, out, targets=["AGB"], fill_distance=100)
    raw = read_map(out / "AGB.tif")
    display = read_map(out / "AGB_display.tif")
    finite = np.isfinite(raw)
    np.testing.assert_array_equal(display[finite], raw[finite])
    assert np.isnan(raw[3, 2]) and np.isfinite(display[3, 2])
    assert np.isnan(display[1:7, 7:11]).all()
    assert np.isnan(display[:, 4:7]).all()
    assert report["targets"]["AGB"]["display_filled_cells"] == 1


def test_fill_uses_metres_and_does_not_mutate_input():
    raw = np.full((3, 3), np.nan)
    raw[1, 1] = 42.0
    before = raw.copy()
    domain = np.ones((3, 3), dtype=bool)
    # Vertical steps are 2 m, horizontal steps 1 m.
    display, filled = pipeline.fill_for_display(raw, domain, Affine(1, 0, 0, 0, -2, 6), 1.5)
    np.testing.assert_array_equal(raw, before)
    assert np.isfinite(display[1]).all()
    assert np.isnan(display[0, 1]) and np.isnan(display[2, 1])
    assert filled.sum() == 2


def test_feature_overrides_are_target_specific(scene_factory, monkeypatch):
    def modify(arrays):
        arrays["Texture_NIR_mean"][:] = 0.25
        arrays["Yield_Texture_NIR_mean"] = np.full((8, 12), 0.75, dtype=np.float32)

    scene, _, out = scene_factory(
        mutate=modify,
        config_changes={"feature_overrides": {"Yield": {"Texture_NIR_mean": "Yield_Texture_NIR_mean"}}},
    )
    seen = {}

    def capture(frame, target):
        seen[target] = frame.copy()
        return frame["Texture_NIR_mean"].to_numpy() * 100

    monkeypatch.setattr(pipeline, "predict_frame", capture)
    pipeline.map_scene(scene, out, targets=["AGB", "Yield"])
    assert (seen["AGB"]["Texture_NIR_mean"] == 0.25).all()
    assert (seen["Yield"]["Texture_NIR_mean"] == 0.75).all()
    assert read_map(out / "AGB.tif")[3, 3] == 25.0
    assert read_map(out / "Yield.tif")[3, 3] == 75.0


@pytest.mark.parametrize(
    "changes",
    [
        {"crs": "EPSG:4326"},  # geographic degrees are not metre distances
        {"crs": "EPSG:2263"},  # projected but US survey feet
        {"crs": "EPSG:32650"},  # local boundary cannot silently acquire this CRS
        {"units": "degrees"},
        {"transform": [1, 0.1, 0, 0, -1, 8]},  # unsupported rotated grid
        {"transform": [0, 0, 0, 0, -1, 8]},
        {"transform": [1, 0, 0, 0, 1, 8]},
        {"transform": [1, 0, 0, 0, -1]},
        {"transform": [1, 0, 0, 0, -1, float("nan")]},
    ],
)
def test_incompatible_crs_or_grid_is_rejected(scene_factory, changes):
    scene, _, out = scene_factory(config_changes=changes)
    with pytest.raises(ValueError):
        pipeline.map_scene(scene, out, targets=["AGB"])
    assert not out.exists()


def test_misaligned_feature_layer_shape_is_rejected(scene_factory):
    def modify(arrays):
        arrays["CHM_MEAN"] = arrays["CHM_MEAN"][:-1, :]

    scene, _, out = scene_factory(mutate=modify)
    with pytest.raises(ValueError, match="same nonempty 2D shape"):
        pipeline.map_scene(scene, out, targets=["AGB"])
    assert not out.exists()


def test_boundary_that_does_not_intersect_grid_is_rejected(scene_factory):
    scene, _, out = scene_factory(geometries=[rectangle(100, 100, 110, 110)])
    with pytest.raises(ValueError, match="does not intersect"):
        pipeline.map_scene(scene, out, targets=["AGB"])


def test_nonempty_output_cannot_be_overwritten(scene_factory):
    scene, _, out = scene_factory()
    out.mkdir()
    marker = out / "existing_result.txt"
    marker.write_text("preserve this result", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        pipeline.map_scene(scene, out, targets=["AGB"])
    assert marker.read_text("utf-8") == "preserve this result"
    assert sorted(p.name for p in out.iterdir()) == ["existing_result.txt"]
