"""Real GeoTIFF input contracts, with synthetic data and no private sources."""

import json
from pathlib import Path
import shutil

from affine import Affine
import numpy as np
import pandas as pd
import pytest
import rasterio

from wheatpheno import pipeline, prepare
from wheatpheno.models import predict_frame
from wheatpheno.schema import feature_names


@pytest.fixture
def raster_inputs(tmp_path):
    base = tmp_path / "inputs"
    base.mkdir()
    transform = Affine(1, 0, 500000, 0, -1, 3100004)
    crs = "EPSG:32648"

    def write(name, data, *, nodata=None, mask=None, raster_crs=crs, tx=transform):
        path = base / name
        with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                           count=1, dtype="float32", crs=raster_crs, transform=tx, nodata=nodata) as dst:
            dst.write(np.asarray(data, dtype=np.float32), 1)
            if mask is not None:
                dst.write_mask(mask)
        return path

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    raster = write("features.tif", data)
    reference = write("reference.tif", np.ones((4, 4)))
    boundary = {
        "type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
        "features": [{"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [[[500000, 3100000], [500004, 3100000],
                [500004, 3100004], [500000, 3100004], [500000, 3100000]]],
        }}],
    }
    (base / "boundary.geojson").write_text(json.dumps(boundary), encoding="utf-8")
    config = {
        "feature_contract": "wheatpheno-frozen-nir2-v1", "reference_raster": reference.name,
        "boundary": "boundary.geojson", "feature_rasters": {name: raster.name for name in feature_names()},
    }

    def save():
        path = base / "pack.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    return base, config, save, write, tmp_path / "packed"


def test_pack_preserves_zero_and_generates_portable_scene(raster_inputs):
    base, cfg, save, _, out = raster_inputs
    report = prepare.pack_rasters(save(), out)
    assert report["shape"] == [4, 4]
    assert report["osavi_output_mask"] is False
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(feature_names()) | {"source_coverage"}
        assert archive["OSAVI"][0, 0] == 0
        assert archive["CC"][0, 0] == 0
        assert archive["source_coverage"][0, 0] == 1
    text = (out / "scene.json").read_text("utf-8") + (out / "input_provenance.json").read_text("utf-8")
    assert str(base) not in text
    assert "features.tif" not in text
    moved = out.parent / "relocated"
    shutil.copytree(out, moved)
    _, arrays, inside, coverage, _, crs, _ = pipeline._load_scene(moved / "scene.json")
    assert inside.all() and coverage.all()
    assert crs == "EPSG:32648"
    assert arrays["CHM_PCT95"].shape == (4, 4)


def test_average_resampling_excludes_real_nodata_and_keeps_zero(raster_inputs):
    _, cfg, save, write, out = raster_inputs
    values = np.full((4, 4), -9999, dtype=np.float32)
    values[:2, :2] = [[0, 2], [4, -9999]]
    src = write("nodata.tif", values, nodata=-9999)
    cfg["feature_rasters"] = {name: src.name for name in feature_names()}
    cfg["reference_raster"] = src.name
    cfg["block_pixels"] = 2
    report = prepare.pack_rasters(save(), out)
    assert report["shape"] == [2, 2]
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["EXR"][0, 0], 2.0)
        assert np.isnan(archive["EXR"][1, 1])
        np.testing.assert_allclose(archive["source_coverage"][0, 0], 0.75)
        assert archive["source_coverage"][1, 1] == 0


def test_dataset_mask_and_nonfinite_reference_are_missing(raster_inputs):
    _, cfg, save, write, out = raster_inputs
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    values[3, 3] = np.nan
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[0, 1] = 0
    src = write("masked.tif", values, mask=mask)
    cfg["reference_raster"] = src.name
    cfg["feature_rasters"] = {name: src.name for name in feature_names()}
    prepare.pack_rasters(save(), out)
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        assert archive["source_coverage"][0, 0] == 1
        assert archive["source_coverage"][0, 1] == 0
        assert archive["source_coverage"][3, 3] == 0
        assert np.isnan(archive["EXR"][0, 1])
        assert np.isnan(archive["EXR"][3, 3])


def test_declared_nodata_and_dataset_mask_are_both_used_in_average(raster_inputs):
    _, cfg, save, write, out = raster_inputs
    values = np.full((4, 4), 8, dtype=np.float32)
    values[:2, :2] = [[0, 20], [4, -9999]]
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[0, 1] = 0
    src = write("union_mask.tif", values, nodata=-9999, mask=mask)
    cfg["reference_raster"] = src.name
    cfg["feature_rasters"] = {name: src.name for name in feature_names()}
    cfg["block_pixels"] = 2
    prepare.pack_rasters(save(), out)
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["EXR"][0, 0], 2)
        np.testing.assert_allclose(archive["source_coverage"][0, 0], .5)


def test_reference_partial_edge_coverage(raster_inputs):
    _, cfg, save, _, out = raster_inputs
    cfg["block_pixels"] = 3
    prepare.pack_rasters(save(), out)
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["source_coverage"], [[1, 1 / 3], [1 / 3, 1 / 9]])


def test_overrides_and_downstream_fixed_mapping(raster_inputs, monkeypatch):
    _, cfg, save, write, out = raster_inputs
    height = write("height.tif", np.full((4, 4), .85))
    cfg["feature_rasters"]["PH_PCT95_unmasked"] = {"path": height.name, "band": 1}
    cfg["feature_rasters"]["Yield_Texture_NIR_mean"] = height.name
    prepare.pack_rasters(save(), out)
    scene = json.loads((out / "scene.json").read_text("utf-8"))
    assert scene["ph_feature"] == "PH_PCT95_unmasked"
    assert scene["feature_overrides"]["Yield"]["Texture_NIR_mean"] == "Yield_Texture_NIR_mean"
    monkeypatch.setattr(pipeline, "_render", lambda *args, **kwargs: None)
    maps = out.parent / "mapped"
    report = pipeline.map_scene(out / "scene.json", maps, targets=["PH", "AGB", "Yield"])
    with rasterio.open(maps / "PH.tif") as src:
        np.testing.assert_allclose(src.read(1), 85, atol=1e-5)
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        row = {name: archive[name][0, 0] for name in feature_names()}
        expected = predict_frame(pd.DataFrame([row]), "AGB")[0]
        row["Texture_NIR_mean"] = archive["Yield_Texture_NIR_mean"][0, 0]
        yield_expected = predict_frame(pd.DataFrame([row]), "Yield")[0]
    for target, value in [("AGB", expected), ("Yield", yield_expected)]:
        with rasterio.open(maps / f"{target}.tif") as src:
            np.testing.assert_allclose(src.read(1)[0, 0], value, rtol=1e-6)
        assert report["targets"][target]["predicted_cells"] == 16


@pytest.mark.parametrize("change", ["missing", "unknown", "contract", "invalid_band", "bad_fraction", "bad_block"])
def test_reject_invalid_contract_before_output(raster_inputs, change):
    _, cfg, save, _, out = raster_inputs
    if change == "missing":
        del cfg["feature_rasters"]["NIR2"]
    elif change == "unknown":
        cfg["feature_rasters"]["mystery"] = "features.tif"
    elif change == "contract":
        cfg["feature_contract"] = "conventional_features"
    elif change == "invalid_band":
        cfg["feature_rasters"]["NIR2"] = {"path": "features.tif", "band": 99}
    elif change == "bad_fraction":
        cfg["minimum_valid_fraction"] = 0
    else:
        cfg["block_pixels"] = 1.5
    with pytest.raises(ValueError):
        prepare.pack_rasters(save(), out)
    assert not out.exists()


@pytest.mark.parametrize("source", ["reference", "feature", "boundary"])
def test_reject_geographic_crs(raster_inputs, source):
    base, cfg, save, write, out = raster_inputs
    if source == "boundary":
        path = base / "boundary.geojson"
        obj = json.loads(path.read_text("utf-8"))
        obj["crs"]["properties"]["name"] = "EPSG:4326"
        path.write_text(json.dumps(obj), encoding="utf-8")
    else:
        path = write("degrees.tif", np.ones((4, 4)), raster_crs="EPSG:4326")
        if source == "reference":
            cfg["reference_raster"] = path.name
        else:
            cfg["feature_rasters"]["NIR2"] = path.name
    with pytest.raises(ValueError, match="projected CRS"):
        prepare.pack_rasters(save(), out)
    assert not out.exists()


def test_output_cell_limit_and_nonempty_output(raster_inputs, monkeypatch):
    _, _, save, _, out = raster_inputs
    monkeypatch.setattr(prepare, "MAX_CELLS", 15)
    with pytest.raises(ValueError, match="cells"):
        prepare.pack_rasters(save(), out)
    assert not out.exists()
    out.mkdir()
    (out / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare.pack_rasters(save(), out)
    assert (out / "keep.txt").read_text() == "keep"


def test_shapefile_sidecars_are_portable(raster_inputs):
    import shapefile
    from pyproj import CRS

    base, cfg, save, _, out = raster_inputs
    with shapefile.Writer(str(base / "source_boundary.shp")) as writer:
        writer.field("id", "N")
        writer.poly([[[500000, 3100000], [500000, 3100004], [500004, 3100004], [500004, 3100000], [500000, 3100000]]])
        writer.record(1)
    (base / "source_boundary.prj").write_text(CRS.from_epsg(32648).to_wkt(), encoding="utf-8")
    cfg["boundary"] = "source_boundary.shp"
    prepare.pack_rasters(save(), out)
    for suffix in [".shp", ".shx", ".dbf", ".prj"]:
        assert (out / "boundary" / ("boundary" + suffix)).is_file()
    loaded = pipeline._load_scene(out / "scene.json")
    assert loaded[2].all()
