"""End-to-end refit deployment and independent terrain/height regression tests."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from wheatpheno import height, pipeline
from wheatpheno.deployment import load_local_model
from wheatpheno.prepare import pack_rasters
from wheatpheno.training import evaluate_table


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_raster(path, values, transform=None):
    values = np.asarray(values, np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype="float32", crs="EPSG:32650", nodata=np.nan,
                       transform=transform or Affine(1, 0, 500000, 0, -1, 3000010)) as dst:
        dst.write(values, 1)
    return path


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).filled(np.nan)


def boundary(path, x=0, y=0, w=4, h=4, crs="LOCAL_METRE"):
    return write_json(path, {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
                            "features": [{"type": "Feature", "properties": {}, "geometry": {
                                "type": "Polygon", "coordinates": [[[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]]]}}]})


@pytest.fixture(scope="module")
def refits(tmp_path_factory):
    root = tmp_path_factory.mktemp("refits")
    rng = np.random.default_rng(81)
    frame = pd.DataFrame(rng.uniform(.1, .8, (24, 3)), columns=["NDRE", "CHM_MEAN", "Texture_NIR_mean"])
    frame["observed"] = 20 + 100*frame.NDRE + 5*frame.CHM_MEAN
    frame["plot"] = np.repeat([f"p{i}" for i in range(12)], 2)
    frame.to_csv(root/"input.csv", index=False)
    config = {"input_csv": "input.csv", "target": "AGB", "protocol": "quick", "outer_folds": 2,
              "inner_folds": 2, "feature_names": list(frame.columns[:3]), "refit": True,
              "models": ["RF", "SVR", "PLSR", "ExtraTrees"], "feature_group_ablation": False}
    evaluate_table(write_json(root/"train.json", config), root/"results")
    return root, frame


@pytest.mark.parametrize("algorithm", ["RF", "SVR", "PLSR", "ExtraTrees"])
def test_refit_table_and_map_reproduce_direct_pipeline_predictions(refits, tmp_path, monkeypatch, algorithm):
    root, frame = refits
    model_path = root/"results/new_refits"/f"AGB_{algorithm}_new_refit.joblib"
    model = load_local_model(model_path, trusted=True)
    expected = joblib.load(model_path).predict(frame[model.metadata["feature_names"]]).reshape(-1)
    np.testing.assert_allclose(model.predict(frame), expected)
    report = pipeline.predict_csv(root/"input.csv", tmp_path/"table", "AGB", model_path, True)
    assert report["model"] == algorithm
    np.testing.assert_allclose(pd.read_csv(tmp_path/"table/predictions.csv").prediction, expected)
    arrays = {n: frame[n].to_numpy()[:16].reshape(4,4) for n in model.metadata["feature_names"]}
    arrays["source_coverage"] = np.ones((4,4))
    np.savez_compressed(tmp_path/"features.npz", **arrays)
    boundary(tmp_path/"boundary.json")
    cfg = {"schema_version": 1, "scene_id": "custom", "feature_contract": "wheatpheno-standard-v1",
           "feature_archive": "features.npz", "boundary": "boundary.json", "units": "m",
           "transform": [1,0,0,0,-1,4], "crs": None}
    monkeypatch.setattr(pipeline, "_render", lambda *a, **kw: None)
    mapped = pipeline.map_scene(write_json(tmp_path/"scene.json", cfg), tmp_path/"map", ["AGB"], model_path=model_path, trust_model=True)
    np.testing.assert_allclose(read_raster(tmp_path/"map/AGB.tif").ravel(), expected[:16], rtol=1e-6)
    assert mapped["model_sha256"] == model.metadata["sha256"]
    with pytest.raises(ValueError, match="target"):
        pipeline.predict_csv(root/"input.csv", tmp_path/"wrong", "Yield", model_path, True)
    cfg["feature_contract"] = "wheatpheno-frozen-ndre-v2"
    with pytest.raises(ValueError, match="feature_contract"):
        pipeline.map_scene(write_json(tmp_path/"bad.json", cfg), tmp_path/"bad", ["AGB"], model_path=model_path, trust_model=True)


def test_local_model_requires_trust_and_matching_hash(refits, tmp_path):
    root, _ = refits
    source = root/"results/new_refits/AGB_ExtraTrees_new_refit.joblib"
    with pytest.raises(ValueError, match="trust-model"):
        load_local_model(source)
    path = tmp_path/source.name
    path.write_bytes(source.read_bytes()+b"modified")
    path.with_suffix(".json").write_bytes(source.with_suffix(".json").read_bytes())
    with pytest.raises(ValueError, match="hash"):
        load_local_model(path, trusted=True)


def test_pack_accepts_only_local_model_subset(refits, tmp_path):
    root, frame = refits
    meta = root/"results/new_refits/AGB_ExtraTrees_new_refit.json"
    names = list(frame.columns[:3])
    for name in names:
        write_raster(tmp_path/f"{name}.tif", np.full((4,4), .5))
    boundary(tmp_path/"field.geojson", 500000, 3000006, crs="EPSG:32650")
    cfg = {"feature_contract": "wheatpheno-standard-v1", "reference_raster": names[0]+".tif",
           "boundary": "field.geojson", "feature_rasters": {n: n+".tif" for n in names}}
    report = pack_rasters(write_json(tmp_path/"pack.json", cfg), tmp_path/"packed", meta)
    assert report["feature_contract"] == "wheatpheno-standard-v1"
    with np.load(tmp_path/"packed/features.npz") as z:
        assert set(z.files) == set(names) | {"source_coverage"}
    from wheatpheno.__main__ import main
    assert main(["map", "--config", str(tmp_path/"packed/scene.json"), "--model", str(meta.with_suffix(".joblib")),
                 "--trust-model", "--output", str(tmp_path/"mapped")]) == 0
    expected = load_local_model(meta.with_suffix(".joblib"), trusted=True).predict(pd.DataFrame([{n: .5 for n in names}]))[0]
    np.testing.assert_allclose(read_raster(tmp_path/"mapped/AGB.tif"), expected, rtol=1e-6)
    assert (tmp_path/"mapped/AGB_clean.png").is_file()


def test_existing_ph_map_accepts_one_height_layer(tmp_path, monkeypatch):
    np.savez(tmp_path/"features.npz", CHM_PCT95=np.ones((4,4))*.8, source_coverage=np.ones((4,4)))
    boundary(tmp_path/"boundary.json")
    cfg = {"schema_version": 1, "scene_id": "height", "feature_archive": "features.npz", "boundary": "boundary.json",
           "units": "m", "transform": [1,0,0,0,-1,4], "crs": None}
    monkeypatch.setattr(pipeline, "_render", lambda *a, **kw: None)
    pipeline.map_scene(write_json(tmp_path/"scene.json", cfg), tmp_path/"out", ["PH"])
    np.testing.assert_allclose(read_raster(tmp_path/"out/PH.tif"), 80)


@pytest.mark.parametrize("method", ["nearest", "idw"])
def test_ground_interpolation_preserves_ground_and_distance_limit(method):
    seed = np.full((5,5), 14.0)
    soil = np.zeros((5,5)); soil[:,0] = 1
    seed[:,0] = 10
    dtm, distance, observed = height.interpolate_ground(seed, soil, Affine(1,0,0,0,-1,5), method=method, radius=1.5)
    np.testing.assert_allclose(dtm[:,:2], 10)
    assert np.isnan(dtm[:,2:]).all()
    assert observed.sum() == 5
    assert np.nanmax(distance) == 1


@pytest.mark.parametrize("terrain", ["dtm", "ground_mask", "seedling_osavi"])
def test_height_from_dsm_and_each_terrain_route(tmp_path, monkeypatch, terrain):
    write_raster(tmp_path/"dsm.tif", np.full((4,4), 11.2))
    cfg = {"dsm": "dsm.tif", "elevation_units": "m", "block_pixels": 2, "percentile": 95}
    if terrain == "dtm":
        write_raster(tmp_path/"dtm.tif", np.full((4,4), 10.0))
        cfg["dtm"] = "dtm.tif"
    else:
        seed = np.full((4,4), 10.4); seed[:,::2] = 10
        mask = np.zeros((4,4)); mask[:,::2] = 1
        write_raster(tmp_path/"seed.tif", seed)
        cfg["seedling_dsm"] = "seed.tif"
        if terrain == "ground_mask":
            write_raster(tmp_path/"mask.tif", mask); cfg[terrain] = "mask.tif"
        else:
            write_raster(tmp_path/"osavi.tif", np.where(mask==1, .1, .6)); cfg[terrain] = "osavi.tif"
    monkeypatch.setattr(height, "_render", lambda *a, **kw: None)
    report = height.estimate_height(write_json(tmp_path/"height.json", cfg), tmp_path/"out")
    np.testing.assert_allclose(read_raster(tmp_path/"out/PH.tif"), 120, atol=.001)
    assert report["osavi_output_mask"] is False
    assert report["predicted_cells"] == 4


def test_height_percentile_uses_native_samples_and_retains_nodata(tmp_path, monkeypatch):
    dsm = np.array([[10,11, np.nan,np.nan], [12,13,np.nan,np.nan]])
    write_raster(tmp_path/"dsm.tif", dsm)
    write_raster(tmp_path/"dtm.tif", np.full(dsm.shape,10))
    cfg = {"dsm":"dsm.tif", "dtm":"dtm.tif", "elevation_units":"m", "block_pixels":2, "percentile":50}
    monkeypatch.setattr(height, "_render", lambda *a, **kw: None)
    height.estimate_height(write_json(tmp_path/"config.json",cfg),tmp_path/"out")
    ph = read_raster(tmp_path/"out/PH.tif")
    assert ph[0,0] == pytest.approx(150)
    assert np.isnan(ph[0,1])


def test_idw_weights_nonconstant_ground():
    seed = np.array([[2., 9., 6.], [9., 9., 9.], [10., 9., 9.]])
    soil = np.zeros((3,3)); soil[0,0] = soil[0,2] = soil[2,0] = 1
    dtm, _, _ = height.interpolate_ground(seed, soil, Affine(1,0,0,0,-1,3), method="idw", radius=5, neighbors=3)
    # At row 0, column 1, squared distances to the three samples are 1, 1, 5.
    assert dtm[0,1] == pytest.approx((2+6+10/5)/(1+1+1/5))


def test_height_offset_boundary_and_real_cli(tmp_path):
    from wheatpheno.__main__ import main
    write_raster(tmp_path/"dsm.tif", np.full((4,4), 9.8))
    write_raster(tmp_path/"dtm.tif", np.full((4,4), 10.0))
    boundary(tmp_path/"field.json", 500000,3000006, w=2, h=4, crs="EPSG:32650")
    cfg = {"dsm":"dsm.tif", "dtm":"dtm.tif", "elevation_units":"m", "block_pixels":1,
           "dsm_offset_m":.5, "boundary":"field.json"}
    config = write_json(tmp_path/"config.json",cfg)
    assert main(["height", "--config",str(config),"--output",str(tmp_path/"out")]) == 0
    ph = read_raster(tmp_path/"out/PH.tif")
    np.testing.assert_allclose(ph[:,:2],30,atol=.001)
    assert np.isnan(ph[:,2:]).all()
    assert (tmp_path/"out/PH_clean.png").is_file()
    assert (tmp_path/"out/PH.png").is_file()


@pytest.mark.parametrize("change, message", [
    ({"elevation_units":"cm"},"elevation_units"),
    ({"seedling_dsm":"seed.tif"},"not both"),
    ({"block_pixels":0},"block_pixels"),
    ({"percentile":101},"percentile"),
])
def test_height_rejects_invalid_configuration_before_output(tmp_path,change,message):
    cfg = {"dsm":"dsm.tif", "dtm":"dtm.tif", "elevation_units":"m", **change}
    with pytest.raises(ValueError,match=message):
        height.estimate_height(write_json(tmp_path/"config.json",cfg),tmp_path/"out")
    assert not (tmp_path/"out").exists()
