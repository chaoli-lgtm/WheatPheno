"""Inference only. No training and no OSAVI threshold in output eligibility."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import platform
import zipfile
import numpy as np
import pandas as pd
from .models import predict_frame, model_manifest
from .schema import feature_names
from .paths import project_root

TARGETS = ("PH", "AGB", "Yield")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def _output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path

def _report(kind):
    return {"schema_version": 1, "operation": kind,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "training_performed": False,
            "validation_claim": "Demonstration/application only; not independent validation."}

def _numeric(series):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, np.nan)

def predict_csv(input_path, output_dir, target="AGB", model_path=None, trust_model=False):
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    local = None
    names = feature_names()
    if model_path is not None:
        from .deployment import load_local_model
        local = load_local_model(model_path, trusted=trust_model)
        if target != local.metadata["target"]:
            raise ValueError("Requested target differs from the local model target.")
        names = local.metadata["feature_names"]
    frame = pd.read_csv(input_path)
    if not len(frame):
        raise ValueError("The input table is empty.")
    if target == "PH":
        if "CHM_PCT95" not in frame:
            raise ValueError("PH requires CHM_PCT95 in metres.")
        predicted = _numeric(frame["CHM_PCT95"]) * 100
        if not np.isfinite(predicted).all():
            raise ValueError("PH cannot be estimated without finite CHM_PCT95.")
        missing = np.zeros(len(frame), dtype=int)
    else:
        predicted = local.predict(frame) if local else predict_frame(frame, target)
        missing = (~np.isfinite(frame[names].to_numpy(dtype=float))).sum(axis=1)
    out = _output(output_dir)
    result = pd.DataFrame({"sample_id": frame.get("sample_id", pd.Series([f"row_{i+1}" for i in range(len(frame))])),
                           "prediction": predicted, "unit": "cm" if target == "PH" else "g m-2",
                           "missing_candidate_features": missing})
    if "observed" in frame:
        result["observed"] = _numeric(frame["observed"])
        result["prediction_minus_observed"] = result.prediction - result.observed
    result.to_csv(out / "predictions.csv", index=False)
    rep = _report("table_prediction")
    rep.update(target=target, rows=len(result), rows_with_imputation=int((missing > 0).sum()),
               input_sha256=sha256(input_path), model="CHM PCT95 × 100" if target == "PH" else "fixed Extra Trees",
               output="predictions.csv", observed_values_not_used_for_prediction=True)
    if local:
        rep.update(model=local.metadata["algorithm"], model_sha256=local.metadata["sha256"],
                   feature_contract=local.metadata["feature_contract"])
    save_json(out / "report.json", rep)
    return rep

def _read_boundary(path, crs):
    """GeoJSON assumes EPSG:4326 unless an explicit CRS is present; SHP needs PRJ."""
    from rasterio.warp import transform_geom
    from rasterio.crs import CRS
    path = Path(path)
    if path.suffix.lower() == ".shp":
        import shapefile
        prj = path.with_suffix(".prj")
        if not prj.exists():
            raise ValueError("A SHP boundary requires its .prj CRS definition.")
        source_crs = CRS.from_wkt(prj.read_text(encoding="utf-8-sig"))
        with shapefile.Reader(str(path)) as reader:
            geoms = [s.__geo_interface__ for s in reader.shapes()]
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
        source_crs = obj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
        if source_crs == "LOCAL_METRE":
            if crs is not None:
                raise ValueError("A local demonstration boundary cannot be assigned a geographic CRS.")
            source_crs = None
        geoms = [f["geometry"] for f in obj["features"]] if obj["type"] == "FeatureCollection" else [obj["geometry"]] if obj["type"] == "Feature" else [obj]
    if not geoms or any(g.get("type") not in ("Polygon", "MultiPolygon") for g in geoms):
        raise ValueError("Boundary must contain nonempty polygons.")
    if (source_crs is None) != (crs is None):
        raise ValueError("Boundary and feature grid need compatible coordinate references.")
    if source_crs is not None and CRS.from_user_input(source_crs) != CRS.from_user_input(crs):
        geoms = [transform_geom(source_crs, crs, g) for g in geoms]
    return geoms

def _load_scene(config_path, required_names=None, expected_contract="wheatpheno-frozen-nir2-v1"):
    from affine import Affine
    from rasterio.features import geometry_mask
    cfgpath = Path(config_path).resolve()
    cfg = json.loads(cfgpath.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise ValueError("Scene schema_version must be 1.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("scene_id", "")):
        raise ValueError("scene_id must contain only letters, digits, underscores and hyphens.")
    if expected_contract is not None and cfg.get("feature_contract") != expected_contract:
        raise ValueError(f"Scene feature_contract must match the model: {expected_contract}.")
    # Explicit local path configuration is supported in CLI; no remote URLs/code evaluation.
    resolve = lambda value: (cfgpath.parent / value).resolve()
    src = resolve(cfg["feature_archive"])
    with zipfile.ZipFile(src) as z:
        if sum(x.file_size for x in z.infolist()) > 2_000_000_000:
            raise ValueError("Feature archive exceeds the 2 GB expanded limit; split the scene into tiles.")
    with np.load(src, allow_pickle=False) as z:
        arrays = {key: z[key].astype(np.float32) for key in z.files}
    if "source_coverage" not in arrays:
        raise ValueError("A source_coverage layer is required; input coverage cannot be assumed.")
    names = feature_names() if required_names is None else required_names
    missing = set(names) - arrays.keys()
    if missing:
        raise ValueError(f"Missing feature layers: {sorted(missing)}")
    shape = arrays[names[0]].shape if names else arrays.get("source_coverage", np.array([])).shape
    if len(shape) != 2 or min(shape) == 0 or any(a.shape != shape for a in arrays.values()):
        raise ValueError("All archive arrays must have the same nonempty 2D shape.")
    tx = cfg["transform"]
    if len(tx) != 6 or not np.isfinite(tx).all():
        raise ValueError("transform requires six finite Affine coefficients [a,b,c,d,e,f].")
    transform = Affine(*tx)
    if transform.a <= 0 or transform.e >= 0 or transform.b != 0 or transform.d != 0:
        raise ValueError("This version requires a north-up, axis-aligned grid.")
    if cfg.get("units") != "m":
        raise ValueError("Spatial grid units must be metres.")
    crs = cfg.get("crs")
    if crs is not None:
        from pyproj import CRS
        c = CRS.from_user_input(crs)
        if not c.is_projected or any(abs(a.unit_conversion_factor-1) > 1e-9 for a in c.axis_info[:2]):
            raise ValueError("Use a projected CRS whose horizontal units are metres.")
    boundary = resolve(cfg["boundary"])
    geoms = _read_boundary(boundary, crs)
    inside = geometry_mask(geoms, shape, transform, invert=True)
    if not inside.any():
        raise ValueError("The SHP/geometry does not intersect this grid. Check CRS and transform.")
    if "source_coverage" not in arrays:
        raise ValueError("A source_coverage layer is required; input coverage cannot be assumed.")
    source_coverage = arrays["source_coverage"]
    coverage = np.isfinite(source_coverage) & (source_coverage > 0)
    hashes = {"config": sha256(cfgpath), "features": sha256(src), "boundary": sha256(boundary)}
    if boundary.suffix.lower() == ".shp":
        hashes["boundary_sidecars"] = {p.suffix.lower(): sha256(p) for p in boundary.parent.iterdir()
                                       if p.stem == boundary.stem and p.suffix.lower() in (".shp", ".shx", ".dbf", ".prj", ".cpg")}
    return cfg, arrays, inside, coverage, transform, crs, hashes

def fill_for_display(raw, domain, transform, max_distance):
    """Nearest fill within each connected region, bounded in metres; never changes raw."""
    from scipy.ndimage import distance_transform_edt, label
    display = raw.copy()
    filled = np.zeros(raw.shape, dtype=bool)
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("fill distance must be finite and nonnegative.")
    if max_distance == 0:
        return display, filled
    regions, count = label(domain)
    for region in range(1, count + 1):
        area = regions == region
        good = area & np.isfinite(raw)
        if not good.any():
            continue
        distances, indices = distance_transform_edt(~good, sampling=(abs(transform.e), transform.a), return_indices=True)
        use = area & ~good & (distances <= max_distance)
        display[use] = raw[tuple(indices[:, use])]
        filled[use] = True
    return display, filled

def _write_tif(path, array, transform, crs, quality=False):
    import rasterio
    dtype, nodata = ("uint8", 0) if quality else ("float32", -9999.0)
    values = array.astype(dtype) if quality else np.where(np.isfinite(array), array, nodata).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype=dtype, transform=transform, crs=crs,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(values, 1)
        dst.update_tags(software="WheatPheno 0.4.0", map_mask="geometry_and_source_coverage; no OSAVI threshold")

def _render(array, output, target, limits=None, pixel_size=(1, 1)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from PIL import Image
    cmap = {"PH": plt.get_cmap("viridis"), "AGB": LinearSegmentedColormap.from_list("biomass", ["#24c849", "#c5f235", "#fff03a", "#ff9d12", "#f0180c"]), "Yield": plt.get_cmap("plasma")}[target]
    valid = array[np.isfinite(array)]
    if limits is None:
        limits = [float(valid.min()), float(valid.max())] if len(valid) else [0., 1.]
    lo, hi = map(float, limits)
    if hi <= lo:
        hi = lo + 1
    norm = Normalize(lo, hi)
    rgba = cmap(norm(np.nan_to_num(array, nan=lo)), bytes=True)
    rgba[~np.isfinite(array), 3] = 0
    im = Image.fromarray(rgba)
    # Native cells are retained; nearest enlargement does not invent spatial detail.
    factor = max(1, min(8, 4800 // max(array.shape)))
    sx, sy = pixel_size
    width = max(1, int(array.shape[1] * factor))
    height = max(1, int(array.shape[0] * factor * sy / sx))
    im.resize((width, height), Image.Resampling.NEAREST).save(output / f"{target}_clean.png", dpi=(600, 600))
    fig, ax = plt.subplots(figsize=(9, max(2.4, min(7, 8*height/width))))
    heat = ax.imshow(np.ma.masked_invalid(array), cmap=cmap, norm=norm, aspect=sy/sx, interpolation="nearest")
    ax.set_axis_off()
    unit = "cm" if target == "PH" else "g m$^{-2}$"
    ax.set_title(f"{target} | fixed model application", loc="left", fontsize=12)
    fig.colorbar(heat, ax=ax, shrink=.74, pad=.025, label=unit, extend="both" if len(valid) and (valid.min()<lo or valid.max()>hi) else "neither")
    fig.tight_layout()
    fig.savefig(output / f"{target}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

def map_scene(config_path, output_dir, targets=TARGETS, fill_distance=0.0, model_path=None, trust_model=False):
    targets = tuple(targets)
    if not targets or any(t not in TARGETS for t in targets) or len(set(targets)) != len(targets):
        raise ValueError(f"Targets must be unique members of {TARGETS}.")
    if not np.isfinite(fill_distance) or fill_distance < 0:
        raise ValueError("fill_distance must be nonnegative and finite.")
    local = None
    names = feature_names()
    manifest = model_manifest()
    contract = "wheatpheno-frozen-nir2-v1"
    if model_path is not None:
        from .deployment import load_local_model
        local = load_local_model(model_path, trusted=trust_model)
        if targets != (local.metadata["target"],):
            raise ValueError("Local model mapping requires its single matching target.")
        names = local.metadata["feature_names"]
        manifest = {"models": {targets[0]: local.metadata}}
        contract = local.metadata["feature_contract"]
    ph_only = targets == ("PH",)
    cfg, arrays, inside, coverage, transform, crs, hashes = _load_scene(
        config_path, [] if ph_only else names, None if ph_only else contract)
    minimum = float(cfg.get("minimum_valid_fraction", .5))
    if not .5 <= minimum <= 1:
        raise ValueError("minimum_valid_fraction must be between 0.5 and 1.")
    # Validate every requested target before creating any final products.
    for target in targets:
        if target == "PH":
            layer = cfg.get("ph_feature", "CHM_PCT95")
            if layer not in arrays:
                raise ValueError(f"PH feature {layer} is not present.")
            eligible = inside & coverage & np.isfinite(arrays[layer])
        else:
            overrides = cfg.get("feature_overrides", {}).get(target, {})
            if set(overrides)-set(names) or any(v not in arrays for v in overrides.values()):
                raise ValueError("Feature overrides must map known predictors to existing archive arrays.")
            selected = manifest["models"][target]["selected_features"]
            counts = sum(np.isfinite(arrays[overrides.get(n,n)]).astype(np.uint8) for n in names)
            used = sum(np.isfinite(arrays[overrides.get(n,n)]).astype(np.uint8) for n in selected)
            eligible = inside & coverage & (counts >= np.ceil(minimum*len(names))) & (used >= np.ceil(minimum*len(selected)))
        if not eligible.any():
            raise ValueError(f"No eligible prediction cells for {target}; no map products were created.")
    out = _output(output_dir)
    rep = _report("spatial_mapping")
    rep.update(scene_id=cfg["scene_id"], feature_contract=cfg.get("feature_contract", "height-only"),
               input_sha256=hashes, shape=list(inside.shape), crs=crs,
               transform=list(transform)[:6], domain_cells=int(inside.sum()),
               source_coverage_cells=int((inside & coverage).sum()), osavi_output_mask=False,
               spatial_validation="No independent pixel validation; plot-to-window scale transfer.",
               minimum_valid_fraction=minimum, display_fill_max_distance_m=fill_distance,
               source_notes=cfg.get("source_notes", []), targets={})
    for target in targets:
        raw = np.full(inside.shape, np.nan, dtype=np.float32)
        quality = np.zeros(inside.shape, dtype=np.uint8)
        range_count = np.zeros(inside.shape, dtype=np.uint8)
        selected_range_count = np.zeros(inside.shape, dtype=np.uint8)
        if target == "PH":
            layer = cfg.get("ph_feature", "CHM_PCT95")
            if layer not in arrays:
                raise ValueError(f"PH feature {layer} is not present.")
            valid = inside & coverage & np.isfinite(arrays[layer])
            raw[valid] = arrays[layer][valid]*100
            quality[valid] = 1
            extra = {"method": f"{layer} × 100", "unit": "cm", "ph_feature": layer,
                     "negative_height_cells": int((valid & (raw < 0)).sum()),
                     "negative_height_interpretation": "Negative DSM-ground differences indicate ground-reference or surface artifacts, not negative plant stature; raw values are retained.",
                     "feature_semantics": cfg.get("ph_definition", "Canopy height 95th percentile, in metres.")}
        else:
            overrides = cfg.get("feature_overrides", {}).get(target, {})
            unknown = set(overrides)-set(names)
            if unknown or any(v not in arrays for v in overrides.values()):
                raise ValueError("Feature overrides must map known predictors to existing archive arrays.")
            layers = [arrays[overrides.get(n,n)] for n in names]
            available = np.zeros(inside.shape, dtype=np.uint8)
            for a in layers:
                available += np.isfinite(a)
            selected = manifest["models"][target]["selected_features"]
            selected_available = np.zeros(inside.shape, dtype=np.uint8)
            for n in selected:
                selected_available += np.isfinite(arrays[overrides.get(n,n)])
            valid = inside & coverage & (available >= int(np.ceil(minimum*len(names)))) & (selected_available >= int(np.ceil(minimum*len(selected))))
            idx = np.flatnonzero(valid)
            for start in range(0,len(idx),20000):
                use = idx[start:start+20000]
                frame = pd.DataFrame({n:a.ravel()[use] for n,a in zip(names,layers)})
                raw.ravel()[use] = local.predict(frame) if local else predict_frame(frame, target)
            quality[valid] = np.where(available[valid]==len(names),1,2)
            ranges = manifest["models"][target]["training_ranges"]
            range_features = {}
            for n,a in zip(names,layers):
                bounds = ranges.get(n, {})
                lo,hi = bounds.get("min"), bounds.get("max")
                if lo is not None and hi is not None:
                    tolerance = 1e-7 + 1e-6*max(abs(lo),abs(hi))
                    outside = np.isfinite(a)&((a<lo-tolerance)|(a>hi+tolerance))
                    range_count += outside
                    if n in selected:
                        selected_range_count += outside
                    range_features[n] = {"affected_cells":int((valid&outside).sum()),"selected":n in selected}
            extra = {"method": local.metadata["algorithm"] if local else "fixed Extra Trees", "unit":"g m-2", "feature_overrides":overrides,
                     "complete_input_cells":int((quality==1).sum()), "imputed_input_cells":int((quality==2).sum()),
                     "cells_with_out_of_training_range_features":int((valid&(range_count>0)).sum()),
                     "cells_with_out_of_selected_training_range_features":int((valid&(selected_range_count>0)).sum()),
                     "out_of_range_features":range_features,
                     "range_tolerance":"1e-7 + 1e-6 * max(abs(training_min), abs(training_max))"}
        display, filled = fill_for_display(raw, inside & coverage, transform, fill_distance)
        quality[filled] = 3
        _write_tif(out/f"{target}.tif",raw,transform,crs)
        _write_tif(out/f"{target}_quality.tif",quality,transform,crs,quality=True)
        if target != "PH":
            _write_tif(out/f"{target}_out_of_range_count.tif",np.where(valid,range_count,np.nan),transform,crs)
            _write_tif(out/f"{target}_selected_out_of_range_count.tif",np.where(valid,selected_range_count,np.nan),transform,crs)
        if filled.any():
            _write_tif(out/f"{target}_display.tif",display,transform,crs)
        _render(display,out,target,cfg.get("display_limits",{}).get(target), (transform.a,abs(transform.e)))
        finite = raw[np.isfinite(raw)]
        extra.update(predicted_cells=int(valid.sum()), display_filled_cells=int(filled.sum()),
                     status="complete" if int(valid.sum())==int((inside&coverage).sum()) else "partial",
                     remaining_unmapped_cells=int((inside&coverage&~np.isfinite(display)).sum()),
                     raw_min=float(finite.min()) if len(finite) else None,
                     raw_max=float(finite.max()) if len(finite) else None)
        rep["targets"][target]=extra
    rep["quality_codes"]={"0":"Outside domain, no source coverage, or insufficient inputs", "1":"All required inputs observed", "2":"At least one candidate input imputed by fixed pipeline", "3":"Display-only nearest fill; raw prediction unchanged"}
    if local:
        rep["model_sha256"] = local.metadata["sha256"]
    save_json(out/"report.json",rep)
    return rep

def run_demo(output_dir):
    root=project_root()
    out=_output(output_dir)
    rep=_report("public_demonstration")
    rep["tables"]={t:predict_csv(root/f"examples/tables/{t.lower()}_demo.csv",out/f"tables/{t}",t) for t in ["AGB","Yield"]}
    rep["feature_extraction"]=extract_archive(root/"examples/synthetic/feature_input.npz",out/"synthetic_features")
    rep["spatial_examples_included"]=False
    save_json(out/"report.json",rep)
    return rep


def extract_archive(input_path, output_dir, profile="standard", block_size=10):
    """Extract descriptors from aligned numeric arrays with explicit profiles."""
    from .features import extract_features, block_mean
    from .schema import FROZEN_CONTRACT, STANDARD_CONTRACT
    if profile not in ("standard", "historical"):
        raise ValueError("profile must be standard or historical.")
    with zipfile.ZipFile(input_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 512_000_000:
            raise ValueError("Native array archive exceeds the 512 MB limit; use smaller tiles.")
    with np.load(input_path,allow_pickle=False) as archive:
        required=("B","G","R","RE","NIR","CHM","VEGETATION")
        if set(required)-set(archive.files):
            raise ValueError("Input archive requires B, G, R, RE, NIR, CHM and VEGETATION arrays.")
        source={name:np.asarray(archive[name],dtype=float) for name in required}
    vegetation=source["VEGETATION"]
    historical=profile=="historical"
    result=extract_features({name:source[name] for name in ("B","G","R","RE","NIR")},source["CHM"],vegetation,
                            block_size=block_size,height_mask=(vegetation==1) if historical else None,
                            texture_profile=profile,tcari_profile=profile,include_nir2=historical)
    coverage=np.isfinite(source["CHM"])
    for name in ("B","G","R","RE","NIR"):
        coverage &= np.isfinite(source[name])
    result["source_coverage"]=block_mean(coverage.astype(float),block_size)
    if not any(np.isfinite(v).any() for k,v in result.items() if k!="source_coverage"):
        raise ValueError("No valid descriptors were extracted.")
    out=_output(output_dir)
    np.savez_compressed(out/"features.npz",**result)
    pd.DataFrame({name:value.ravel() for name,value in result.items()}).to_csv(out/"features.csv",index=False)
    report=_report("feature_extraction")
    report.update(profile=profile,feature_contract=FROZEN_CONTRACT if historical else STANDARD_CONTRACT,
                  input_sha256=sha256(input_path),block_size=block_size,
                  grid_shape=list(result["source_coverage"].shape),ndre_formula="(NIR-RE)/(NIR+RE)",
                  nir2_included=historical,osavi_output_mask=False,
                  provenance_note="The profile specifies equations, not independent proof of radiometric or spatial compatibility. Supply original grid metadata and verify task-specific inputs before mapping.")
    save_json(out/"report.json",report)
    return report
