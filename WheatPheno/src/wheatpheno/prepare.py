"""Pack precomputed feature rasters into the fixed mapping input contract.

This is grid alignment, not feature extraction or model fitting. Raster values
are never filtered by OSAVI or canopy cover, and valid zeroes remain valid.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

from .schema import feature_names


MAX_CELLS = 4_000_000
CONTRACT = "wheatpheno-frozen-ndre-v2"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _local_path(value: object, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError("Input paths must name local files, not URLs.")
    path = (base / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"A configured input file does not exist: {path.name}")
    return path


def _metre_crs(crs, role: str):
    from pyproj import CRS

    if crs is None:
        raise ValueError(f"{role} needs an explicit projected CRS in metres.")
    parsed = CRS.from_user_input(crs)
    if not parsed.is_projected or len(parsed.axis_info) < 2 or any(
        abs(axis.unit_conversion_factor - 1.0) > 1e-9 for axis in parsed.axis_info[:2]
    ):
        raise ValueError(f"{role} must use a projected CRS in metres, not geographic degrees or feet.")
    return parsed


def _band(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{role} band must be a positive integer.")
    return value


def _boundary_files(path: Path) -> list[Path]:
    """Require a declared metre CRS and return only recognized geometry sidecars."""
    if path.suffix.lower() == ".shp":
        files = {p.suffix.lower(): p for p in path.parent.iterdir() if p.stem == path.stem}
        missing = {".shp", ".shx", ".dbf", ".prj"} - files.keys()
        if missing:
            raise ValueError("SHP boundary is missing sidecars: " + ", ".join(sorted(missing)))
        _metre_crs(files[".prj"].read_text(encoding="utf-8-sig"), "Boundary")
        return [files[suffix] for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx") if suffix in files]
    if path.suffix.lower() not in (".json", ".geojson"):
        raise ValueError("Boundary must be a SHP or GeoJSON file.")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    declared = obj.get("crs", {}).get("properties", {}).get("name")
    _metre_crs(declared, "GeoJSON boundary")
    return [path]


def _coverage(reference, band: int, block: int, height: int, width: int) -> np.ndarray:
    """Average true reference validity; zero reflectance is not a missing value."""
    from rasterio.windows import Window

    result = np.zeros((height, width), dtype=np.float32)
    step = max(1, 512 // block)
    for row in range(0, height, step):
        for col in range(0, width, step):
            nr, nc = min(step, height - row), min(step, width - col)
            src_h = min(nr * block, reference.height - row * block)
            src_w = min(nc * block, reference.width - col * block)
            values = reference.read(band, window=Window(col * block, row * block, src_w, src_h), masked=True)
            valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
            nodata = reference.nodatavals[band - 1]
            if nodata is not None:
                valid &= values.data != nodata
            padded = np.zeros((nr * block, nc * block), dtype=np.float32)
            padded[:src_h, :src_w] = valid
            result[row:row + nr, col:col + nc] = padded.reshape(nr, block, nc, block).mean(axis=(1, 3))
    return result


def _write_array(archive: zipfile.ZipFile, name: str, values: np.ndarray) -> None:
    with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)


def _normalized_band(src, band: int, destination: Path) -> None:
    """Materialize the union of per-band NoData, masks and nonfinite values.

    GDAL's warp can prioritize explicit source NoData over a dataset mask.
    Normalizing a single band in bounded windows avoids losing either mask,
    without holding a potentially enormous source raster in memory.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(destination, "w", driver="GTiff", width=src.width, height=src.height,
                       count=1, dtype="float32", transform=src.transform, crs=src.crs,
                       nodata=np.nan, compress="deflate", BIGTIFF="IF_SAFER") as dst:
        nodata = src.nodatavals[band - 1]
        for row in range(0, src.height, 512):
            for col in range(0, src.width, 512):
                window = Window(col, row, min(512, src.width - col), min(512, src.height - row))
                values = src.read(band, window=window, masked=True)
                valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                if nodata is not None:
                    valid &= values.data != nodata
                data = np.where(valid, values.data, np.nan).astype(np.float32)
                dst.write(data, 1, window=window)


def pack_rasters(config_path, output_dir, model_metadata=None) -> dict:
    """Align model-specific precomputed rasters and package a portable ``map_scene`` input.

    Parameters are read from JSON, never evaluated. Output is a new or empty
    directory. Absolute input paths stay in the local config, not the package.
    """
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.vrt import WarpedVRT

    from .pipeline import _read_boundary

    config_file = Path(config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8-sig"))
    contract = CONTRACT
    names = feature_names()
    if model_metadata is not None:
        from .deployment import read_metadata
        metadata = read_metadata(model_metadata)
        names, contract = metadata["feature_names"], metadata["feature_contract"]
    if not isinstance(cfg, dict) or cfg.get("feature_contract") != contract:
        raise ValueError(f"Declare feature_contract={contract} after checking the feature definitions.")
    supplied = cfg.get("feature_rasters")
    if not isinstance(supplied, dict):
        raise ValueError("feature_rasters must map the exact model feature names to raster inputs.")
    missing = set(names) - supplied.keys()
    allowed = set(names) | {"Yield_" + name for name in names} | {"PH_PCT95_unmasked"}
    unknown = set(supplied) - allowed
    if missing or unknown:
        raise ValueError(f"Feature names do not match the fixed contract; missing={sorted(missing)}, unknown={sorted(unknown)}")
    block = cfg.get("block_pixels", 1)
    if isinstance(block, bool) or not isinstance(block, int) or not 1 <= block <= 256:
        raise ValueError("block_pixels must be an integer from 1 to 256.")
    minimum = cfg.get("minimum_valid_fraction", 0.5)
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not np.isfinite(minimum) or not .5 <= minimum <= 1:
        raise ValueError("minimum_valid_fraction must be finite and between 0.5 and 1.")
    scene_id = cfg.get("scene_id", "prepared_scene")
    if not isinstance(scene_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id):
        raise ValueError("scene_id must contain only letters, digits, underscores and hyphens.")
    reference = _local_path(cfg.get("reference_raster"), config_file.parent)
    ref_band = _band(cfg.get("reference_band", 1), "Reference")
    boundary = _local_path(cfg.get("boundary"), config_file.parent)
    sidecars = _boundary_files(boundary)
    sources = {}
    for name, specification in supplied.items():
        if isinstance(specification, str):
            source, band = specification, 1
        elif isinstance(specification, dict) and set(specification) <= {"path", "band"} and "path" in specification:
            source, band = specification["path"], _band(specification.get("band", 1), name)
        else:
            raise ValueError(f"Invalid raster specification for {name}; use a path or {{path, band}}.")
        sources[name] = (_local_path(source, config_file.parent), band)
    out = Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Output directory must be new or empty; source data are never overwritten.")
    with rasterio.open(reference) as ref:
        crs = _metre_crs(ref.crs, "Reference raster")
        tx = ref.transform
        if tx.a <= 0 or tx.e >= 0 or tx.b != 0 or tx.d != 0:
            raise ValueError("Reference raster must use a north-up, axis-aligned grid.")
        if ref_band > ref.count:
            raise ValueError("Reference band is outside the raster band count.")
        if np.dtype(ref.dtypes[ref_band - 1]).kind not in "iuf":
            raise ValueError("Reference must be a real numeric raster.")
        height, width = math.ceil(ref.height / block), math.ceil(ref.width / block)
        if height * width > MAX_CELLS:
            raise ValueError(f"Output exceeds {MAX_CELLS:,} cells; increase block_pixels or split the scene into tiles.")
        if (height * width * 4 + 1024) * (len(sources) + 1) > 2_000_000_000:
            raise ValueError("Expanded feature archive would exceed 2 GB; increase block_pixels or split the scene.")
        transform = tx @ Affine.scale(block)
        geoms = _read_boundary(boundary, ref.crs)
        inside = geometry_mask(geoms, (height, width), transform, invert=True)
        if not inside.any():
            raise ValueError("Boundary does not intersect the output grid; check CRS and extent.")
        coverage = _coverage(ref, ref_band, block, height, width)
        reference_auxiliary = [Path(item).resolve() for item in ref.files if Path(item).resolve() != reference]
    # Validate the complete input set before creating any output artifacts.
    for name, (path, band) in sources.items():
        with rasterio.open(path) as src:
            _metre_crs(src.crs, f"Feature {name}")
            if band > src.count:
                raise ValueError(f"Feature {name} band is outside the raster band count.")
            if np.dtype(src.dtypes[band - 1]).kind not in "iuf":
                raise ValueError(f"Feature {name} must be a real numeric raster.")
    if not np.any(inside & (coverage > 0)):
        raise ValueError("Reference has no valid source coverage inside the boundary.")
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def file_hash(path):
        if path not in hashes:
            hashes[path] = _hash(path)
        return hashes[path]

    feature_records = {}
    with TemporaryDirectory(prefix="wheatpheno_pack_") as temp, zipfile.ZipFile(out / "features.npz", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        normalized = Path(temp) / "normalized.tif"
        previous = None
        for name, (path, band) in sources.items():
            with rasterio.open(path) as src:
                if previous != (path, band):
                    _normalized_band(src, band, normalized)
                    previous = (path, band)
                with rasterio.open(normalized) as clean, WarpedVRT(
                    clean, crs=crs.to_wkt(), transform=transform, width=width, height=height,
                    dtype="float32", src_nodata=np.nan, nodata=np.nan, resampling=Resampling.average,
                ) as vrt:
                    data = vrt.read(1, masked=True).filled(np.nan).astype(np.float32)
                data[~np.isfinite(data)] = np.nan
                _write_array(archive, name, data)
                feature_records[name] = {
                    "sha256": file_hash(path), "band": band,
                    "source_crs": src.crs.to_string(),
                    "source_shape": [src.height, src.width],
                    "output_finite_cells": int(np.isfinite(data).sum()),
                    "auxiliary_file_hashes": [
                        {"suffix": Path(item).suffix, "sha256": file_hash(Path(item).resolve())}
                        for item in src.files if Path(item).resolve() != path
                    ],
                }
        _write_array(archive, "source_coverage", coverage)
    boundary_dir = out / "boundary"
    boundary_dir.mkdir()
    boundary_records = {}
    for path in sidecars:
        destination = boundary_dir / ("boundary" + path.suffix.lower())
        shutil.copyfile(path, destination)
        boundary_records[destination.name] = file_hash(path)
    scene = {
        "schema_version": 1, "scene_id": scene_id, "feature_contract": contract,
        "feature_archive": "features.npz",
        "boundary": "boundary/boundary" + boundary.suffix.lower(),
        "transform": list(transform)[:6], "crs": crs.to_string(), "units": "m",
        "minimum_valid_fraction": float(minimum),
        "source_notes": [
            "Precomputed model features aligned by area average; no formulas recalculated.",
            "Footprint follows the boundary and true reference coverage; no OSAVI or CC threshold.",
            "Grid alignment does not validate transfer from plot summaries to local spatial inputs.",
        ],
    }
    overrides = {name.removeprefix("Yield_"): name for name in sources if name.startswith("Yield_")}
    if overrides:
        scene["feature_overrides"] = {"Yield": overrides}
    if "PH_PCT95_unmasked" in sources:
        scene["ph_feature"] = "PH_PCT95_unmasked"
        scene["ph_definition"] = "Supplied unmasked CHM PCT95 in metres, aligned by area average."
    _save_json(out / "scene.json", scene)
    report = {
        "schema_version": 1, "operation": "pack_precomputed_rasters",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_contract": contract, "training_performed": False,
        "feature_formulas_recomputed": False, "osavi_output_mask": False,
        "input_config_sha256": _hash(config_file),
        "reference_sha256": file_hash(reference), "reference_band": ref_band,
        "reference_auxiliary_file_hashes": [
            {"suffix": item.suffix, "sha256": file_hash(item)} for item in reference_auxiliary
        ],
        "boundary_sha256": boundary_records, "feature_inputs": feature_records,
        "shape": [height, width], "block_pixels": block, "resampling": "average",
        "source_coverage_definition": "fraction of finite, unmasked reference cells; map eligibility requires > 0",
        "source_coverage_cells": int((coverage > 0).sum()),
        "source_coverage_cells_inside_boundary": int((inside & (coverage > 0)).sum()),
        "output_sha256": {"features.npz": _hash(out / "features.npz"), "scene.json": _hash(out / "scene.json")},
        "outputs": {"scene": "scene.json", "features": "features.npz", "provenance": "input_provenance.json"},
    }
    _save_json(out / "input_provenance.json", report)
    return report
