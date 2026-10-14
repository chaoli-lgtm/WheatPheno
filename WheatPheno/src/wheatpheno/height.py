"""Independent DSM/DTM to CHM and local-percentile plant-height workflow."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import warnings

import numpy as np
from scipy.spatial import cKDTree

from .pipeline import (_output, _read_boundary, _render, _report, _write_tif,
                       fill_for_display, save_json, sha256)
from .prepare import _metre_crs, _normalized_band


def interpolate_ground(seedling, soil, transform, *, method="nearest", radius=10.0,
                       neighbors=8, power=2.0):
    """Interpolate ground elevations in metres; retain observed ground values."""
    if method not in ("nearest", "idw"):
        raise ValueError("interpolation must be nearest or idw.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("ground_max_distance_m must be finite and positive.")
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or not 1 <= neighbors <= 32:
        raise ValueError("ground_neighbors must be an integer from 1 to 32.")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("idw_power must be finite and positive.")
    seedling, soil = np.asarray(seedling, float), np.asarray(soil, float)
    if seedling.ndim != 2 or seedling.shape != soil.shape:
        raise ValueError("Ground inputs must be aligned two-dimensional arrays.")
    if np.any(np.isfinite(soil) & ~np.isin(soil, [0, 1])):
        raise ValueError("ground_mask must contain 1 for ground, 0 for vegetation, or NoData.")
    valid = np.isfinite(seedling)
    observed = valid & (soil == 1)
    if observed.sum() < 3:
        raise ValueError("At least three finite ground samples are required.")
    rows, cols = np.nonzero(observed)
    xy = np.column_stack((rows * abs(transform.e), cols * transform.a))
    tree = cKDTree(xy)
    z = seedling[observed]
    dtm = np.where(observed, seedling, np.nan)
    distance = np.where(observed, 0.0, np.nan)
    query_rows, query_cols = np.nonzero(valid & ~observed)
    k = 1 if method == "nearest" else min(neighbors, len(z))
    for first in range(0, len(query_rows), 50000):
        rr, cc = query_rows[first:first+50000], query_cols[first:first+50000]
        query = np.column_stack((rr * abs(transform.e), cc * transform.a))
        dist, index = tree.query(query, k=k, distance_upper_bound=radius)
        dist, index = np.asarray(dist).reshape(len(rr), k), np.asarray(index).reshape(len(rr), k)
        nearest = dist[:, 0]
        values = z[np.minimum(index, len(z)-1)]
        weight = np.zeros(dist.shape)
        finite = np.isfinite(dist)
        weight[finite] = 1.0 / np.maximum(dist[finite], 1e-12)**power
        denom = weight.sum(axis=1)
        estimates = np.full(len(rr), np.nan)
        np.divide((weight * values).sum(axis=1), denom, out=estimates, where=denom > 0)
        dtm[rr, cc] = estimates
        distance[rr, cc] = np.where(np.isfinite(nearest), nearest, np.nan)
    return dtm, distance, observed


def _read_grid(path):
    import rasterio
    with rasterio.open(path) as src:
        _metre_crs(src.crs, "Elevation raster")
        tx = src.transform
        if tx.a <= 0 or tx.e >= 0 or tx.b != 0 or tx.d != 0:
            raise ValueError("Elevation rasters must use a north-up grid.")
        if src.width * src.height > 4_000_000:
            raise ValueError("Elevation raster exceeds four million cells; supply a cropped field or tile.")
        data = src.read(1, masked=True).astype(float).filled(np.nan)
        data[~np.isfinite(data)] = np.nan
        return data, tx, src.crs


def _align(path, shape, transform, crs, *, nearest=False):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    with TemporaryDirectory(prefix="wheatpheno_height_") as temp:
        clean_path = Path(temp) / "normalized.tif"
        with rasterio.open(path) as src:
            _metre_crs(src.crs, "Input raster")
            _normalized_band(src, 1, clean_path)
        with rasterio.open(clean_path) as src, WarpedVRT(
                src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                dtype="float32", src_nodata=np.nan, nodata=np.nan,
                resampling=Resampling.nearest if nearest else Resampling.bilinear) as vrt:
            return vrt.read(1, masked=True).filled(np.nan).astype(float)


def estimate_height(config_path, output_dir):
    """Use a supplied DTM or interpolate seedling ground; output CHM and PH."""
    from affine import Affine
    from rasterio.features import geometry_mask
    from .features import _blocks

    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"dsm", "dtm", "seedling_dsm", "ground_mask", "seedling_osavi",
               "soil_osavi_threshold", "interpolation", "ground_max_distance_m",
               "ground_neighbors", "idw_power", "boundary", "block_pixels", "percentile",
               "minimum_valid_fraction", "dsm_offset_m", "display_fill_distance_m",
               "display_limits", "elevation_units"}
    if set(cfg) - allowed:
        raise ValueError("Unknown height settings: " + ", ".join(sorted(set(cfg)-allowed)))
    if cfg.get("elevation_units") != "m":
        raise ValueError("Declare elevation_units=m; DSM and DTM need a common vertical datum.")
    def path(key):
        if not isinstance(cfg.get(key), str):
            raise ValueError(f"Supply a local {key} raster/path.")
        result = (config_path.parent / cfg[key]).resolve()
        if not result.is_file():
            raise FileNotFoundError(f"Missing {key}: {result.name}")
        return result

    provided = "dtm" in cfg
    if provided and any(k in cfg for k in ("seedling_dsm", "ground_mask", "seedling_osavi")):
        raise ValueError("Choose supplied DTM or seedling-ground generation, not both.")
    if not provided and ("seedling_dsm" not in cfg or ("ground_mask" in cfg) == ("seedling_osavi" in cfg)):
        raise ValueError("Seedling DTM generation requires seedling_dsm and exactly one ground_mask or seedling_osavi.")
    block = cfg.get("block_pixels", 10)
    if isinstance(block, bool) or not isinstance(block, int) or not 1 <= block <= 256:
        raise ValueError("block_pixels must be an integer from 1 to 256.")
    percentile = float(cfg.get("percentile", 95))
    minimum = float(cfg.get("minimum_valid_fraction", .5))
    offset = float(cfg.get("dsm_offset_m", 0))
    fill = float(cfg.get("display_fill_distance_m", 0))
    if not np.isfinite([percentile, minimum, offset, fill]).all() or not 0 <= percentile <= 100 or not 0 < minimum <= 1 or fill < 0:
        raise ValueError("Invalid percentile, valid fraction, elevation offset or display fill distance.")
    inputs = {"dsm": path("dsm")}
    dsm, transform, crs = _read_grid(inputs["dsm"])
    domain = np.ones(dsm.shape, bool)
    if "boundary" in cfg:
        inputs["boundary"] = path("boundary")
        domain = geometry_mask(_read_boundary(inputs["boundary"], crs), dsm.shape, transform, invert=True)
        if not domain.any():
            raise ValueError("Boundary does not overlap DSM.")
    seed_products = None
    if provided:
        inputs["dtm"] = path("dtm")
        dtm = _align(inputs["dtm"], dsm.shape, transform, crs)
        method = "supplied DTM"
    else:
        inputs["seedling_dsm"] = path("seedling_dsm")
        seedling, seed_tx, seed_crs = _read_grid(inputs["seedling_dsm"])
        if "ground_mask" in cfg:
            inputs["ground_mask"] = path("ground_mask")
            soil = _align(inputs["ground_mask"], seedling.shape, seed_tx, seed_crs, nearest=True)
        else:
            inputs["seedling_osavi"] = path("seedling_osavi")
            osavi = _align(inputs["seedling_osavi"], seedling.shape, seed_tx, seed_crs)
            threshold = float(cfg.get("soil_osavi_threshold", .15))
            if not np.isfinite(threshold):
                raise ValueError("soil_osavi_threshold must be finite.")
            soil = np.where(np.isfinite(osavi), (osavi < threshold).astype(float), np.nan)
        method = cfg.get("interpolation", "nearest")
        ground, distance, observed = interpolate_ground(
            seedling, soil, seed_tx, method=method,
            radius=cfg.get("ground_max_distance_m", 10.0),
            neighbors=cfg.get("ground_neighbors", 8), power=cfg.get("idw_power", 2.0))
        # Keep the terrain-generation support on the native seedling grid.
        seed_products = (ground, distance, observed, seed_tx, seed_crs)
        with TemporaryDirectory(prefix="wheatpheno_dtm_") as temp:
            generated = Path(temp)/"ground.tif"
            _write_tif(generated, ground, seed_tx, seed_crs)
            dtm = _align(generated, dsm.shape, transform, crs)

    chm = np.where(domain, dsm + offset - dtm, np.nan)
    if not np.isfinite(chm).any():
        raise ValueError("DSM and DTM have no finite overlapping canopy heights.")
    cells = _blocks(chm, block)
    valid_count = np.isfinite(cells).sum(axis=-1)
    support_count = np.nansum(_blocks(domain.astype(float), block), axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ph = np.nanpercentile(cells, percentile, axis=-1, method="linear") * 100
    ph[(support_count == 0) | (valid_count < np.ceil(minimum * support_count))] = np.nan
    tx = transform @ Affine.scale(block)
    coarse_domain = support_count > 0
    if "boundary" in inputs:
        coarse_domain &= geometry_mask(_read_boundary(inputs["boundary"], crs), ph.shape, tx, invert=True)
    ph[~coarse_domain] = np.nan
    if not np.isfinite(ph).any():
        raise ValueError("No height blocks meet the valid-coverage requirement.")
    source_count = np.isfinite(_blocks(np.where(domain, dsm, np.nan), block)).sum(axis=-1)
    display, filled = fill_for_display(ph, coarse_domain & (source_count > 0), tx, fill)
    quality = np.where(np.isfinite(ph), 1, 0).astype(np.uint8)
    quality[filled] = 2
    out = _output(output_dir)
    _write_tif(out/"DTM.tif", np.where(domain, dtm, np.nan), transform, crs)
    _write_tif(out/"CHM.tif", chm, transform, crs)
    _write_tif(out/"PH.tif", ph, tx, crs)
    _write_tif(out/"PH_quality.tif", quality, tx, crs, quality=True)
    if filled.any():
        _write_tif(out/"PH_display.tif", display, tx, crs)
    if seed_products:
        ground, distance, observed, seed_tx, seed_crs = seed_products
        _write_tif(out/"DTM_seedling.tif", ground, seed_tx, seed_crs)
        _write_tif(out/"ground_distance.tif", distance, seed_tx, seed_crs)
        _write_tif(out/"ground_samples.tif", observed.astype(np.uint8), seed_tx, seed_crs, quality=True)
    _render(display, out, "PH", cfg.get("display_limits"), (tx.a, abs(tx.e)))
    report = _report("height_from_elevation")
    hashes = {k: sha256(p) for k, p in inputs.items()}
    if "boundary" in inputs and inputs["boundary"].suffix.lower() == ".shp":
        for ext in (".prj", ".shx", ".dbf"):
            side = inputs["boundary"].with_suffix(ext)
            if side.exists():
                hashes["boundary"+ext] = sha256(side)
    report.update(configuration=cfg, input_sha256=hashes, terrain_method=method,
                  chm_formula="DSM + dsm_offset_m - DTM", chm_unit="m", ph_unit="cm",
                  percentile=percentile, block_pixels=block, osavi_output_mask=False,
                  predicted_cells=int(np.isfinite(ph).sum()), display_filled_cells=int(filled.sum()),
                  negative_chm_cells=int((np.isfinite(chm) & (chm < 0)).sum()),
                  quality_codes={"0": "No prediction", "1": "Calculated height", "2": "Display-only nearest fill"})
    save_json(out/"report.json", report)
    return report
