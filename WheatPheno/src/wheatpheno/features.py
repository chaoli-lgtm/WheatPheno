"""Array-based, versioned UAV feature extraction.

The standard spectral profile always uses (NIR - RE) / (NIR + RE) for NDRE.
These functions do not imply compatibility with previously fitted estimators.
"""
from __future__ import annotations

from collections.abc import Mapping
import warnings

import numpy as np
from scipy.ndimage import uniform_filter

BANDS = ("B", "G", "NIR", "RE", "R")
TEXTURES = ("asm", "contrast", "correlation", "dissimilarity", "energy",
            "entropy", "homogeneity", "mean", "std")
EXTRACTION_CONTRACT = "wheatpheno-standard-v1"
FROZEN_CONTRACT = "wheatpheno-frozen-nir2-v1"
TEXTURE_PROFILES = ("standard", "historical")


def _array(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    return np.where(np.isfinite(array), array, np.nan)


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result,
              where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0))
    return result


def spectral_features(bands: Mapping[str, np.ndarray], *, include_nir2=False,
                      tcari_profile="standard") -> dict[str, np.ndarray]:
    """Calculate 15 indices from reflectance fractions, without vegetation masking.

    NDRE uses the conventional normalized difference. TCARI uses
    3 * ((RE - R) - 0.2 * (RE - G) * (RE / R)). Invalid observations and
    zero denominators remain NaN. Reflectance outside [0, 1] is rejected.
    NIR2 can be included as a separately named compatibility feature. The
    historical TCARI equation is available only through an explicit profile.
    """
    if tcari_profile not in ("standard", "historical"):
        raise ValueError("tcari_profile must be standard or historical.")
    if set(bands) != set(BANDS):
        raise ValueError(f"Supply exactly these band names: {', '.join(BANDS)}.")
    data = {name: _array(bands[name], name) for name in BANDS}
    if len({array.shape for array in data.values()}) != 1:
        raise ValueError("All band arrays must have the same shape and spatial alignment.")
    for name, array in data.items():
        if np.any(np.isfinite(array) & ((array < 0) | (array > 1))):
            raise ValueError(f"{name} requires reflectance fractions in [0, 1].")
    b, g, n, re, r = (data[name] for name in BANDS)
    result = {
        "EXR": 1.4 * r - g,
        "SDRB": r - b,
        "RBRI": _divide(r, b),
        "IKAW": _divide(r - b, r + b),
        "CI": _divide(r - b, r),
        "NDVI": _divide(n - r, n + r),
        "RVI": _divide(n, r),
        "CIgreen": _divide(n, g) - 1,
        "EVI": _divide(2.5 * (n - r), n + 6 * r - 7.5 * b + 1),
        "OSAVI": _divide(1.16 * (n - r), n + r + 0.16),
        "CIre": _divide(n, re) - 1,
        "NDRE": _divide(n - re, n + re),
        "SARE": _divide(1.25 * (n - re), n + re + 0.25),
        "TCARI": 3 * ((re - r) - 0.2 * (re - g) * _divide(re, r)),
        "MEVI": _divide(2.5 * (n - re), n + 6 * re - 7.5 * g + 1),
    }
    if include_nir2:
        result["NIR2"] = 2 * n
    if tcari_profile == "historical":
        result["TCARI"] = 3 * (re - r) - 0.2 * (re - g) * _divide(n, r)
    return result


def quantize(image, *, levels=8, policy="fixed", bounds=(0.0, 1.0),
             historical=False):
    """Return integer gray levels and a validity mask; NoData is never a gray level.

    Fixed quantization rejects values outside explicit bounds. Image min/max
    quantization uses finite observations only and must be selected explicitly.
    Historical quantization first rounds to 8 bit values before binning.
    """
    original = np.asarray(image)
    data = _array(image, "image")
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or not 2 <= levels <= 32:
        raise ValueError("levels must be an integer between 2 and 32.")
    if policy not in ("fixed", "image_minmax"):
        raise ValueError("policy must be fixed or image_minmax.")
    valid = np.isfinite(data)
    q = np.full(data.shape, -1, dtype=np.int16)
    if policy == "fixed":
        lo, hi = map(float, bounds)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("Quantization bounds must be finite and increasing.")
        if np.any(valid & ((data < lo) | (data > hi))):
            raise ValueError("Values outside quantization bounds must be resolved upstream.")
    elif np.any(valid):
        lo, hi = float(data[valid].min()), float(data[valid].max())
    else:
        return q, valid
    if not np.any(valid):
        return q, valid
    if historical and original.dtype == np.uint8 and policy == "image_minmax":
        scaled = data[valid]
    else:
        scaled = np.zeros(valid.sum()) if hi == lo else (data[valid] - lo) / (hi - lo)
        if historical:
            scaled = np.rint(scaled * 255)
    q[valid] = np.minimum(levels - 1, np.floor(scaled * (levels / 256 if historical else levels))).astype(np.int16)
    return q, valid


def texture_features(image, *, profile="standard", levels=8, window=5,
                     offset=(0, 1), quantization=None, bounds=(0.0, 1.0),
                     minimum_valid_fraction=0.5):
    """Calculate nine local GLCM descriptors with explicit numerical semantics.

    Standard uses pairs whose two endpoints lie inside the local window,
    directional probabilities, marginal mean/standard deviation and Pearson
    correlation. Historical reproduces the original
    mean/std/correlation equations, final-column replication and reflect101
    window borders. Both profiles exclude missing pairs and retain missing
    center pixels. Neither applies an OSAVI threshold.
    """
    if profile not in TEXTURE_PROFILES:
        raise ValueError("profile must be standard or historical.")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer.")
    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must lie in (0, 1].")
    if len(offset) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in offset) or tuple(offset) == (0, 0):
        raise ValueError("offset must contain two integer offsets and cannot be (0, 0).")
    if profile == "historical" and tuple(offset) != (0, 1):
        raise ValueError("The historical profile requires the original horizontal offset (0, 1).")
    if profile == "standard" and any(abs(value) >= window for value in offset):
        raise ValueError("Standard offsets must fit both pair endpoints inside the local window.")
    policy = quantization or ("image_minmax" if profile == "historical" else "fixed")
    q, valid = quantize(image, levels=levels, policy=policy, bounds=bounds,
                        historical=profile == "historical")
    h, w = q.shape
    rr, cc = np.indices(q.shape)
    rr, cc = rr + offset[0], cc + offset[1]
    in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    target = q[np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)]
    possible = np.ones(q.shape, bool) if profile == "historical" else in_bounds
    paired = valid & (target >= 0) & possible
    if profile == "historical":
        # Historical filter2D averages all pair origins, including neighbors
        # beyond the window itself; keep that numerical contract unchanged.
        smooth = lambda x: uniform_filter(x.astype(float), size=window, mode="mirror")
    else:
        radius = window // 2
        low = [max(-radius, -radius-delta) for delta in offset]
        high = [min(radius, radius-delta) for delta in offset]

        def smooth(x):
            # Sum only origins whose offset endpoints also fall in this patch.
            # An integral image supports asymmetric windows and either offset
            # sign without introducing reflected or replicated observations.
            padded = np.pad(x.astype(float), radius)
            integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            y0, x0 = radius + low[0], radius + low[1]
            y1, x1 = radius + high[0] + 1, radius + high[1] + 1
            return (integral[y1:y1+h, x1:x1+w] - integral[y0:y0+h, x1:x1+w]
                    - integral[y1:y1+h, x0:x0+w] + integral[y0:y0+h, x0:x0+w])
    denominator = smooth(paired)
    available = smooth(possible)
    eligible = valid & (denominator > 0) & (denominator >= minimum_valid_fraction * available)
    fields = {name: np.zeros(q.shape, float) for name in
              ("asm", "contrast", "dissimilarity", "entropy", "homogeneity", "mu_x", "mu_y", "xx", "yy", "xy", "p2x2")}
    for i in range(levels):
        for j in range(levels):
            probability = np.divide(smooth(paired & (q == i) & (target == j)), denominator,
                                    out=np.zeros(q.shape), where=denominator > 0)
            fields["asm"] += probability ** 2
            fields["contrast"] += probability * (i - j) ** 2
            fields["dissimilarity"] += probability * abs(i - j)
            fields["homogeneity"] += probability / (1 + (i - j) ** 2)
            fields["entropy"] -= probability * np.log(np.maximum(probability, 1e-300))
            fields["mu_x"] += probability * i
            fields["mu_y"] += probability * j
            fields["xx"] += probability * i * i
            fields["yy"] += probability * j * j
            fields["xy"] += probability * i * j
            fields["p2x2"] += probability ** 2 * i * i
    if profile == "historical":
        mean = fields["mu_x"] / levels ** 2
        variance = fields["p2x2"] - 2 * mean * fields["mu_x"] + levels ** 2 * mean ** 2
        std = np.sqrt(np.maximum(variance, 0))
        correlation = (fields["xy"] - mean * (fields["mu_x"] + fields["mu_y"]) + mean ** 2) / (std ** 2 + 1e-10)
    else:
        mean = fields["mu_x"]
        std = np.sqrt(np.maximum(fields["xx"] - mean ** 2, 0))
        std_y = np.sqrt(np.maximum(fields["yy"] - fields["mu_y"] ** 2, 0))
        correlation = np.divide(fields["xy"] - mean * fields["mu_y"], std * std_y,
                                out=np.ones(q.shape), where=std * std_y > 1e-14)
        correlation = np.clip(correlation, -1, 1)
    result = {name: fields[name] for name in ("asm", "contrast", "dissimilarity", "entropy", "homogeneity")}
    result.update(mean=mean, std=std, correlation=correlation, energy=np.sqrt(fields["asm"]))
    return {name: np.where(eligible, result[name], np.nan) for name in TEXTURES}


def _blocks(array, block_size):
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    h, w = array.shape
    rows, cols = (h + block_size - 1) // block_size, (w + block_size - 1) // block_size
    padded = np.pad(array, ((0, rows * block_size - h), (0, cols * block_size - w)), constant_values=np.nan)
    return padded.reshape(rows, block_size, cols, block_size).transpose(0, 2, 1, 3).reshape(rows, cols, -1)


def block_mean(array, block_size=10):
    """Aggregate finite values; missing or padded samples do not become zeros."""
    cells = _blocks(_array(array, "array"), block_size)
    return np.divide(np.nansum(cells, axis=-1), np.isfinite(cells).sum(axis=-1),
                     out=np.full(cells.shape[:2], np.nan), where=np.isfinite(cells).sum(axis=-1) > 0)


def structural_features(chm, vegetation_mask, *, block_size=10, height_mask=None):
    """Calculate CC and 11 height statistics on native CHM samples in metres.

    The caller supplies a 0/1 vegetation mask (NaN means unknown). CC is the
    vegetation fraction among finite CHM samples with known mask status.
    Height statistics use all finite CHM values unless height_mask is explicit.
    No vegetation criterion is applied implicitly to height or output coverage.
    """
    height = _array(chm, "chm")
    mask = _array(vegetation_mask, "vegetation_mask")
    if mask.shape != height.shape or np.any(np.isfinite(mask) & ~np.isin(mask, [0, 1])):
        raise ValueError("vegetation_mask must be aligned and contain only 0, 1 or NaN.")
    cc_values = np.where(np.isfinite(height) & np.isfinite(mask), mask, np.nan)
    if height_mask is not None:
        selected = np.asarray(height_mask)
        if selected.shape != height.shape or selected.dtype != bool:
            raise ValueError("height_mask must be an aligned boolean array.")
        height = np.where(selected, height, np.nan)
    cells = _blocks(height, block_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        percentiles = np.nanpercentile(cells, [50, 70, 90, 95, 99], axis=-1, method="linear")
        stats = {"MIN": np.nanmin(cells, axis=-1), "MAX": np.nanmax(cells, axis=-1),
                 "MEAN": np.nanmean(cells, axis=-1), "STD": np.nanstd(cells, axis=-1, ddof=0)}
    stats["RANGE"] = stats["MAX"] - stats["MIN"]
    stats["CV"] = _divide(stats["STD"], stats["MEAN"])
    result = {"CC": block_mean(cc_values, block_size)}
    result.update({f"CHM_PCT{p}": value for p, value in zip([50, 70, 90, 95, 99], percentiles)})
    result.update({f"CHM_{name}": value for name, value in stats.items()})
    return result


def extract_features(bands, chm, vegetation_mask, *, block_size=10, height_mask=None,
                     texture_profile="standard", quantization=None, texture_window=5,
                     include_nir2=False, tcari_profile="standard"):
    """Extract all 72 descriptors from aligned native arrays to a block grid.

    Indices are calculated before spatial averaging. Texture descriptors are
    calculated on native bands before averaging. Percentiles are calculated
    from native CHM samples, never by averaging precomputed percentiles.
    The returned fields use correct NDRE and require a matching model profile.
    """
    indices = spectral_features(bands, include_nir2=include_nir2, tcari_profile=tcari_profile)
    shape = next(iter(indices.values())).shape
    if _array(chm, "chm").shape != shape:
        raise ValueError("Band arrays and CHM must be aligned with the same shape.")
    result = {name: block_mean(value, block_size) for name, value in indices.items()}
    result.update(structural_features(chm, vegetation_mask, block_size=block_size, height_mask=height_mask))
    for band in BANDS:
        descriptors = texture_features(bands[band], profile=texture_profile,
                                       quantization=quantization, window=texture_window)
        result.update({f"Texture_{band}_{name}": block_mean(value, block_size) for name, value in descriptors.items()})
    return result
