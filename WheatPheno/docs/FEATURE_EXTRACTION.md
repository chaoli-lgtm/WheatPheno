# Reproducible feature extraction

The `wheatpheno.features` module computes spectral, structural and texture descriptors from aligned numerical arrays. It contains no file paths, private imagery, model fitting or output vegetation mask. Band calibration, photogrammetry, ground reference construction and geospatial alignment precede this interface.

## Spectral descriptors

Supply `B`, `G`, `R`, `RE` and `NIR` as reflectance fractions between zero and one. Invalid samples are `NaN`; zero is a valid reflectance unless it produces a zero denominator. The default interface returns 15 indices and always calculates:

`NDRE = (NIR - RE) / (NIR + RE)`

NDRE is calculated as `(NIR - RE) / (NIR + RE)` throughout training, extraction and frozen-model inference; the ordered inputs are defined in `models/schema.json`.

The default TCARI equation is `3 * ((RE - R) - 0.2 * (RE - G) * (RE / R))`. The explicit `tcari_profile="historical"` option computes the stored historical implementation `3 * (RE - R) - 0.2 * (RE - G) * (NIR / R)`. This option never changes the definition of NDRE. Formulas for every other spectral descriptor are given directly in `spectral_features`.

## Structural descriptors

`structural_features(chm, vegetation_mask, block_size=10)` accepts canopy heights in metres and a caller-supplied mask whose entries are one for vegetation, zero for nonvegetation and NaN for unknown status. It computes canopy cover and eleven height descriptors: PCT50, PCT70, PCT90, PCT95, PCT99, minimum, maximum, range, mean, population standard deviation and coefficient of variation.

Canopy cover is the fraction of vegetation among finite CHM samples with known vegetation status. Height statistics use all finite CHM samples by default. The optional boolean `height_mask` explicitly selects samples for height statistics when reproducing a specified training protocol. Neither interface derives this mask from OSAVI or uses it to decide whether a predicted map cell is displayed. Empty blocks remain NaN, padded edges do not contribute artificial zeros, and zero mean height produces an undefined coefficient of variation. Negative CHM differences are retained as input diagnostics.

Percentiles use NumPy's linear interpolation method on the native height samples. The module does not approximate a larger-support percentile by averaging smaller-support percentiles.

## Two explicit texture profiles

`texture_features` returns ASM, contrast, correlation, dissimilarity, energy, entropy, homogeneity, mean and standard deviation. Defaults are eight gray levels, a 5 × 5 window and a horizontal one-pixel offset, expressed as `(row_offset, column_offset) = (0, 1)`. The co-occurrence matrix is directional rather than symmetrized. In the standard profile, both endpoints of a pair must fall inside that centered window and the actual image. At image boundaries the window is intersected with the image extent, without reflection or replication. An offset whose magnitude along either axis is at least the window size is rejected because no pair can fit inside that window.

For the **standard** profile, fixed quantization bounds default to reflectance `[0, 1]`; values outside those bounds are rejected. Levels are `min(L - 1, floor(L * (x - low) / (high - low)))`. The explicitly selected `quantization="image_minmax"` alternative uses the finite image minimum and maximum. Changing bounds or image extent can therefore change a descriptor and must be recorded.

Let `p(i,j)` be the normalized local probability of a valid neighboring gray-level pair. Standard mean and standard deviation are the marginal moments of `i`, while correlation is the covariance of `i` and `j` divided by their marginal standard deviations. A zero marginal variance yields correlation one by convention. ASM is the sum of squared probabilities; energy is its square root; contrast weights probabilities by `(i-j)^2`; dissimilarity uses `abs(i-j)`; homogeneity uses `1 / (1 + (i-j)^2)`; entropy uses the natural logarithm.

The **historical** profile retains the original research implementation's descriptor equations for compatibility review:

```text
m = sum(p(i,j) * i) / L^2
s = sqrt(sum((p(i,j) * i - m)^2))
c = sum(p(i,j) * (i-m) * (j-m)) / (s^2 + 1e-10)
```

Its mean, standard deviation and correlation are those three quantities, not conventional GLCM moments. Historical quantization scales non-uint8 finite data to `[0, 255]`, rounds to integer values and bins them into `L` equal intervals over `[0, 256)`. Existing uint8 input is retained. The historical window selects pair origins rather than requiring both endpoints inside the window, so a horizontal one-pixel offset expands the effective native footprint of a 5 × 5 origin window to 5 × 6 pixels away from image boundaries. Its final column repeats the last horizontal neighbor, and local windows use reflect101 borders, matching the original finite-image implementation. The reimplementation uses double precision; historical float32 files can differ at floating-point rounding precision.

Both profiles omit pairs involving missing observations, normalize by the valid pair count and preserve missing center pixels. A local descriptor requires a positive valid-pair count and at least `minimum_valid_fraction=0.5` of geometrically available pairs. In the standard profile, the denominator counts pairs whose two endpoints both fit inside the window and image, regardless of their numerical validity. These explicit missing-data rules prevent NaN or NoData from becoming a dark gray level. 

## Complete extraction example

```python
import numpy as np
from wheatpheno.features import extract_features

rows, cols = np.indices((30, 40))
bands = {
    "B": 0.08 + 0.01 * cols / 39,
    "G": 0.14 + 0.02 * rows / 29,
    "R": 0.10 + 0.01 * cols / 39,
    "RE": 0.24 + 0.04 * rows / 29,
    "NIR": 0.45 + 0.08 * cols / 39,
}
chm = 0.4 + 0.6 * rows / 29
vegetation = np.ones(chm.shape, dtype=bool)
features = extract_features(bands, chm, vegetation, block_size=10)
assert len(features) == 72
assert features["NDRE"].shape == (3, 4)
```

NDRE is calculated as `(NIR - RE) / (NIR + RE)` throughout training, extraction and frozen-model inference; the ordered inputs are defined in `models/schema.json`.

Use the extraction profile, units, masks and spatial support specified for the selected model.
