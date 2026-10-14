# Methods

## Feature extraction

The array-based extractor calculates 15 spectral, 12 structural and 45 texture descriptors from five aligned multispectral bands, CHM samples and an explicit vegetation mask. Spectral and texture descriptors are calculated on native arrays before block aggregation; height percentiles are calculated directly from native CHM samples. Ground-reference construction, radiometric calibration and image alignment precede the array interface.

NDRE is calculated as `(NIR - RE) / (NIR + RE)` throughout training, extraction and frozen-model inference; the ordered inputs are defined in `models/schema.json`.

Plant height is CHM PCT95 in metres multiplied by 100. The selected height mask and aggregation support form part of its definition. A percentile from an unmasked local window does not automatically inherit the validation of a vegetation-selected plot percentile.

## Model development and evaluation

The training module evaluates Random Forest, SVR, PLSR and Extra Trees. In each outer split, imputation, absolute Spearman ranking, predictor-count selection, standardization and hyperparameter selection occur within the training data through grouped inner validation. All observations from a physical plot remain together in both inner and outer splits. Inner selection minimizes RMSE; the selected pipeline is refitted on the outer training set before predicting held-out plots.

Metrics are calculated from pooled outer-fold predictions within each repetition and summarized across repetitions. Conditional feature contributions compare Extra Trees with all groups against models excluding one group, using matched splits and paired outer-fold errors. These contrasts are predictive rather than causal and are not additive percentages.

The executable synthetic example uses a reduced search for software checks. The study protocol, search ranges, seeds, bootstrap design and output files are documented in `docs/TRAINING.md`. New runs remain separate from archived historical scores, and exact historical reconstruction additionally requires the original locked observations and feature values.

## Fixed deployment inference

The bundled AGB and yield models are unchanged Extra Trees pipelines refitted on all retained observations for each target. They retain their fitted preprocessing and use the frozen input contract. Reference predictions verify the supplied inference pipeline.

New models export deployment metadata and use the local-model loader via `--model`. The input contract and feature order are checked before prediction. The bundled-model loader remains unchanged.

## Spatial mapping

Mapping consumes aligned, verified feature layers, source coverage and a polygon boundary. The raster packer aligns precomputed descriptors to a projected metre grid using average resampling; this alignment is not feature extraction and does not recalculate percentiles over native CHM samples.

Within the polygon and true source coverage, the mapper evaluates the fixed model where sufficient candidate and selected predictors are finite. It applies no OSAVI or canopy-cover eligibility threshold. OSAVI remains a numerical predictor, and a training feature that uses vegetation selection retains that specified definition. Original source gaps cannot be repaired by removing an output mask.

Optional nearest-neighbor filling creates a separate display product, leaving numerical predictions unchanged and identifying filled cells. Model imputation and display filling have different quality indicators. Negative CHM differences and out-of-training-range predictors remain diagnostic information rather than being silently clipped.

Plot-level validation does not establish local-window or pixel-level prediction accuracy. The original training population, measurement support determine the scope of spatial interpretation. No real field raster examples are included in this release.

## Run records

Configurations, input and artifact hashes, dependency versions, feature contracts, split memberships, model searches and numerical outputs accompany new runs. Release checks establish file integrity and executable behavior; they do not substitute for independent scientific validation.

## Independent terrain and height workflow

The height command aligns a supplied DTM to a target DSM, or interpolates a DTM from bare-ground samples in a seedling DSM. Ground samples come from a binary mask or a threshold applied to a seedling OSAVI raster. Nearest or inverse-distance interpolation operates within a specified distance of observed ground. CHM is DSM plus a supplied vertical offset minus DTM. PH is a configurable percentile of native CHM samples per output block, converted to centimetres. Ground selection does not mask the final height output.
