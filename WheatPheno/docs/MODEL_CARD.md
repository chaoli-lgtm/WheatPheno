# Model card

## Estimators

| Target | Estimator | Inputs | Output unit |
| --- | --- | --- | --- |
| Plant height | CHM PCT95 multiplied by 100 | CHM percentile in metres | cm |
| Aboveground biomass | Extra Trees, 200 trees | 72 features, all retained | g m-2 |
| Grain yield | Extra Trees, 200 trees | 72 features, 16 selected internally | g m-2 |

The bundled regression pipelines are full-data refits. Stored imputation, feature selection and scaling are applied during inference. Exact artifact hashes, training counts and selected features are recorded in `models/manifest.json`.

## Input specification

Fixed inference uses `wheatpheno-frozen-nir2-v1`, with ordered feature names, units and equations in `models/schema.json`. NIR2 is the independent descriptor `2 * NIR`. NDRE is `(NIR - RE) / (NIR + RE)` in the standard extraction profile. Texture and TCARI profiles are defined in `docs/FEATURE_EXTRACTION.md`.

Each prediction requires at least 50% finite candidate inputs and at least 50% finite selected inputs. Remaining missing inputs use stored training medians. The loader checks pinned artifact hashes and the scikit-learn version.

## Development and evaluation

The training module supports RF, SVR, PLSR and Extra Trees with training-fold preprocessing, repeated nested cross-validation grouped by plot and feature-group ablation. New refits are stored separately and can be deployed using `--model` with matching metadata. See `docs/TRAINING.md`.

Published benchmark files summarize plot-level cross-validation by algorithm, repetition and subgroup. The 21 measured public examples overlap training and check reproducible execution. The complete training population is not distributed; the retained AGB training data exclude Changxing maturity observations from 2024.

## Independent plant height

The `height` command accepts DSM and DTM rasters, or generates a DTM from seedling ground samples before subtraction. It computes a configurable CHM percentile from native samples in each output block. No spectral or texture model inputs are required. See `docs/HEIGHT.md`.

## Spatial inference

Mapping applies plot-trained models to local feature summaries. Source coverage and valid inputs determine prediction eligibility within the supplied polygon; no OSAVI threshold controls display coverage. Gap filling produces a separate display raster with quality flags. Plot-level evaluation and spatial visualization are separate outputs.
