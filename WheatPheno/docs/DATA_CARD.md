# Data card

## Released assets

| Asset | Scope | Intended use |
| --- | --- | --- |
| Measured tabular examples | 12 AGB records and nine yield records | Fixed-inference execution and reference predictions |
| Synthetic extraction fixture | Generated multispectral and CHM arrays | Formula, aggregation and missing-data checks |
| Synthetic training fixture | 48 observations from 12 synthetic plots and three synthetic trials | Executable grouped evaluation example |
| Fixed model artifacts | Two full-data Extra Trees refits | Repeat deployment inference |
| Aggregate study records | Algorithm, repetition and subgroup summaries | Inspect historical evaluation evidence |

No real field raster crops, full UAV mosaics, complete measured training tables, original coordinates or original plot identifiers are distributed. Synthetic data are explicitly identified and are not derived from model predictions or presented as field observations.

## Measured example selection

The measured records were selected at fixed positions within each trial, without screening residuals. They overlap model training and cannot estimate generalization performance. Sample identifiers are anonymous, and the selection policy is recorded in `examples/manifest.json`.

## Feature meaning and units

Frozen inference requires the 72 names in `models/schema.json`; standard extraction equations are specified in `docs/FEATURE_EXTRACTION.md`.

Reflectance inputs are fractions. CHM descriptors are in metres; PH output is centimetres. AGB and grain yield are expressed in g m⁻². Missing samples remain distinguishable from legitimate zeros, and preprocessing is fitted only on training data during evaluation.

## Reproducibility and privacy

The released source implements feature extraction and model development on user-supplied inputs. Exact regeneration of the historical study scores additionally requires the original locked observations and feature values, which are not all released. The published benchmark scope comprises algorithm, repetition and subgroup summaries.

User-generated evaluation outputs retain supplied identifiers and remain local.

## License

Supplied model artifacts, measured examples, synthetic fixtures and aggregate numerical records follow `DATA_LICENSE.md`. Unreleased source experiments and third-party materials are outside that grant. The original software and prose documentation follow `LICENSE`.
