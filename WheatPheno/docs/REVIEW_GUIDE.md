# Reviewer and release guide

Record the exact release archive or revision, environment, commands and findings. WheatPheno separates executable algorithm checks, frozen-model compatibility and historical evaluation evidence.

## 1. Installation and fixed predictions

From a clean Python 3.12 environment at the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m wheatpheno verify
python -m wheatpheno demo --output outputs/demo
python -m wheatpheno review --output outputs/review
```

NDRE is calculated as `(NIR - RE) / (NIR + RE)` throughout training, extraction and frozen-model inference; the ordered inputs are defined in `models/schema.json`.

## 2. Feature extraction and evaluation

```bash
python -m wheatpheno extract --input examples/synthetic/feature_input.npz --output outputs/features --profile standard --block-size 10
python -m wheatpheno evaluate --config configs/training.example.json --output outputs/training
```

Inspect formula tests, quantization settings, missing-data behavior and native height percentiles. Confirm that standard and frozen feature contracts cannot be silently exchanged. The supplied fixtures are synthetic execution checks, not independent field validation.

For evaluation, inspect outer and inner plot memberships, training-fold preprocessing, hyperparameter candidates, selected predictors and pooled repetition metrics. Confirm that ablations share splits. Optional new refits must remain separate from bundled deployment artifacts and independent validation scores.

## 3. Scientific scope

| Check | Evidence |
| --- | --- |
| Standard NDRE | `(NIR - RE) / (NIR + RE)` in training and frozen-model inference |
| Exact texture definitions | Standard and historical equations, quantization and formula tests |
| Training and mapping supports are explicit | Methods, raster guide and input configuration |
| Examples are not external validation | Data card and example manifest |
| New model development is reproducible | Training source, split records, search outputs and configuration |

The complete private training population is not distributed. Public code permits new experiments on supplied data; exact regeneration of all historical results additionally requires the original locked inputs. Successful tests do not establish pixel accuracy or transfer to a new trial.

## 4. Spatial application and publication assets

Use verified, user-owned rasters following `docs/RASTER_INPUT.md`. Inspect projected coordinates, true source coverage, boundary alignment, feature profiles, missingness and out-of-range counts. Confirm that no OSAVI threshold controls the final footprint and that display interpolation remains distinguishable from predictions.

Inspect the actual release archive and manifest. It should contain source code, fixed models, limited measured examples, synthetic fixtures, aggregate benchmarks and licenses. It must exclude real field imagery, complete private tables, personal identifiers, credentials, private workstation paths and generated outputs. This release contains no real field raster examples.

## 5. Review record

Retain test results and adverse findings, state which checks were actually executed, and distinguish local checks from remote CI and independent review. Code uses MIT; model and data assets follow the separate CC BY 4.0 scope. Record the exact artifact hashes and release decision without implying that software execution certifies scientific validity.
