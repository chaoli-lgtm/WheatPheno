# WheatPheno

**UAV-based wheat phenotyping, model evaluation and spatial mapping**

Version **0.4.0** | Python **3.12** | Code **MIT** | Models and data **CC BY 4.0**

WheatPheno integrates canopy feature extraction, regression modelling and spatial inference for wheat plant height, aboveground biomass and grain yield. Its command-line workflow is supported by an optional local web interface.

## Capabilities

- Extract 15 spectral, 12 structural and 45 texture descriptors.
- Train and compare Random Forest, SVR, PLSR and Extra Trees using repeated nested cross-validation grouped by plot.
- Quantify conditional feature-group contributions using matched validation splits.
- Apply bundled models or deploy new RF, SVR, PLSR and Extra Trees refits.
- Derive CHM and plant height from DSM/DTM, including terrain generation from seedling ground samples.
- Record feature definitions, model hashes, validation splits and run settings.

## Installation

Use Python 3.12, then run from the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -e '.[test]'
python -m wheatpheno verify
python -m pytest -q
```

For the optional interface:

```bash
python -m pip install -e '.[ui]'
streamlit run app.py --server.address 127.0.0.1
```

## Quick start

Run the bundled prediction examples:

```bash
python -m wheatpheno demo --output outputs/demo
python -m wheatpheno predict --input examples/tables/agb_demo.csv --target AGB --output outputs/agb
```

Extract features and run model development with synthetic examples:

```bash
python -m wheatpheno extract --input examples/synthetic/feature_input.npz --output outputs/features --profile standard --block-size 10
python -m wheatpheno evaluate --config configs/training.example.json --output outputs/training
```

The training example contains 48 synthetic observations from 12 plots and uses the quick search preset. Use the study preset and your own feature table for research experiments.

Prepare and map your own raster inputs after configuring their paths:

```bash
python -m wheatpheno pack --config configs/rasters.example.json --output outputs/prepared
python -m wheatpheno map --config outputs/prepared/scene.json --output outputs/maps
```

Output directories must be new or empty. See [Raster inputs](docs/RASTER_INPUT.md) for the required feature layers and polygon geometry.

## Independent plant height

Supply preprocessed elevation rasters with metre elevation units and a common vertical datum:

```bash
python -m wheatpheno height --config configs/height.dtm.example.json --output outputs/height
```

For terrain generation from a seedling DSM and a ground mask, use `configs/height.seedling.example.json`. A seedling OSAVI raster can also identify ground samples. Ground extraction does not mask the final height map. See [Plant height](docs/HEIGHT.md).

## Deploy a newly trained model

Set `refit: true` in the training configuration. The exported joblib and matching JSON define the deployment inputs. Use only trusted local models:

```bash
python -m wheatpheno predict --input my_features.csv --target AGB --model outputs/training/new_refits/AGB_ExtraTrees_new_refit.joblib --trust-model --output outputs/new_predictions
python -m wheatpheno pack --config my_rasters.json --model-metadata outputs/training/new_refits/AGB_ExtraTrees_new_refit.json --output outputs/new_scene
python -m wheatpheno map --config outputs/new_scene/scene.json --model outputs/training/new_refits/AGB_ExtraTrees_new_refit.joblib --trust-model --output outputs/new_maps
```

See [Model deployment](docs/DEPLOYMENT.md). DSMs, reflectance products and aligned feature inputs are supplied by the user; photogrammetry and radiometric calibration are upstream processing.

## Features and fixed models

NDRE is calculated as `(NIR - RE) / (NIR + RE)`. Complete extraction equations and profiles are specified in [Feature extraction](docs/FEATURE_EXTRACTION.md). Fixed-model input names, order, units and equations are specified in [models/schema.json](models/schema.json).

| Target | Fixed estimator | Required input | Unit |
| --- | --- | --- | --- |
| Plant height | CHM PCT95 multiplied by 100 | Height percentile in metres | cm |
| Aboveground biomass | Extra Trees, 200 trees | 72 predictors, all retained | g m-2 |
| Grain yield | Extra Trees, 200 trees | 72 predictors, all retained | g m-2 |

The supplied models retain their fitted imputation, selection and scaling. [Model metadata](models/manifest.json) records artifact hashes and training details.

The map footprint follows the supplied polygon and source coverage, without an OSAVI display mask. Optional gap filling creates a separate display product and records interpolated cells in quality flags.

## Evaluation and public data

Five repetitions of nested cross-validation grouped by plot produced the following study results for Extra Trees with UAV predictors:

| Metric, mean +/- SD | AGB | Grain yield |
| --- | ---: | ---: |
| R2 | 0.835 +/- 0.003 | 0.865 +/- 0.011 |
| RMSE, g m-2 | 244.54 +/- 1.94 | 25.04 +/- 1.01 |

[Benchmark tables](benchmarks/README.md) provide algorithm comparisons, repetition metrics and subgroup summaries from the study's plot-level evaluation.

The release includes two fixed models, 12 measured biomass examples, nine measured yield examples and synthetic extraction/training fixtures. Measured examples overlap the model training data and serve as execution checks. Full training tables and real field rasters are not distributed. Spatial mapping applies the plot-trained models to local feature summaries; the reported validation scores concern plots.

## Documentation

- [Feature extraction](docs/FEATURE_EXTRACTION.md) and [training](docs/TRAINING.md)
- [Model card](docs/MODEL_CARD.md), [data card](docs/DATA_CARD.md) and [methods](docs/METHODS.md)
- [Raster inputs](docs/RASTER_INPUT.md) and [review guide](docs/REVIEW_GUIDE.md)
- [Local checks](docs/LOCAL_VALIDATION.md), [contributing](CONTRIBUTING.md) and [security](SECURITY.md)

```bash
python -m wheatpheno review --output outputs/review
```

## Citation and license

Citation metadata are provided in [CITATION.cff](CITATION.cff). Source code and documentation use the [MIT license](LICENSE); supplied models, example data and benchmark records use [CC BY 4.0](DATA_LICENSE.md).
