# Deploying trained models

## Train and export

Set `refit: true` in an evaluation configuration, then run:

```bash
python -m wheatpheno evaluate --config configs/training.refit.example.json --output outputs/training
```

Each algorithm exports a joblib and matching JSON in `new_refits/`. The JSON records the target, feature order, selected predictors, feature contract, units, training ranges, scikit-learn version and SHA256. Stored imputation, selection and scaling are applied during prediction without fitting again. Keep the two files together.

The default contract is `wheatpheno-standard-v1`. A table using the frozen extraction profile must explicitly set `feature_contract: "wheatpheno-frozen-ndre-v2"`. A feature contract specifies extraction equations; use the same units, masks, spatial support and texture settings when preparing deployment data.

## Table predictions

```bash
python -m wheatpheno predict --input my_features.csv --target AGB --model outputs/training/new_refits/AGB_ExtraTrees_new_refit.joblib --trust-model --output outputs/predictions
```

Replace AGB with Yield for a yield model. The target and ordered input names must match the metadata. Every row needs at least 50% finite candidate features and 50% finite selected features; remaining missing values use the fitted imputer.

## Spatial predictions

Prepare a raster configuration with `feature_contract` matching the model. Its `feature_rasters` object contains every name in the exported metadata, including predictors subsequently removed by the fitted selector. A model trained on a subset only needs that subset, not all 72 candidates.

```bash
python -m wheatpheno pack --config my_rasters.json --model-metadata outputs/training/new_refits/AGB_ExtraTrees_new_refit.json --output outputs/scene
python -m wheatpheno map --config outputs/scene/scene.json --model outputs/training/new_refits/AGB_ExtraTrees_new_refit.joblib --trust-model --output outputs/maps
```

Without `--targets`, the model's target is selected automatically. Each call maps one local model. The supplied feature rasters are aligned to the reference grid; no formulas are recalculated during packing. Precompute height percentiles from native CHM samples at the intended support instead of averaging fine-grid percentiles.

The standard `extract` command also produces a feature archive for standard-profile models. To map that archive directly, supply a scene JSON with its affine transform, projected metre CRS, boundary and matching feature contract. The transform must describe the aggregated grid, not the original native grid.

Mapping writes numerical GeoTIFFs, clean PNGs, annotated previews and input-quality rasters. The boundary and source coverage determine the domain; OSAVI does not mask the output. `--fill-distance` enables a separate nearest-filled display product in metres. Raw predictions stay unchanged.

## Model loading

Joblib files can execute Python when loaded. `--trust-model` explicitly selects a trusted local artifact. The loader checks its hash, scikit-learn version, target, input order and selected predictors. Bundled models continue to use pinned hashes and their original fixed loader. New model loading is a CLI/API feature; the optional web interface retains the bundled-model workflow.
