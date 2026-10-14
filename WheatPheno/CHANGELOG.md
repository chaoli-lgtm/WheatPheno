# Changelog

## 0.4.0 NDRE correction - 2026-10-14

- Recomputed standard NDRE from source NIR and red-edge reflectances.
- Refit both bundled Extra Trees pipelines and updated input contracts, example values and reference predictions.
- Repeated the original grouped evaluation protocol with unchanged observations and targets.

## 0.4.0 - 2026-10-14

- Connected exported RF, SVR, PLSR and Extra Trees refits to table prediction and spatial mapping.
- Added metadata-driven raster packing for model-specific feature subsets.
- Added independent DSM/DTM plant-height estimation and seedling terrain generation using ground masks or OSAVI ground selection.
- Added nearest-neighbor and inverse-distance terrain interpolation, configurable CHM percentiles and height quality outputs.
- Removed the 72-feature requirement for PH-only scene mapping.
- Preserved the bundled model bytes and existing reference predictions.

## 0.3.0 - 2026-10-14

- Adopted the WheatPheno package name and English command-line documentation.
- Packaged grouped validation, feature ablation, fixed inference and polygon-based mapping.
