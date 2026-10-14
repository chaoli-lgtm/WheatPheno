# Release checks

WheatPheno 0.4.0 was checked on Windows with Python 3.12.14 on 2026-10-14.

- 158 automated tests passed, covering feature equations, plot-grouped splits, frozen predictions, raster masks, invalid inputs and release manifests.
- End-to-end tests cover all four exported regression algorithms, metadata-driven raster packing, PH-only archives, DSM/DTM subtraction, both seedling ground-selection routes and bounded terrain interpolation.
- All 21 measured examples reproduced their stored reference predictions.
- Both model artifact hashes remained unchanged.
- CLI demonstration, standard feature extraction and synthetic four-model evaluation completed successfully.
- All five interface pages opened successfully; AGB, yield and PH table predictions completed without application errors.

Synthetic spatial tests use generated grids. No real field raster examples are included. GitHub Actions configures Python 3.12 checks on Windows and Linux.
