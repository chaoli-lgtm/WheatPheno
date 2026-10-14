# Raster inputs and spatial mapping

The mapper uses verified feature rasters, a reference grid, real source coverage and a polygon boundary. The `pack` command aligns precomputed descriptors; the separate extraction module computes descriptors from aligned arrays. Neither command reconstructs photogrammetric surfaces or performs radiometric calibration.

## Feature contract

For bundled AGB/yield models, supply all 72 names from `models/schema.json` and declare `feature_contract: "wheatpheno-frozen-nir2-v1"`. The frozen contract uses the equations and input order recorded in `models/schema.json`.

Every feature must match its model's numerical scale, extraction support, mask definition and units. Optional `Yield_<feature name>` layers become explicit yield-specific overrides. An optional `PH_PCT95_unmasked` layer supplies height percentiles for PH separately from the structural inputs used by AGB and yield. Its percentile support and validation must be documented.

## Configuration

Edit `configs/rasters.example.json` to reference real, user-owned files. This abbreviated example illustrates the configuration structure; all remaining required names must also be supplied:

```json
{
  "scene_id": "my_field",
  "feature_contract": "wheatpheno-frozen-nir2-v1",
  "reference_raster": "inputs/unmasked_reference.tif",
  "reference_band": 1,
  "boundary": "inputs/field.shp",
  "block_pixels": 10,
  "minimum_valid_fraction": 0.5,
  "feature_rasters": {
    "NIR2": "inputs/NIR2.tif",
    "CHM_PCT95": {"path": "inputs/structural.tif", "band": 4},
    "PH_PCT95_unmasked": "inputs/ph_pct95.tif"
  }
}
```

A feature entry can be a path, using its first band, or an object containing `path` and a one-based `band`. Paths resolve relative to the configuration file; local absolute paths are also accepted. The output provenance records source hashes rather than private absolute paths. Trusted local refits use `--model` and `--trust-model`; remote URLs and executable expressions are not supported.

```bash
python -m wheatpheno pack --config configs/rasters.example.json --output outputs/prepared
python -m wheatpheno map --config outputs/prepared/scene.json --output outputs/maps
```

Both output directories must be new or empty.

## Coordinates and spatial support

Reference and feature rasters require explicit projected coordinate systems with horizontal units in metres. Geographic degrees, feet, missing CRS definitions and rotated reference grids are rejected. Feature rasters in different valid projected systems are reprojected to the reference grid.

A SHP boundary requires `.shp`, `.shx`, `.dbf` and `.prj` files. A GeoJSON must declare its projected CRS in `crs.properties.name`. The boundary must intersect the reference grid.

`block_pixels: 1` retains the reference grid; `block_pixels: 10` makes each output side span ten reference pixels. Partial edge cells can extend beyond the original extent. Average resampling respects source masks and NoData, but **averaging percentile rasters does not recalculate a percentile from native height samples**. For correct larger-support height percentiles, use native CHM extraction before packing.

## Coverage and missing data

Valid zeros are retained unless a source raster explicitly declares zero as NoData. Reference coverage comes from finite, unmasked values of the chosen reference band. The mapper requires positive coverage inside the boundary; a reference that was already vegetation-masked retains its source gaps. Use a genuinely unmasked reference when full available coverage is intended.

`minimum_valid_fraction` ranges from 0.5 to 1.0 and applies separately to candidate inputs and the target model's selected predictors. It determines whether stored preprocessing can impute remaining feature gaps; it never supplies information to cells without real source coverage. Manually assembled NPZ inputs must include an explicit `source_coverage` array.

**No OSAVI or canopy-cover threshold controls final output eligibility.** OSAVI remains a numerical feature, and extraction masks remain part of the input definition. Disabling an output threshold does not remove a predictor or change how training features were calculated.

## Outputs

Packing writes `features.npz`, `scene.json`, copied boundary files and `input_provenance.json`. Relative scene paths allow the prepared directory to be moved as a unit. Real projected coordinates remain intact; these application outputs are not anonymized public data.

Mapping writes numerical GeoTIFF predictions, quality indicators, out-of-range counts, previews and a run report. Optional `--fill-distance 0.3` permits a separate display fill within 0.3 metres while retaining original predictions. Source gaps, model imputation and display interpolation have distinct meanings and must remain identifiable.

The packer limits output to four million cells, expanded archives to 2 GB and `block_pixels` to integers from 1 to 256. Use tiles or a coarser support for larger datasets. Packing writes one normalized source band at a time; inference still requires memory for the scene arrays.

This release distributes no real field raster examples. Synthetic extraction fixtures test algorithms and must not be presented as field maps or accuracy evidence. User-owned mapping requires input-quality review and validation appropriate to its spatial support and target domain.

## Model-specific and height inputs

For a newly trained model, `pack --model-metadata MODEL.json` reads the exact required feature list and contract from its export. Then run `map --model MODEL.joblib --trust-model`. See [Deployment](DEPLOYMENT.md).

For PH-only archives, `map --targets PH` requires only the chosen height layer and `source_coverage`, plus the scene geometry and grid metadata. For DSM/DTM inputs or seedling terrain generation, use the independent [height command](HEIGHT.md), which requires no regression feature archive.
