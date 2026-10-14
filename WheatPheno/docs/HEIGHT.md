# Plant height from elevation rasters

The independent `height` command requires no trained regression model or 72-feature stack. Inputs are preprocessed GeoTIFFs; each height input uses its first raster band. DSM and DTM elevations must be metres in the same vertical datum; horizontal coordinates must use projected metre CRSs. Raster alignment is handled by the command. Supply a cropped field or tile of at most four million cells for each DSM.

## Route 1: supplied DTM

```json
{
  "dsm": "inputs/maturity_dsm.tif",
  "dtm": "inputs/ground_dtm.tif",
  "elevation_units": "m",
  "boundary": "inputs/field.shp",
  "block_pixels": 10,
  "percentile": 95,
  "minimum_valid_fraction": 0.5,
  "dsm_offset_m": 0,
  "display_fill_distance_m": 0
}
```

```bash
python -m wheatpheno height --config configs/height.dtm.example.json --output outputs/height
```

DTM is bilinearly aligned to the DSM. CHM is `DSM + dsm_offset_m - DTM`; positive `dsm_offset_m` raises the DSM. Set an offset only from a measured vertical-alignment correction. Negative CHM values are retained and counted in the report.

## Route 2: generate DTM from seedling ground

```json
{
  "dsm": "inputs/maturity_dsm.tif",
  "seedling_dsm": "inputs/seedling_dsm.tif",
  "ground_mask": "inputs/seedling_ground_mask.tif",
  "elevation_units": "m",
  "boundary": "inputs/field.shp",
  "interpolation": "nearest",
  "ground_max_distance_m": 10,
  "block_pixels": 10,
  "percentile": 95
}
```

The ground mask uses 1 for bare ground, 0 for vegetation and NoData for unknown pixels; it is aligned with nearest-neighbor resampling. Seedling vegetation is excluded from terrain samples, and its locations receive interpolated ground elevations.

Alternatively replace `ground_mask` with `seedling_osavi` and optionally set `soil_osavi_threshold` (default 0.15). Finite pixels with OSAVI below that threshold supply ground elevations. This segmentation is used only to construct the DTM, not to exclude vegetation from CHM or PH outputs.

`interpolation` supports `nearest` and inverse-distance weighting (`idw`). IDW uses `ground_neighbors` (default 8, maximum 32) and `idw_power` (default 2). Only ground samples within `ground_max_distance_m` contribute; cells beyond that distance remain NoData. At least three finite ground samples are required. Observed ground elevations are preserved, and interpolation stays within the seedling DSM's valid coverage. Terrain is generated on the native seedling grid before alignment to the target DSM.

## Height aggregation and outputs

PH is the chosen percentile (0-100, default 95) of native CHM samples inside each nonoverlapping `block_pixels` block, multiplied by 100 to yield centimetres. For example, 10 pixels at 0.05 m resolution produce 0.5 m blocks. `block_pixels: 1` gives the individual CHM pixel height. Partial edge blocks use available in-domain native cells; blocks must satisfy `minimum_valid_fraction` (default 0.5). The optional boundary clips the output and excludes outside samples from percentiles.

- `DTM.tif`: terrain aligned to the target DSM, metres.
- `CHM.tif`: native-grid canopy height, metres.
- `PH.tif`: local percentile height, centimetres.
- `PH_clean.png` and `PH.png`: display products.
- `PH_quality.tif`: 0 missing/outside, 1 calculated, 2 display-only fill.
- `report.json`: parameters, input hashes, height counts and units.
- Generated terrain route: `DTM_seedling.tif`, `ground_samples.tif` and `ground_distance.tif` on the seedling grid; distances are metres to the nearest accepted ground sample.

Optional `display_fill_distance_m` creates `PH_display.tif` when gaps are filled. It does not modify CHM or the raw PH raster. The final map uses no OSAVI display mask. Photogrammetry, radiometric calibration and the acquisition of vertically consistent source products remain upstream tasks.
