# Optional interface

WheatPheno is a Python and command line project. The optional Streamlit interface
supports fixed table prediction and model inspection; feature extraction,
training evaluation and spatial mapping are provided by the core code.

Install the UI dependencies with `python -m pip install -e ".[ui]"` and run
`streamlit run app.py --server.address 127.0.0.1` from the repository root.

## Workspaces

- **Overview** describes the code, fixed models and limited public inputs.
- **Table Prediction** runs AGB, yield or PH prediction from a CSV.
- **Spatial Mapping** provides command line instructions for users' own verified
  rasters, feature extraction and training evaluation. It does not load field
  scenes or display previously generated field maps.
- **Model Specification** lists the input columns, units and artifact provenance.
- **Quality Reports** shows table prediction reports from the current session.

The public observed examples comprise 12 AGB records and nine yield records.
They overlap model training and serve as execution fixtures, not independent
validation data. Synthetic inputs demonstrate algorithms. Field scene archives,
field maps and the complete measured dataset are not distributed.

## Input and result handling

CSV uploads are limited to 32 MB and processed in isolated system temporary
directories. The UI neither accepts model binaries nor executes shell commands.
Example files remain unchanged. A hosted deployment processes uploaded data on
its server; local use binds the interface to the local machine. Temporary files
remain until the host cleans them up.

Fixed model inputs follow `models/schema.json`. PH table input is `CHM_PCT95` in metres; output is in centimetres.

Versioned session migration clears earlier table results and integrity status.
Every app rerun also discards former map results and download archives, so an
already open session cannot continue presenting withdrawn field scenes.

## Core commands

```bash
python -m wheatpheno predict --input examples/tables/agb_demo.csv --target AGB --output outputs/agb
python -m wheatpheno extract --input examples/synthetic/feature_input.npz --output outputs/features --profile standard --block-size 10
python -m wheatpheno evaluate --config configs/training.example.json --output outputs/training
python -m wheatpheno pack --config configs/rasters.example.json --output outputs/prepared
python -m wheatpheno map --config outputs/prepared/scene.json --output outputs/mapped
```

The raster configuration requires users' actual verified input paths before
packing and mapping. Mapping applies no OSAVI output threshold; source coverage,
geometry and feature quality still determine eligibility. See
[the raster input guide](RASTER_INPUT.md) for the spatial input requirements.

## Smoke checks

1. Open all five workspaces without uploaded data.
2. Run AGB, yield and PH table examples and inspect their prediction reports.
3. Confirm the mapping page shows instructions without missing-scene warnings.
4. Seed an old session with map results and archive bytes, rerun the app and
   confirm that these state entries are removed.
5. Confirm the interface offers no field-map display or cached map download.

Model integrity status is produced by an executed core check. The presence of a
manifest alone does not establish successful verification, and example results
do not establish prediction accuracy on an unseen agricultural dataset.

## Version 0.4.0 command-line workflows

Independent DSM/DTM height mapping and deployment of newly trained models use the CLI/API. The web pages continue to use the bundled models. See `HEIGHT.md` and `DEPLOYMENT.md` for commands and input configurations.
