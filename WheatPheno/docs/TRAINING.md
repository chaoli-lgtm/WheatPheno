# Model development and reproducible evaluation

The `wheatpheno.training` module provides reusable training and evaluation code for
Random Forest, Support Vector Regression, Partial Least Squares Regression and
Extra Trees. It runs new experiments on a supplied feature table. The bundled
deployment weights and archived study scores remain unchanged.

## Run the executable example

From the repository root:

```python
from wheatpheno.training import evaluate_table

evaluate_table("configs/training.example.json", "outputs/training_example")
```

The example uses 48 explicitly synthetic observations from 12 synthetic plots,
four stages per plot and three synthetic trials, with a reduced `quick` search.
**This run checks the algorithm; its scores are not a reproduction of study
validation or an independent accuracy estimate.** No private field records are
used. Do not substitute row identifiers for plot groups when your table contains
repeated observations from the same plot.

The fixture can be regenerated with
`make_synthetic_table("new_synthetic.csv", seed=42)` from this module. Reflectance
and CHM samples are generated mathematically; the target is a synthetic growth
function with noise. Texture values come from generated 5 × 5 patches using
eight fixed reflectance bins, horizontal distance one and standard probability
matrix statistics. No model weights or real observations generate its targets.

## Supply a research table

A CSV contains one row per observation, an observed target, a globally unique
plot identifier, an observation identifier and a trial identifier. All stages
from one physical plot must share its plot identifier. Plot IDs cannot be reused
across trials. Every training
partition must contain enough distinct plots for the configured inner folds.

By default the module reads 72 standard features, including
`NDRE = (NIR - RE) / (NIR + RE)`. Supply `feature_names` to select
specific predictors. Feature-group ablation requires at least one predictor
from each group. Metadata are excluded from the predictor matrix. Nonfinite
predictors are imputed within training folds; missing targets, empty feature
rows or columns, and duplicate observation IDs are rejected.

For the study search design, adapt this configuration to your own local data:

```json
{
  "schema_version": 1,
  "input_csv": "my_features.csv",
  "target": "AGB",
  "target_column": "observed",
  "group_column": "plot",
  "id_column": "id",
  "trial_column": "trial",
  "protocol": "study",
  "models": ["RF", "SVR", "PLSR", "ExtraTrees"],
  "seeds": [42, 142, 242, 342, 442],
  "outer_folds": 5,
  "inner_folds": 5,
  "feature_group_ablation": true,
  "bootstrap_replicates": 1000,
  "bootstrap_seed": 20261010,
  "refit": false,
  "n_jobs": 1
}
```

Paths resolve relative to the configuration file. Output must be a new or empty
directory. The full `study` search performs many fits; `n_jobs` controls parallel
search jobs while each tree estimator uses one thread. A failed fit raises an
error rather than silently omitting a model or evaluation group.

## Protocol and provenance

The implementation ports the research workflow's absolute Spearman selector,
model families, parameter ranges, parameter sampling seeds and repeated grouped
split design. The original research scripts were `run_validation.py`,
`final_validation.py`, `optimize.py` and `run_contributions.py`. This portable
implementation does not import their private paths, source reconciliation or
intermediate checkpoints. It supports the UAV predictor configuration; the
historical trial/stage context comparison is not implemented here.

For every outer split, median imputation, absolute Spearman ranking, predictor
count selection and standardization are fitted inside each inner training fold.
The inner search minimizes RMSE, and its selected pipeline is refitted only on
the outer training observations before prediction on held-out plots. Plot
groups are disjoint in both outer and inner splits. Stable feature order breaks
ties in Spearman scores. Constant predictors receive a zero score.

| Model | Study search |
| --- | --- |
| RF | 150 trees; 10 sampled candidates; 8, 16 or all predictors; depth unlimited or 12; leaf size 1, 3 or 5; feature fraction 0.7 or 1.0 |
| SVR | RBF kernel with training-fold target scaling; 18 sampled candidates; 8, 16 or all predictors; C 0.1, 1, 10 or 100; epsilon 0.05 or 0.2; gamma scale, 0.01 or 0.1 |
| PLSR | 8, 16 or all predictors; exhaustive component counts from 1 to 12, bounded by retained predictors and inner training size; external input scaling |
| Extra Trees | 200 trees; 12 sampled candidates; 8, 16, 32 or all predictors; depth unlimited or 12; leaf size 1, 2 or 4; feature fraction 0.5, 0.8 or 1.0 |

Counts are capped at the number of available predictors. Search sampling uses
seed 42 for RF/SVR and 20261009 for Extra Trees unless `search_seed` overrides
them. PLSR uses its exhaustive grid. Estimator seeds equal the outer repetition
seed, and inner split seeds equal repetition seed plus outer fold number.
All candidate scores and selected parameters are
written to the output.

The `quick` preset defaults to one repetition and two outer/inner folds, uses
16 trees, one sampled candidate for RF/SVR/Extra Trees, and one PLS component.
It exists for execution checks and is not the publication search budget.

Two defensive differences from the original scripts are explicit: all-empty
predictors within a training fold retain their column positions and are filled
by `SimpleImputer(keep_empty_features=True)`, and PLS components are capped at
the smallest inner training sample count minus one. No claim of byte-identical
historical retraining is made. Exact study regeneration additionally requires
the original locked observations, historical feature values and source choices,
which are not all distributed.

## Metrics and feature contributions

R², RMSE, MAE, bias and NRMSE are calculated from all outer fold predictions
pooled **within each repetition**. Bias is prediction minus observation, and
NRMSE is RMSE divided by the observed range of the evaluated population. R² and
NRMSE are undefined for a constant target and are recorded as missing. Summaries
give arithmetic means and sample SD across repetitions; SD is missing for a
single repetition. Repetitions are not independent experimental replicates.

Conditional feature comparisons use Extra Trees with all features and with one
of the three feature groups removed. Every configuration uses the same outer
and inner splits, the same search budget and independent fitting within those
splits. A contribution is RMSE without the group minus RMSE with all groups;
it is neither a causal effect nor an additive percentage.

Optional confidence intervals resample whole plots, stratified by trial, from
the paired stored outer fold errors. Each draw uses the same sampled plots
across repetitions and feature configurations, then averages the repetition
contrasts. These are pointwise percentile intervals conditional on fitted
predictions, without multiplicity adjustment. They do not represent a bootstrap
of the complete training procedure. Set `bootstrap_replicates` to zero to omit
interval estimation.

## Files and optional new models

| Output | Contents |
| --- | --- |
| `predictions.csv` | Every outer-fold prediction, observation, group and split |
| `outer_splits.csv` | Outer test membership for every observation and repetition |
| `inner_validation_splits.csv` | Inner validation membership; complementary rows in that outer training set form inner training |
| `searches/*.json`, `fits.json` | Candidate scores, selected predictors, parameters, seeds and fold metrics |
| `repetition_metrics.csv`, `model_comparison.csv` | Pooled repetition metrics and their summaries |
| `feature_contributions_by_repeat.csv`, `feature_contributions.json` | Paired conditional contrasts and optional intervals |
| `configuration.json`, `report.json` | Resolved settings, hashes, versions and scope |
| `new_refits/` | Optional separately named full-data fitted models and metadata |

Setting `refit` to true selects each model using grouped inner validation on
all supplied rows, then saves a **new** full-data model. Its selection score is
not independent validation. These outputs use the trainable
`wheatpheno.training.SpearmanSelector`; the fixed deployment compatibility class
and bundled model files remain untouched. New models are ordinary local joblib
artifacts and must only be loaded from a trusted source. Use `predict --model` and `map --model` with `--trust-model` to deploy these artifacts. See `docs/DEPLOYMENT.md` for the complete workflow.

Output tables retain the supplied observation, plot and trial identifiers for
auditability. They remain local and must be reviewed before public sharing.
