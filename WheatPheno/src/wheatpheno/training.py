"""Reusable grouped model development, isolated from the frozen deployment models.

This module ports the study's training-fold preprocessing and model search design.
It accepts user-owned feature tables; no private data paths or hidden row filters
are embedded. Results are a new experiment, never a replacement for archived scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .schema import feature_names, feature_schema

MODELS = ("RF", "SVR", "PLSR", "ExtraTrees")


def standard_feature_names():
    """Ordered model inputs, including standard NDRE."""
    return feature_names()


def make_synthetic_table(output_csv, seed=42):
    """Create 48 deterministic, explicitly synthetic workflow-test observations.

    Reflectance and local CHM samples are generated mathematically. No field data
    or trained-model predictions enter this fixture. NDRE uses the conventional
    ratio; texture descriptors use an eight-level horizontal co-occurrence matrix.
    """
    output_csv = Path(output_csv)
    if output_csv.exists():
        raise FileExistsError("Synthetic table output already exists.")
    rng = np.random.default_rng(seed)
    records = []
    for plot in range(12):
        plot_offset = rng.normal(0, .025)
        for stage in range(1, 5):
            B = float(np.clip(.05 + rng.normal(0, .004), .01, .1))
            G = float(.10 + .006*stage + rng.normal(0, .005))
            R = float(.08 - .006*stage + rng.normal(0, .003))
            RE = float(.19 + .014*stage + rng.normal(0, .008))
            NIR = float(.35 + .05*stage + rng.normal(0, .012))
            values = {"EXR": 1.4*R-G, "SDRB": R-B, "RBRI": R/B, "IKAW": (R-B)/(R+B),
                      "CI": (R-B)/R, "NDVI": (NIR-R)/(NIR+R), "RVI": NIR/R,
                      "CIgreen": NIR/G-1, "EVI": 2.5*(NIR-R)/(NIR+6*R-7.5*B+1),
                      "OSAVI": 1.16*(NIR-R)/(NIR+R+.16), "CIre": NIR/RE-1,
                      "NDRE": (NIR-RE)/(NIR+RE), "SARE": 1.25*(NIR-RE)/(NIR+RE+.25),
                      "TCARI": 3*((RE-R)-.2*(RE-G)*(RE/R)),
                      "MEVI": 2.5*(NIR-RE)/(NIR+6*RE-7.5*G+1)}
            heights = np.maximum(0, rng.normal(.16*stage+plot_offset, .025+.005*stage, 100))
            values["CC"] = float(np.clip(.2+.17*stage+rng.normal(0,.025), 0, 1))
            for percentile in (50, 70, 90, 95, 99):
                values[f"CHM_PCT{percentile}"] = float(np.percentile(heights, percentile))
            values.update(CHM_MIN=float(heights.min()), CHM_MAX=float(heights.max()),
                          CHM_RANGE=float(np.ptp(heights)), CHM_MEAN=float(heights.mean()),
                          CHM_STD=float(heights.std()), CHM_CV=float(heights.std()/heights.mean()))
            for band, mean in (("B", B), ("G", G), ("NIR", NIR), ("RE", RE), ("R", R)):
                raw = np.clip(rng.normal(mean, .08, (5, 5)), 0, 1)
                q = np.minimum((raw*8).astype(int), 7)
                glcm = np.zeros((8, 8), dtype=float)
                np.add.at(glcm, (q[:, :-1].ravel(), q[:, 1:].ravel()), 1)
                glcm /= glcm.sum()
                i, j = np.indices(glcm.shape)
                mu_i, mu_j = float((i*glcm).sum()), float((j*glcm).sum())
                sd_i = float(np.sqrt(((i-mu_i)**2*glcm).sum()))
                sd_j = float(np.sqrt(((j-mu_j)**2*glcm).sum()))
                texture = {"asm": float((glcm**2).sum()), "contrast": float(((i-j)**2*glcm).sum()),
                           "correlation": float(((i-mu_i)*(j-mu_j)*glcm).sum()/(sd_i*sd_j)) if sd_i*sd_j else 1.,
                           "dissimilarity": float((np.abs(i-j)*glcm).sum()),
                           "energy": float(np.sqrt((glcm**2).sum())),
                           "entropy": float(-(glcm[glcm>0]*np.log(glcm[glcm>0])).sum()),
                           "homogeneity": float((glcm/(1+(i-j)**2)).sum()), "mean": mu_i, "std": sd_i}
                values.update({f"Texture_{band}_{name}": value for name, value in texture.items()})
            records.append({"sample_id": f"synthetic_{plot:02d}_{stage}", "plot": f"synthetic_plot_{plot:02d}",
                            "trial": f"synthetic_trial_{plot//4+1}", "stage": stage,
                            "observed": float(150+2100*values["CHM_MEAN"]+35*(plot//4)+rng.normal(0,35)),
                            **values})
    frame = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, float_format="%.10g")
    return frame


class SpearmanSelector(TransformerMixin, BaseEstimator):
    """Rank absolute Spearman associations using only the current fitting fold."""

    def __init__(self, k=8):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Selector fitting requires aligned finite X and y.")
        if not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError("k must be a positive integer.")
        rx = rankdata(X, axis=0)
        ry = rankdata(y)
        rx -= rx.mean(axis=0)
        ry -= ry.mean()
        denominator = np.sqrt((rx * rx).sum(axis=0) * (ry * ry).sum())
        self.scores_ = np.divide(np.abs(rx.T @ ry), denominator,
                                 out=np.zeros(X.shape[1]), where=denominator > 0)
        self.indices_ = np.argsort(-self.scores_, kind="stable")[:min(self.k, X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "indices_"):
            raise ValueError("The selector must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitted width.")
        return X[:, self.indices_]


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(path, value):
    Path(path).write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def regression_metrics(y, predicted):
    """Pooled metrics, with range-normalized RMSE and prediction-minus-observation bias."""
    y, predicted = np.asarray(y, float), np.asarray(predicted, float)
    if not len(y) or y.shape != predicted.shape or not np.isfinite(y).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require aligned, nonempty finite arrays.")
    error = predicted - y
    rmse = float(np.sqrt(np.mean(error ** 2)))
    span = float(np.ptp(y))
    return {"n": len(y), "R2": float(r2_score(y, predicted)) if len(y) > 1 and span > 0 else None,
            "RMSE": rmse, "MAE": float(np.mean(np.abs(error))), "Bias": float(np.mean(error)),
            "NRMSE": rmse / span if span > 0 else None}


def grouped_splits(groups, folds, seed):
    """Return disjoint plot-grouped splits without silently reducing fold count."""
    groups = np.asarray(groups)
    _positive_int(folds, "folds", 2)
    if len(np.unique(groups)) < folds:
        raise ValueError(f"Need at least {folds} distinct plots for grouped validation.")
    cv = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros((len(groups), 1)), groups=groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Plot leakage detected in a generated split.")
    return splits


def _search(model_name, p, seed, protocol, min_inner_train, search_seed):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}.")
    quick = protocol == "quick"
    ks = [min(8, p)] if quick else sorted({min(k, p) for k in ([8, 16, 32, 72] if model_name == "ExtraTrees" else [8, 16, p])})
    if model_name in ("RF", "ExtraTrees"):
        cls = RandomForestRegressor if model_name == "RF" else ExtraTreesRegressor
        model = cls(n_estimators=16 if quick else (150 if model_name == "RF" else 200), random_state=seed, n_jobs=1)
        space = {"select__k": ks, "model__max_depth": [None, 12],
                 "model__min_samples_leaf": [1, 3, 5] if model_name == "RF" else [1, 2, 4],
                 "model__max_features": [.7, 1.] if model_name == "RF" else [.5, .8, 1.]}
        count = 1 if quick else (10 if model_name == "RF" else 12)
        candidates = list(ParameterSampler(space, n_iter=count, random_state=search_seed))
    elif model_name == "SVR":
        model = TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler())
        space = {"select__k": ks, "model__regressor__C": [.1, 1, 10, 100],
                 "model__regressor__epsilon": [.05, .2], "model__regressor__gamma": ["scale", .01, .1]}
        candidates = list(ParameterSampler(space, n_iter=1 if quick else 18, random_state=search_seed))
    else:
        model = PLSRegression(scale=False, max_iter=2000)
        candidates = [{"select__k": k, "model__n_components": n} for k in ks
                      for n in range(1, min(1 if quick else 12, k, min_inner_train - 1) + 1)]
        if not candidates:
            raise ValueError("PLSR requires at least two training observations in every inner fold.")
    pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("select", SpearmanSelector()), ("scale", StandardScaler()), ("model", model)])
    return pipe, [{name: [value] for name, value in candidate.items()} for candidate in candidates]


def _feature_sets(names, ablation):
    sets = {"all": names}
    if ablation:
        schema = feature_schema()
        for group in ("spectral", "structural", "texture"):
            members = {item["name"] for item in schema["features"] if item["group"] == group}
            if "NIR2" in members or "NDRE" in members:
                members.update(("NIR2", "NDRE"))
            reduced = [name for name in names if name not in members]
            if len(reduced) == len(names) or not reduced:
                raise ValueError("Feature group ablation requires a nonempty input from all three feature groups.")
            sets["without_" + group] = reduced
    return sets


def _paired_contributions(oof, replicates, seed):
    """Paired plot bootstrap over stored OOF errors, not over model fitting."""
    et = oof[(oof["mode"] == "nested") & (oof["model"] == "ExtraTrees")]
    if et.empty or et["feature_set"].nunique() < 2:
        return [], []
    keys = ["repeat", "row_position", "plot", "trial"]
    full = et[et.feature_set == "all"][keys + ["observed", "predicted"]]
    delta_records, summary = [], []
    for group in ("spectral", "structural", "texture"):
        reduced = et[et.feature_set == "without_" + group][keys + ["predicted"]]
        paired = full.merge(reduced, on=keys, suffixes=("_full", "_without"), validate="one_to_one")
        if len(paired) != len(full):
            raise RuntimeError("Ablation predictions are not paired on identical held-out records.")
        paired["e_full"] = (paired.predicted_full - paired.observed) ** 2
        paired["e_without"] = (paired.predicted_without - paired.observed) ** 2
        deltas = []
        for repetition, part in paired.groupby("repeat", sort=True):
            value = float(np.sqrt(part.e_without.mean()) - np.sqrt(part.e_full.mean()))
            deltas.append(value)
            delta_records.append({"group": group, "repeat": int(repetition), "delta_RMSE": value})
        record = {"group": group, "mean_delta_RMSE": float(np.mean(deltas)), "bootstrap_replicates": replicates,
                  "interval_type": "Pointwise paired plot bootstrap of stored OOF errors; no multiplicity adjustment",
                  "lower_95": None, "upper_95": None}
        if replicates:
            # Sum within each plot and repetition; sampling a plot retains all of its observations.
            sums = paired.groupby(["repeat", "plot", "trial"], sort=True).agg(
                full=("e_full", "sum"), without=("e_without", "sum"), n=("e_full", "size")).reset_index()
            plot_trials = sums[["plot", "trial"]].drop_duplicates()
            if plot_trials["plot"].duplicated().any():
                raise ValueError("Plot identifiers must be unique across trials for bootstrap stratification.")
            plots = sorted(plot_trials["plot"].tolist())
            positions = {plot: i for i, plot in enumerate(plots)}
            strata = [[positions[p] for p in part["plot"]] for _, part in plot_trials.groupby("trial", dropna=False)]
            full_matrix = sums.pivot(index="repeat", columns="plot", values="full").reindex(columns=plots).to_numpy()
            reduced_matrix = sums.pivot(index="repeat", columns="plot", values="without").reindex(columns=plots).to_numpy()
            counts = sums.pivot(index="repeat", columns="plot", values="n").reindex(columns=plots).to_numpy()
            rng = np.random.default_rng(seed)
            draws = np.empty(replicates)
            for i in range(replicates):
                sample = np.concatenate([rng.choice(stratum, len(stratum), replace=True) for stratum in strata])
                denominator = counts[:, sample].sum(axis=1)
                draws[i] = np.mean(np.sqrt(reduced_matrix[:, sample].sum(axis=1) / denominator)
                                   - np.sqrt(full_matrix[:, sample].sum(axis=1) / denominator))
            record["lower_95"], record["upper_95"] = np.quantile(draws, [.025, .975]).tolist()
        summary.append(record)
    return delta_records, summary


def evaluate_table(config_path, output_dir) -> dict:
    """Run configured grouped evaluation and optional new refits, never altering bundled assets.

    Paths in the configuration resolve relative to its directory. The supplied
    feature table is already extracted: formulas and values are never rewritten.
    Output contains input row identifiers and should be reviewed before sharing.
    """
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "input_csv", "target", "target_column", "group_column", "id_column",
               "trial_column", "feature_names", "models", "protocol", "seeds", "outer_folds", "inner_folds",
               "feature_group_ablation", "bootstrap_replicates", "bootstrap_seed",
               "refit", "n_jobs", "search_seed", "feature_contract"}
    if set(cfg) - allowed:
        raise ValueError("Unknown training configuration keys: " + ", ".join(sorted(set(cfg) - allowed)))
    if cfg.get("schema_version", 1) != 1:
        raise ValueError("Training configuration schema_version must be 1.")
    protocol = cfg.get("protocol", "study")
    if protocol not in ("study", "quick"):
        raise ValueError("protocol must be 'study' or 'quick'.")
    target = cfg.get("target", "AGB")
    if target not in ("AGB", "Yield"):
        raise ValueError("target must be 'AGB' or 'Yield'.")
    names = cfg.get("feature_names", standard_feature_names())
    permitted = set(standard_feature_names()) | set(feature_names())
    if not names or len(names) != len(set(names)) or not set(names).issubset(permitted):
        raise ValueError("feature_names must contain distinct names from the released feature schema.")
    contract = cfg.get("feature_contract", "wheatpheno-standard-v1")
    if contract not in ("wheatpheno-standard-v1", "wheatpheno-frozen-ndre-v2"):
        raise ValueError("Unknown feature_contract.")
    if "NIR2" in names:
        raise ValueError("This protocol uses standard NDRE; twice-NIR inputs are unsupported.")
    models = cfg.get("models", list(MODELS))
    if not models or len(models) != len(set(models)) or not set(models).issubset(MODELS):
        raise ValueError("models must contain distinct supported model names.")
    quick = protocol == "quick"
    seeds = cfg.get("seeds", [42] if quick else [42, 142, 242, 342, 442])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of distinct integers.")
    for seed in seeds:
        _positive_int(seed, "seed", 0)
    outer_folds = _positive_int(cfg.get("outer_folds", 2 if quick else 5), "outer_folds", 2)
    inner_folds = _positive_int(cfg.get("inner_folds", 2 if quick else 5), "inner_folds", 2)
    n_jobs = _positive_int(cfg.get("n_jobs", 1), "n_jobs")
    bootstrap = _positive_int(cfg.get("bootstrap_replicates", 0 if quick else 1000), "bootstrap_replicates", 0)
    bootstrap_seed = _positive_int(cfg.get("bootstrap_seed", 20261010), "bootstrap_seed", 0)
    if "search_seed" in cfg:
        _positive_int(cfg["search_seed"], "search_seed", 0)
    for key in ("feature_group_ablation", "refit"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be true or false.")
    ablation = cfg.get("feature_group_ablation", True)
    if ablation and "ExtraTrees" not in models:
        raise ValueError("Feature group ablation requires ExtraTrees in models.")
    sets = _feature_sets(names, ablation)
    source = (config_path.parent / cfg["input_csv"]).resolve()
    table = pd.read_csv(source)
    target_col, group_col = cfg.get("target_column", "observed"), cfg.get("group_column", "plot")
    id_col, trial_col = cfg.get("id_column", "id"), cfg.get("trial_column", "trial")
    required = set(names + [target_col, group_col])
    if required - set(table.columns):
        raise ValueError("Missing table columns: " + ", ".join(sorted(required - set(table.columns))))
    if {target_col, group_col, id_col, trial_col} & set(names):
        raise ValueError("Target and metadata columns cannot also be predictors.")
    if table.empty or table[group_col].isna().any():
        raise ValueError("The table must contain rows with nonmissing plot identifiers.")
    y = pd.to_numeric(table[target_col], errors="raise").to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Observed targets must be finite; no target rows are silently removed.")
    X = table[names].apply(pd.to_numeric, errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
    if X.isna().all(axis=1).any() or X.isna().all(axis=0).any():
        raise ValueError("Wholly missing predictor rows or columns require upstream investigation.")
    groups = table[group_col].astype(str).to_numpy()
    ids = table[id_col].astype(str).to_numpy() if id_col in table else np.array([str(i) for i in range(len(table))])
    if len(set(ids)) != len(ids):
        raise ValueError("Observation identifiers must be unique; repeated plots use distinct row IDs.")
    trials = table[trial_col].astype(str).to_numpy() if trial_col in table else np.full(len(table), "unspecified")
    if trial_col in table and table[trial_col].isna().any():
        raise ValueError("Trial identifiers cannot be missing.")
    if pd.DataFrame({"plot": groups, "trial": trials}).groupby("plot")["trial"].nunique().max() > 1:
        raise ValueError("A plot identifier spans multiple trials; use globally unique plot IDs.")
    jobs = []
    for repetition, seed in enumerate(seeds, 1):
        for fold, (train, test) in enumerate(grouped_splits(groups, outer_folds, seed), 1):
            jobs.append(("nested", repetition, fold, seed, train, test))
    # Fail before creating output if any planned inner split is impossible.
    for _, _, fold, seed, train, test in jobs:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Outer train and test plots overlap.")
        grouped_splits(groups[train], inner_folds, seed + fold)
    out = Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Training output must be a new or empty directory.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "searches").mkdir()
    resolved = {**cfg, "input_csv": source.name, "target": target, "protocol": protocol, "feature_names": names,
                "models": models, "seeds": seeds, "outer_folds": outer_folds, "inner_folds": inner_folds,
                "feature_group_ablation": ablation, "bootstrap_replicates": bootstrap,
                "bootstrap_seed": bootstrap_seed, "n_jobs": n_jobs, "feature_contract": contract}
    _save(out / "configuration.json", resolved)
    predictions, splits, inner_records, fit_records = [], [], [], []
    for mode, repetition, fold, seed, train, test in jobs:
        inner = grouped_splits(groups[train], inner_folds, seed + fold)
        key_base = {"mode": mode, "repeat": repetition, "fold": fold}
        for row in test:
            splits.append({**key_base, "row_position": int(row), "id": ids[row], "plot": groups[row], "trial": trials[row]})
        for inner_fold, (a, b) in enumerate(inner, 1):
            for row in train[b]:
                inner_records.append({**key_base, "inner_fold": inner_fold, "row_position": int(row),
                                      "id": ids[row], "plot": groups[row]})
        tasks = [(model, "all") for model in models]
        if ablation and mode == "nested":
            tasks.extend(("ExtraTrees", feature_set) for feature_set in sets if feature_set != "all")
        for model, feature_set in tasks:
            cols = sets[feature_set]
            search_seed = cfg.get("search_seed", 20261009 if model == "ExtraTrees" else 42)
            pipe, grid = _search(model, len(cols), seed, protocol, min(len(a) for a, _ in inner), search_seed)
            search = GridSearchCV(pipe, grid, cv=inner, scoring="neg_root_mean_squared_error",
                                  n_jobs=n_jobs, error_score="raise", refit=True)
            search.fit(X.iloc[train][cols], y[train])
            predicted = np.asarray(search.predict(X.iloc[test][cols])).ravel()
            metrics = regression_metrics(y[test], predicted)
            selected = [cols[i] for i in search.best_estimator_.named_steps["select"].indices_]
            record = {**key_base, "model": model, "feature_set": feature_set, "estimator_seed": seed,
                      "inner_split_seed": seed + fold, "search_seed": search_seed,
                      "train_n": len(train), "test_n": len(test), "train_plots": len(set(groups[train])),
                      "test_plots": len(set(groups[test])), "best_params": search.best_params_,
                      "selected_features": selected, "inner_RMSE": -float(search.best_score_), "metrics": metrics}
            fit_records.append(record)
            _save(out / "searches" / f"{mode}_{repetition}_{fold}_{model}_{feature_set}.json",
                  {**record, "candidates": [{"parameters": p, "mean_inner_RMSE": -float(score)}
                   for p, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])]})
            for row, value in zip(test, predicted):
                predictions.append({**key_base, "model": model, "feature_set": feature_set, "row_position": int(row),
                                    "id": ids[row], "plot": groups[row], "trial": trials[row],
                                    "observed": float(y[row]), "predicted": float(value)})
    oof = pd.DataFrame(predictions)
    oof.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "outer_splits.csv", index=False)
    pd.DataFrame(inner_records).to_csv(out / "inner_validation_splits.csv", index=False)
    _save(out / "fits.json", fit_records)
    pooled = []
    for (model, feature_set, repeat), part in oof[oof["mode"] == "nested"].groupby(["model", "feature_set", "repeat"], sort=True):
        if len(part) != len(table) or part.row_position.nunique() != len(table):
            raise RuntimeError("Every observation must be predicted once per repetition and feature set.")
        pooled.append({"model": model, "feature_set": feature_set, "repeat": int(repeat),
                       **regression_metrics(part.observed.to_numpy(), part.predicted.to_numpy())})
    pooled_frame = pd.DataFrame(pooled)
    pooled_frame.to_csv(out / "repetition_metrics.csv", index=False)
    summary_rows = []
    for (model, feature_set), part in pooled_frame.groupby(["model", "feature_set"], sort=True):
        record = {"model": model, "feature_set": feature_set, "repetitions": len(part)}
        for metric in ("R2", "RMSE", "MAE", "Bias", "NRMSE"):
            numeric = pd.to_numeric(part[metric], errors="coerce")
            record[metric + "_mean"] = numeric.mean()
            record[metric + "_SD"] = numeric.std(ddof=1) if len(part) > 1 else None
        summary_rows.append(record)
    pd.DataFrame(summary_rows).to_csv(out / "model_comparison.csv", index=False)
    delta, contributions = _paired_contributions(oof, bootstrap, bootstrap_seed)
    pd.DataFrame(delta, columns=["group", "repeat", "delta_RMSE"]).to_csv(out / "feature_contributions_by_repeat.csv", index=False)
    _save(out / "feature_contributions.json", contributions)
    new_refits = []
    if cfg.get("refit", False):
        (out / "new_refits").mkdir()
        inner = grouped_splits(groups, inner_folds, seeds[0])
        for model in models:
            pipe, grid = _search(model, len(names), seeds[0], protocol, min(len(a) for a, _ in inner),
                                 cfg.get("search_seed", 20261009 if model == "ExtraTrees" else 42))
            search = GridSearchCV(pipe, grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=n_jobs, error_score="raise")
            search.fit(X, y)
            path = out / "new_refits" / f"{target}_{model}_new_refit.joblib"
            joblib.dump(search.best_estimator_, path)
            metadata = {"deployment_schema": 1, "target": target, "algorithm": model,
                        "unit": "g m-2", "feature_contract": contract,
                        "scikit_learn": sklearn.__version__,
                        "selected_features": [names[int(i)] for i in search.best_estimator_.named_steps["select"].indices_],
                        "training_ranges": {n: {"min": float(X[n].min()), "max": float(X[n].max())} for n in names},
                        "file": path.name, "sha256": _sha(path), "feature_names": names,
                        "best_params": search.best_params_, "selection_RMSE": -float(search.best_score_),
                        "role": "New full-data refit; selection RMSE is not independent evaluation."}
            _save(path.with_suffix(".json"), metadata)
            new_refits.append(metadata)
    report = {"schema_version": 1, "operation": "new_grouped_model_development", "target": target,
              "input_sha256": _sha(source), "configuration_sha256": _sha(config_path), "n": len(table),
              "plots": len(set(groups)), "trials": len(set(trials)), "python": platform.python_version(),
              "scikit_learn": sklearn.__version__, "protocol": protocol, "resolved_configuration": resolved,
              "fit_count": len(fit_records), "new_refits": new_refits, "bundled_weights_modified": False,
              "metric_scope": "Nested metrics use pooled outer-fold predictions per repetition; SD is across repetitions.",
              "feature_contribution_definition": "RMSE without group minus RMSE with all groups; paired held-out plots.",
              "bootstrap_scope": "Conditional on stored OOF errors; not a bootstrap of the complete training procedure.",
              "study_reproduction": "New portable run. Exact archived scores require original locked rows, preprocessing and protocol; quick mode is a software smoke test.",
              "spectral_contract": "NDRE means (NIR-RE)/(NIR+RE) in all workflows. Supplied table values must follow this definition.",
              "port_changes": ["All-empty fold predictors retain their column positions through keep_empty_features=True.",
                               "PLSR components are bounded by the smallest inner training sample count.",
                               "No context predictors, private source reconciliation, target editing or row selection are applied."]}
    _save(out / "report.json", report)
    return report
