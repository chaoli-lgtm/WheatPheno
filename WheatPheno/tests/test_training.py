"""Scientific split, preprocessing and experiment-output checks on synthetic data."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from wheatpheno.training import (SpearmanSelector, _search, evaluate_table,
                               grouped_splits, regression_metrics, make_synthetic_table,
                               standard_feature_names)


def test_rank_selector_uses_absolute_association_and_stable_ties():
    y = np.arange(10, dtype=float)
    x = np.column_stack([y, -y, np.ones(10)])
    selector = SpearmanSelector(2).fit(x, y)
    assert selector.indices_.tolist() == [0, 1]
    np.testing.assert_allclose(selector.scores_, [1, 1, 0])
    assert selector.transform(x).shape == (10, 2)


def test_group_splits_never_separate_repeated_plot_rows():
    groups = np.repeat([f"plot{i}" for i in range(12)], 3)
    seen = []
    for train, test in grouped_splits(groups, 3, 42):
        assert set(groups[train]).isdisjoint(groups[test])
        seen.extend(test)
        for a, b in grouped_splits(groups[train], 2, 43):
            assert set(groups[train[a]]).isdisjoint(groups[train[b]])
    assert sorted(seen) == list(range(36))
    with pytest.raises(ValueError, match="distinct plots"):
        grouped_splits(np.array(["one", "one"]), 2, 42)


def test_imputation_and_selection_fit_only_given_training_rows():
    pipe, _ = _search("ExtraTrees", 3, 42, "quick", 4, 42)
    x = np.array([[0., 1., np.nan], [1., np.nan, 3.], [2., 3., 4.], [3., 4., 5.]])
    pipe.fit(x, np.arange(4.))
    medians = pipe.named_steps["impute"].statistics_.copy()
    before = pipe.named_steps["select"].indices_.copy()
    pipe.predict(np.array([[1e12, np.nan, np.nan]]))
    np.testing.assert_allclose(medians, [1.5, 3., 4.])
    np.testing.assert_array_equal(pipe.named_steps["impute"].statistics_, medians)
    np.testing.assert_array_equal(pipe.named_steps["select"].indices_, before)


def test_metrics_pool_observations_and_handle_constant_target():
    m = regression_metrics(np.array([0., 2.]), np.array([1., 2.]))
    assert m["RMSE"] == pytest.approx(np.sqrt(.5))
    assert m["Bias"] == .5
    assert m["NRMSE"] == pytest.approx(np.sqrt(.5)/2)
    m = regression_metrics(np.ones(3), np.zeros(3))
    assert m["R2"] is None and m["NRMSE"] is None


def test_synthetic_fixture_has_repeated_plots_and_conventional_ndre(tmp_path):
    frame = make_synthetic_table(tmp_path/"synthetic.csv")
    assert len(frame) == 48 and frame["plot"].nunique() == 12 and frame.trial.nunique() == 3
    assert (frame.groupby("plot").size() == 4).all()
    assert set(standard_feature_names()).issubset(frame.columns)
    # CIre = NIR / RE - 1 implies conventional NDRE = CIre / (CIre + 2).
    np.testing.assert_allclose(frame.NDRE, frame.CIre / (frame.CIre + 2))
    assert "NIR2" not in frame
    with pytest.raises(FileExistsError):
        make_synthetic_table(tmp_path/"synthetic.csv")


@pytest.fixture
def experiment(tmp_path):
    rng = np.random.default_rng(42)
    n = 24
    x = rng.normal(size=(n, 3))
    frame = pd.DataFrame(x, columns=["OSAVI", "CHM_MEAN", "Texture_NIR_mean"])
    frame["observed"] = 50 + 10*x[:, 0] + 5*x[:, 1] + rng.normal(size=n)
    frame["id"] = [f"r{i}" for i in range(n)]
    frame["plot"] = np.repeat([f"p{i}" for i in range(12)], 2)
    frame["trial"] = np.repeat(["trial_A", "trial_B", "trial_C"], 8)
    frame.loc[1, "OSAVI"] = np.nan
    frame.to_csv(tmp_path/"input.csv", index=False)
    cfg = {"input_csv": "input.csv", "protocol": "quick", "target": "AGB",
           "feature_names": ["OSAVI", "CHM_MEAN", "Texture_NIR_mean"],
           "models": ["RF", "SVR", "PLSR", "ExtraTrees"], "seeds": [42, 142],
           "outer_folds": 3, "inner_folds": 2,
           "feature_group_ablation": True, "bootstrap_replicates": 20,
           "refit": True}
    (tmp_path/"config.json").write_text(json.dumps(cfg))
    return tmp_path, frame, cfg


def test_full_quick_experiment_keeps_paired_splits_and_refits_separate(experiment):
    root, frame, _ = experiment
    out = root/"results"
    report = evaluate_table(root/"config.json", out)
    assert report["fit_count"] == 42
    assert report["bundled_weights_modified"] is False
    assert len(report["new_refits"]) == 4
    outer = pd.read_csv(out/"outer_splits.csv")
    inner = pd.read_csv(out/"inner_validation_splits.csv")
    for (mode, repeat, fold), heldout in outer.groupby(["mode", "repeat", "fold"]):
        matching = inner[(inner["mode"] == mode) & (inner["repeat"] == repeat) & (inner["fold"] == fold)]
        assert set(heldout["plot"]).isdisjoint(matching["plot"])
        assert len(matching) + len(heldout) == len(frame)
        for _, part in matching.groupby("plot"):
            assert part.inner_fold.nunique() == 1
    predictions = pd.read_csv(out/"predictions.csv")
    nested = predictions[predictions["mode"] == "nested"]
    for _, group in nested.groupby(["model", "feature_set", "repeat"]):
        assert len(group) == len(frame) and group.row_position.nunique() == len(frame)
    metrics = pd.read_csv(out/"repetition_metrics.csv")
    part = nested[(nested.model == "ExtraTrees") & (nested.feature_set == "all") & (nested["repeat"] == 1)]
    expected = regression_metrics(part.observed.to_numpy(), part.predicted.to_numpy())
    metric = metrics[(metrics.model == "ExtraTrees") & (metrics.feature_set == "all") & (metrics["repeat"] == 1)].iloc[0]
    assert metric.RMSE == pytest.approx(expected["RMSE"])
    intervals = json.loads((out/"feature_contributions.json").read_text())
    assert len(intervals) == 3
    assert all(record["lower_95"] <= record["upper_95"] for record in intervals)
    delta = pd.read_csv(out/"feature_contributions_by_repeat.csv")
    for name, group in delta.groupby("group"):
        full_rmse = metrics[(metrics.model == "ExtraTrees") & (metrics.feature_set == "all")].sort_values("repeat").RMSE.to_numpy()
        reduced_rmse = metrics[(metrics.model == "ExtraTrees") & (metrics.feature_set == "without_"+name)].sort_values("repeat").RMSE.to_numpy()
        np.testing.assert_allclose(group.sort_values("repeat").delta_RMSE, reduced_rmse-full_rmse)
    artifact = out/"new_refits"/"AGB_ExtraTrees_new_refit.joblib"
    model = joblib.load(artifact)
    assert np.isfinite(model.predict(frame[["OSAVI", "CHM_MEAN", "Texture_NIR_mean"]])).all()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == report["new_refits"][3]["sha256"]
    with pytest.raises(FileExistsError):
        evaluate_table(root/"config.json", out)


@pytest.mark.parametrize("mutation,match", [
    ({"feature_names": ["observed"]}, "schema"),
    ({"models": ["Unknown"]}, "supported"),
    ({"seeds": [42, 42]}, "distinct"),
    ({"outer_folds": 20}, "distinct plots"),
    ({"feature_group_ablation": "true"}, "true or false"),
])
def test_invalid_protocol_fails_before_writing(experiment, mutation, match):
    root, _, cfg = experiment
    cfg.update(mutation)
    (root/"config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=match):
        evaluate_table(root/"config.json", root/"results")
    assert not (root/"results").exists()


def test_cross_trial_plot_identifier_collision_is_rejected(experiment):
    root, frame, _ = experiment
    frame.loc[8, "plot"] = "p0"
    frame.to_csv(root/"input.csv", index=False)
    with pytest.raises(ValueError, match="globally unique"):
        evaluate_table(root/"config.json", root/"results")
