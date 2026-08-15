"""
test_evaluation_framework.py
"""

import numpy as np
import pandas as pd
import pytest

import evaluation_framework as ef


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_df():
    """A small, hand-built dataset: enough cost-eligible rows (recall + fpr
    both present) per category to exercise the GA's fitness function
    (which requires at least 3 valid rows), without depending on the real
    project data."""
    rows = [
        {"category": "Banking/URL Phishing", "study": "s1", "model": "m1",
         "accuracy": 95, "precision": 94, "recall": 90, "f1": 92, "fpr": 4,
         "dataset": "d1", "source": "lit", "evidence_type": "secondary"},
        {"category": "Banking/URL Phishing", "study": "s2", "model": "m2",
         "accuracy": 80, "precision": 78, "recall": 70, "f1": 74, "fpr": 20,
         "dataset": "d2", "source": "lit", "evidence_type": "primary"},
        {"category": "Business Email Compromise", "study": "s3", "model": "m3",
         "accuracy": 97, "precision": 96, "recall": 95, "f1": 95, "fpr": 1,
         "dataset": "d3", "source": "lit", "evidence_type": "secondary"},
        {"category": "Business Email Compromise", "study": "s4", "model": "m4",
         "accuracy": 60, "precision": 58, "recall": 55, "f1": 56, "fpr": 30,
         "dataset": "d4", "source": "lit", "evidence_type": "secondary"},
        {"category": "Mass/Email Phishing", "study": "s5", "model": "m5",
         "accuracy": 90, "precision": 88, "recall": 85, "f1": 86, "fpr": 8,
         "dataset": "d5", "source": "lit", "evidence_type": "primary"},
        {"category": "Spear Phishing", "study": "s6", "model": "m6",
         "accuracy": 85, "precision": np.nan, "recall": np.nan, "f1": np.nan, "fpr": np.nan,
         "dataset": "d6", "source": "lit", "evidence_type": "secondary"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_score_row
# ---------------------------------------------------------------------------
class TestComputeScoreRow:
    def test_full_row_weighted_average(self):
        row = pd.Series({"recall": 80.0, "accuracy": 90.0, "precision": 70.0, "f1": 75.0, "fpr_inverse": 95.0})
        weights = {"recall": 0.5, "accuracy": 0.2, "precision": 0.1, "f1": 0.1, "fpr_inverse": 0.1}
        expected = 80 * 0.5 + 90 * 0.2 + 70 * 0.1 + 75 * 0.1 + 95 * 0.1
        assert ef.compute_score_row(row, weights) == round(expected, 2)

    def test_missing_metric_is_renormalised_not_zero(self):
        row = pd.Series({"recall": 80.0, "accuracy": np.nan, "precision": 70.0, "f1": 75.0, "fpr_inverse": np.nan})
        weights = {"recall": 0.5, "accuracy": 0.2, "precision": 0.1, "f1": 0.1, "fpr_inverse": 0.1}
        remaining_weight = 0.5 + 0.1 + 0.1
        expected = (80 * 0.5 + 70 * 0.1 + 75 * 0.1) / remaining_weight
        assert ef.compute_score_row(row, weights) == round(expected, 2)

    def test_all_metrics_missing_returns_nan(self):
        row = pd.Series({c: np.nan for c in ef.CRITERIA_ORDER})
        assert pd.isna(ef.compute_score_row(row, ef.UNIFORM_PRIOR))

    def test_zero_weighted_metric_ignored_even_if_present(self):
        row = pd.Series({"recall": 50.0, "accuracy": 100.0, "precision": 50.0, "f1": 50.0, "fpr_inverse": 50.0})
        weights = {"recall": 1.0, "accuracy": 0.0, "precision": 0.0, "f1": 0.0, "fpr_inverse": 0.0}
        assert ef.compute_score_row(row, weights) == 50.0

    def test_score_stays_within_0_100(self, toy_df):
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        scores = df.apply(lambda r: ef.compute_score_row(r, ef.UNIFORM_PRIOR), axis=1).dropna()
        assert (scores >= 0).all() and (scores <= 100).all()


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------
class TestValidateDataset:
    def test_clean_dataset_has_no_issues(self, toy_df):
        assert ef.validate_dataset(toy_df) == []

    def test_catches_unknown_category(self, toy_df):
        df = toy_df.copy()
        df.loc[0, "category"] = "Not A Real Category"
        issues = ef.validate_dataset(df)
        assert any("Unknown category" in i for i in issues)

    def test_catches_unknown_evidence_type(self, toy_df):
        df = toy_df.copy()
        df.loc[0, "evidence_type"] = "tertiary"
        issues = ef.validate_dataset(df)
        assert any("Unknown evidence_type" in i for i in issues)

    def test_catches_out_of_range_metric(self, toy_df):
        df = toy_df.copy()
        df.loc[0, "accuracy"] = 150
        issues = ef.validate_dataset(df)
        assert any("accuracy" in i and "0-100" in i for i in issues)

    def test_catches_negative_fpr(self, toy_df):
        df = toy_df.copy()
        df.loc[0, "fpr"] = -5
        issues = ef.validate_dataset(df)
        assert any("fpr" in i and "negative" in i for i in issues)

    def test_catches_duplicate_rows(self, toy_df):
        df = pd.concat([toy_df, toy_df.iloc[[0]]], ignore_index=True)
        issues = ef.validate_dataset(df)
        assert any("duplicate" in i for i in issues)


# ---------------------------------------------------------------------------
# load_data error handling
# ---------------------------------------------------------------------------
class TestLoadData:
    def test_missing_file_raises_clear_error(self):
        with pytest.raises(FileNotFoundError, match="Could not find"):
            ef.load_data("this_file_does_not_exist.csv")

    def test_missing_required_column_raises_value_error(self, tmp_path, toy_df):
        bad = toy_df.drop(columns=["recall"])
        path = tmp_path / "bad.csv"
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required column"):
            ef.load_data(str(path))

    def test_fpr_inverse_computed_correctly(self, tmp_path, toy_df):
        path = tmp_path / "toy.csv"
        toy_df.to_csv(path, index=False)
        df = ef.load_data(str(path))
        valid = df["fpr"].notna()
        assert np.allclose(df.loc[valid, "fpr_inverse"], 100 - df.loc[valid, "fpr"])


# ---------------------------------------------------------------------------
# Spearman correlation and expected cost
# ---------------------------------------------------------------------------
class TestStatsHelpers:
    def test_spearman_perfect_positive_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        assert ef.spearman_corr(a, b) == pytest.approx(1.0)

    def test_spearman_perfect_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([40.0, 30.0, 20.0, 10.0])
        assert ef.spearman_corr(a, b) == pytest.approx(-1.0)

    def test_expected_cost_row_matches_formula(self):
        row = pd.Series({"recall": 90.0, "fpr": 5.0})
        cost = ef.expected_cost_row(row, cost_fn=10, cost_fp=2)
        expected = (1 - 0.90) * 10 + 0.05 * 2
        assert cost == pytest.approx(expected)

    def test_expected_cost_row_nan_when_metric_missing(self):
        row = pd.Series({"recall": np.nan, "fpr": 5.0})
        assert pd.isna(ef.expected_cost_row(row, cost_fn=10, cost_fp=2))


# ---------------------------------------------------------------------------
# GA weight validity and determinism
# ---------------------------------------------------------------------------
class TestGAWeights:
    def test_weights_sum_to_one(self, toy_df):
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        weights, _, _ = ef.ga_optimise_weights(df, cost_fn=1, cost_fp=1, seed=0)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_weights_are_non_negative(self, toy_df):
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        weights, _, _ = ef.ga_optimise_weights(df, cost_fn=1, cost_fp=1, seed=0)
        assert all(w >= 0 for w in weights.values())

    def test_same_seed_reproduces_identical_result(self, toy_df):
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        w1, f1, _ = ef.ga_optimise_weights(df, cost_fn=1, cost_fp=1, seed=7)
        w2, f2, _ = ef.ga_optimise_weights(df, cost_fn=1, cost_fp=1, seed=7)
        assert w1 == w2
        assert f1 == f2

    def test_multistart_picks_the_best_of_its_seeds(self, toy_df):
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        seeds = (0, 1, 2)
        _, best_fitness, _, diagnostics = ef.multistart_optimise_weights(df, cost_fn=1, cost_fp=1, seeds=seeds)
        individual_fitness = [ef.ga_optimise_weights(df, 1, 1, seed=s)[1] for s in seeds]
        assert best_fitness == pytest.approx(max(individual_fitness))
        assert diagnostics["winning_seed"] in seeds


# ---------------------------------------------------------------------------
# Grid-search cross-check
# ---------------------------------------------------------------------------
class TestGridSearch:
    def test_grid_search_returns_valid_simplex_weights(self, toy_df):
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        weights, fitness, n_points = ef.grid_search_weights(df, cost_fn=1, cost_fp=1, step=0.1)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert n_points > 0

    def test_grid_search_is_within_reach_of_ga(self, toy_df):
        """The GA should not be meaningfully worse than exhaustive grid
        search at a comparable resolution - a large negative gap would
        indicate the GA is failing to find good solutions it should."""
        df = toy_df.copy()
        df["fpr_inverse"] = 100 - df["fpr"]
        ga_weights, ga_fitness, _, _ = ef.multistart_optimise_weights(df, cost_fn=1, cost_fp=1, seeds=range(5))
        _, grid_fitness, _ = ef.grid_search_weights(df, cost_fn=1, cost_fp=1, step=0.05)
        assert ga_fitness >= grid_fitness - 0.05


# ---------------------------------------------------------------------------
# category_summary
# ---------------------------------------------------------------------------
class TestCategorySummary:
    def test_summary_has_one_row_per_category(self, toy_df):
        df = toy_df.copy()
        df["DES"] = 50.0
        summary = ef.category_summary(df)
        assert set(summary.index) == set(toy_df["category"].unique())

    def test_summary_counts_primary_and_secondary_correctly(self, toy_df):
        df = toy_df.copy()
        df["DES"] = 50.0
        summary = ef.category_summary(df)
        row = summary.loc["Banking/URL Phishing"]
        assert row["n_primary"] == 1
        assert row["n_secondary"] == 1
