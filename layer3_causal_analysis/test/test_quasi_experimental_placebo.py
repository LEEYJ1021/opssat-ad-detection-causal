"""quasi_experimental_placebo.py 핵심 로직 단위테스트 — 효과크기 계산
(cohens_d_abs, log_var_ratio, compute_effects), 경험적 p-value, Mann-Whitney U."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_experimental_placebo import (
    cohens_d_abs, log_var_ratio, compute_effects,
    compute_empirical_pvalues, summarize_channel_significance, run_mannwhitney,
)
from conftest import make_variance_surge_series, make_flat_series


# =============================================================================
# cohens_d_abs / log_var_ratio
# =============================================================================

class TestCohensDAbs:
    def test_always_nonnegative(self):
        rng = np.random.default_rng(0)
        a = rng.normal(-5, 1, 200)
        b = rng.normal(0, 1, 200)
        assert cohens_d_abs(a, b) > 0
        assert cohens_d_abs(a, b) == pytest.approx(cohens_d_abs(b, a), rel=1e-9)

    def test_too_few_points_nan(self):
        assert np.isnan(cohens_d_abs(np.array([1.0]), np.array([1.0, 2.0])))


class TestLogVarRatio:
    def test_increase_in_variance_is_positive(self):
        rng = np.random.default_rng(1)
        pre = rng.normal(0, 1, 100)
        post = rng.normal(0, 5, 100)
        r = log_var_ratio(pre, post)
        assert r > 0
        assert r == pytest.approx(np.log(25), abs=0.6)  # var ratio ~ 5^2=25

    def test_decrease_in_variance_is_negative(self):
        rng = np.random.default_rng(2)
        pre = rng.normal(0, 5, 100)
        post = rng.normal(0, 1, 100)
        assert log_var_ratio(pre, post) < 0

    def test_too_few_points_nan(self):
        assert np.isnan(log_var_ratio(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])))


# =============================================================================
# compute_effects
# =============================================================================

class TestComputeEffects:
    def test_variance_surge_shows_up_in_diff_diff2_not_level(self):
        values = make_variance_surge_series(300, onset_idx=150, pre_std=0.05, post_std=3.0)
        out = compute_effects(values, pivot_idx=150)
        assert out["diff_logvar"] > 1.0
        assert out["diff2_logvar"] > 1.0
        # level 효과크기는 diff/diff2에 비해 훨씬 작아야 한다(§3 핵심 주장의 재현)
        assert out["level_d_abs"] < out["diff_logvar"]

    def test_flat_series_gives_near_zero_effects(self):
        values = make_flat_series(200, seed=42)
        out = compute_effects(values, pivot_idx=100)
        assert abs(out["diff_logvar"]) < 1.0
        assert out["level_d_abs"] < 1.0


# =============================================================================
# compute_empirical_pvalues
# =============================================================================

class TestComputeEmpiricalPvalues:
    def test_observed_value_far_above_placebo_gives_small_pvalue(self):
        observed_df = pd.DataFrame({
            "channel": ["CADC0872"], "segment": [1], "n_points": [100],
            "level_d_abs": [10.0], "diff_logvar": [10.0], "diff2_logvar": [10.0],
        })
        rng = np.random.default_rng(0)
        placebo_df = pd.DataFrame({
            "channel": ["CADC0872"] * 100,
            "level_d_abs": rng.normal(0, 1, 100),
            "diff_logvar": rng.normal(0, 1, 100),
            "diff2_logvar": rng.normal(0, 1, 100),
        })
        out = compute_empirical_pvalues(observed_df, placebo_df)
        row = out.iloc[0]
        assert row["level_d_abs_pvalue"] < 0.05
        assert row["diff_logvar_pvalue"] < 0.05

    def test_insufficient_placebo_pool_gives_nan_pvalue(self):
        observed_df = pd.DataFrame({
            "channel": ["CADC0872"], "segment": [1], "n_points": [100],
            "level_d_abs": [1.0], "diff_logvar": [1.0], "diff2_logvar": [1.0],
        })
        placebo_df = pd.DataFrame({
            "channel": ["CADC0872"] * 5,   # 20개 미만 -> 판정 불가
            "level_d_abs": [0.1] * 5, "diff_logvar": [0.1] * 5, "diff2_logvar": [0.1] * 5,
        })
        out = compute_empirical_pvalues(observed_df, placebo_df)
        assert np.isnan(out.iloc[0]["level_d_abs_pvalue"])
        assert out.iloc[0]["level_d_abs_placebo_n"] == 5


# =============================================================================
# summarize_channel_significance
# =============================================================================

class TestSummarizeChannelSignificance:
    def test_frac_significant_computed_per_channel(self, final_channels):
        rows = []
        for ch in final_channels:
            for p in [0.01, 0.02, 0.5, 0.9]:
                rows.append({"channel": ch, "level_d_abs_pvalue": p,
                             "diff_logvar_pvalue": p, "diff2_logvar_pvalue": p})
        pval_df = pd.DataFrame(rows)
        summary = summarize_channel_significance(pval_df)
        assert len(summary) == len(final_channels)
        row = summary.iloc[0]
        assert row["level_d_abs_frac_significant"] == pytest.approx(0.5)
        assert row["level_d_abs_n_valid"] == 4


# =============================================================================
# run_mannwhitney
# =============================================================================

class TestRunMannwhitney:
    def test_anomaly_greater_than_placebo_is_significant(self, final_channels):
        rng = np.random.default_rng(7)
        ch = final_channels[0]
        observed_df = pd.DataFrame({
            "channel": [ch] * 30,
            "level_d_abs": rng.normal(5, 1, 30),
            "diff_logvar": rng.normal(5, 1, 30),
            "diff2_logvar": rng.normal(5, 1, 30),
        })
        placebo_df = pd.DataFrame({
            "channel": [ch] * 200,
            "level_d_abs": rng.normal(0, 1, 200),
            "diff_logvar": rng.normal(0, 1, 200),
            "diff2_logvar": rng.normal(0, 1, 200),
        })
        mw_df = run_mannwhitney(observed_df, placebo_df)
        row = mw_df[(mw_df["channel"] == ch) & (mw_df["feature"] == "level_d_abs")].iloc[0]
        assert row["p_value"] < 0.001
        assert row["median_anomaly"] > row["median_placebo"]

    def test_insufficient_sample_flagged(self, final_channels):
        ch = final_channels[0]
        observed_df = pd.DataFrame({"channel": [ch] * 2, "level_d_abs": [1.0, 2.0],
                                     "diff_logvar": [1.0, 2.0], "diff2_logvar": [1.0, 2.0]})
        placebo_df = pd.DataFrame({"channel": [ch] * 2, "level_d_abs": [0.1, 0.2],
                                    "diff_logvar": [0.1, 0.2], "diff2_logvar": [0.1, 0.2]})
        mw_df = run_mannwhitney(observed_df, placebo_df)
        assert (mw_df["note"] == "표본부족 (n<5)").all()
        assert mw_df["p_value"].isna().all()

    def test_bonferroni_correction_applied(self, final_channels):
        rng = np.random.default_rng(8)
        rows_obs, rows_pla = [], []
        for ch in final_channels:
            rows_obs.append(pd.DataFrame({
                "channel": [ch] * 10,
                "level_d_abs": rng.normal(3, 1, 10),
                "diff_logvar": rng.normal(3, 1, 10),
                "diff2_logvar": rng.normal(3, 1, 10),
            }))
            rows_pla.append(pd.DataFrame({
                "channel": [ch] * 50,
                "level_d_abs": rng.normal(0, 1, 50),
                "diff_logvar": rng.normal(0, 1, 50),
                "diff2_logvar": rng.normal(0, 1, 50),
            }))
        observed_df = pd.concat(rows_obs, ignore_index=True)
        placebo_df = pd.concat(rows_pla, ignore_index=True)
        mw_df = run_mannwhitney(observed_df, placebo_df)
        valid = mw_df.dropna(subset=["p_value"])
        # bonferroni 보정값은 항상 원래 p_value 이상이어야 하고 1을 넘지 않아야 함
        assert (valid["p_value_bonferroni"] >= valid["p_value"] - 1e-12).all()
        assert (valid["p_value_bonferroni"] <= 1.0).all()
