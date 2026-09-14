"""onset_mc_propagation.py 핵심 로직 단위테스트 — Cohen's d, Welch 검정
피처추출, reliability tier 판정, ①②보정(correct_segment/correct_mc_results)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_mc_propagation import (
    cohens_d, welch_test_features, tier, correct_segment, correct_mc_results,
    MIN_DRAWS_HIGH, MIN_DRAWS_MEDIUM,
)
from conftest import make_variance_surge_series, make_step_series


# =============================================================================
# cohens_d
# =============================================================================

class TestCohensD:
    def test_too_few_points_returns_nan(self):
        assert np.isnan(cohens_d(np.array([1.0]), np.array([1.0, 2.0])))

    def test_zero_pooled_sd_returns_nan(self):
        a = np.full(5, 1.0)
        b = np.full(5, 1.0)
        assert np.isnan(cohens_d(a, b))

    def test_known_effect_size_sign_and_magnitude(self):
        rng = np.random.default_rng(0)
        a = rng.normal(5, 1, 500)   # post
        b = rng.normal(0, 1, 500)   # pre
        d = cohens_d(a, b)
        assert d > 0
        assert d == pytest.approx(5.0, abs=0.3)

    def test_symmetry_flips_sign(self):
        rng = np.random.default_rng(0)
        a = rng.normal(5, 1, 300)
        b = rng.normal(0, 1, 300)
        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a), rel=1e-9)


# =============================================================================
# welch_test_features
# =============================================================================

class TestWelchTestFeatures:
    def test_level_shift_is_detected(self):
        values = make_step_series(200, onset_idx=100, pre_mean=0.0, post_mean=10.0,
                                   noise_std=0.2)
        out = welch_test_features(values, onset_idx=100)
        assert out["level_p"] < 0.01
        assert out["level_d"] > 0

    def test_variance_surge_is_detected_in_diff_diff2_not_level(self):
        values = make_variance_surge_series(300, onset_idx=150, pre_std=0.05, post_std=3.0)
        out = welch_test_features(values, onset_idx=150)
        # diff/diff2의 제곱값 비교 -> post가 크므로 유의해야 함
        assert out["diff_p"] < 0.05
        assert out["diff2_p"] < 0.05
        assert out["diff_d"] > 0
        assert out["diff2_d"] > 0

    def test_insufficient_window_returns_nan(self):
        # onset이 배열 맨 끝 근처라 post 구간을 거의 못 만드는 경우
        values = np.arange(30, dtype=float)
        out = welch_test_features(values, onset_idx=28)
        assert np.isnan(out["level_p"])
        assert np.isnan(out["diff_p"])
        assert np.isnan(out["diff2_p"])


# =============================================================================
# tier — reliability tier 판정
# =============================================================================

class TestTier:
    def test_zero_draws_is_excluded(self):
        assert tier(0) == "excluded"

    def test_boundaries(self):
        assert tier(MIN_DRAWS_HIGH) == "high"
        assert tier(MIN_DRAWS_HIGH - 1) == "medium"
        assert tier(MIN_DRAWS_MEDIUM) == "medium"
        assert tier(MIN_DRAWS_MEDIUM - 1) == "low"


# =============================================================================
# correct_segment — 부호-무관 크기 / 부호 있는 값 / 방향 일관성
# =============================================================================

class TestCorrectSegment:
    def test_all_draws_agree_in_sign_gives_consistency_one(self):
        sub = pd.DataFrame({
            "level_d": [1.0, 1.2, 0.9, 1.1],
            "level_p": [0.01, 0.02, 0.03, 0.04],
        })
        out = correct_segment(sub, "level")
        assert out["level_direction_consistency"] == pytest.approx(1.0)
        assert out["level_d_signed_median"] == pytest.approx(np.median([1.0, 1.2, 0.9, 1.1]))
        assert out["level_d_abs_median"] == pytest.approx(np.median([1.0, 1.2, 0.9, 1.1]))

    def test_mixed_sign_draws_gives_partial_consistency(self):
        sub = pd.DataFrame({
            "level_d": [1.0, -1.0, -1.0, -1.0],
            "level_p": [0.5, 0.5, 0.5, 0.5],
        })
        out = correct_segment(sub, "level")
        assert out["level_direction_consistency"] == pytest.approx(0.75)
        # 부호 무관 크기는 모두 1.0이므로 abs_median도 1.0
        assert out["level_d_abs_median"] == pytest.approx(1.0)

    def test_empty_draws_returns_nan_fields(self):
        sub = pd.DataFrame({"level_d": [np.nan, np.nan], "level_p": [np.nan, np.nan]})
        out = correct_segment(sub, "level")
        assert np.isnan(out["level_d_abs_median"])
        assert np.isnan(out["level_direction_consistency"])
        assert np.isnan(out["level_frac_significant"])

    def test_frac_significant_counts_p_below_threshold(self):
        sub = pd.DataFrame({
            "level_d": [1, 1, 1, 1],
            "level_p": [0.01, 0.2, 0.03, 0.9],
        })
        out = correct_segment(sub, "level")
        assert out["level_frac_significant"] == pytest.approx(0.5)


# =============================================================================
# correct_mc_results — 세그먼트 -> 채널 단위 통합 (reliability tier + 가중평균)
# =============================================================================

class TestCorrectMcResults:
    def _make_draws(self, channel, segment, n_draws, d_val=1.0, p_val=0.01):
        return pd.DataFrame({
            "channel": channel, "segment": segment,
            "level_d": [d_val] * n_draws, "level_p": [p_val] * n_draws,
            "diff_d": [d_val] * n_draws, "diff_p": [p_val] * n_draws,
            "diff2_d": [d_val] * n_draws, "diff2_p": [p_val] * n_draws,
        })

    def test_tier_assignment_from_draw_counts(self):
        draws = pd.concat([
            self._make_draws("CADC0872", 1, MIN_DRAWS_HIGH),
            self._make_draws("CADC0872", 2, MIN_DRAWS_MEDIUM),
            self._make_draws("CADC0872", 3, MIN_DRAWS_MEDIUM - 1),
        ], ignore_index=True)
        seg_summary = pd.DataFrame({
            "channel": ["CADC0872"], "segment": [99], "n_points": [10],
            "n_valid_draws": [0], "skip_reason": ["onset 미검출"],
        })
        full_df, channel_df, tier_report = correct_mc_results(draws, seg_summary)

        tiers = full_df.set_index("segment")["reliability_tier"].to_dict()
        assert tiers[1] == "high"
        assert tiers[2] == "medium"
        assert tiers[3] == "low"
        assert tiers[99] == "excluded"

        # 채널 요약에는 high+medium만 반영 -> n_segments_reliable == 2
        row = channel_df[channel_df["channel"] == "CADC0872"].iloc[0]
        assert row["n_segments_reliable(high+medium)"] == 2
        assert row["n_segments_total"] == 4
        assert row["n_segments_excluded(low+none)"] == 2

    def test_weighted_mean_favors_higher_draw_count_segment(self):
        # 세그먼트1: draw 많고 d=10, 세그먼트2: draw 적당하지만 d=0
        # 가중평균은 세그먼트1 쪽으로 크게 치우쳐야 함
        draws = pd.concat([
            self._make_draws("CADC0888", 1, MIN_DRAWS_HIGH, d_val=10.0),
            self._make_draws("CADC0888", 2, MIN_DRAWS_MEDIUM, d_val=0.0),
        ], ignore_index=True)
        seg_summary = pd.DataFrame({
            "channel": [], "segment": [], "n_points": [], "n_valid_draws": [], "skip_reason": [],
        })
        full_df, channel_df, _ = correct_mc_results(draws, seg_summary)
        row = channel_df[channel_df["channel"] == "CADC0888"].iloc[0]
        assert row["level_d_abs_median_weighted"] > 5.0

    def test_tier_report_crosstab_shape(self):
        draws = self._make_draws("CADC0894", 1, MIN_DRAWS_HIGH)
        seg_summary = pd.DataFrame({
            "channel": [], "segment": [], "n_points": [], "n_valid_draws": [], "skip_reason": [],
        })
        _, _, tier_report = correct_mc_results(draws, seg_summary)
        assert list(tier_report.columns) == ["high", "medium", "low", "excluded"]
        assert tier_report.loc["CADC0894", "high"] == 1
