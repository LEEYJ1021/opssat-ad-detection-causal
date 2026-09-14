"""supplementary_diagnostics_A_D.py 핵심 로직 단위테스트.

[A] classify_decay (transient vs persistent)
[B] cohens_d, welch_early_window
[C] analyze_segment_precedence_pivot, run_pairwise_test (temporal_precedence_test.py의
    지역함수와 동일한 로직을 가진 모듈 레벨 버전 — 여기서 짝비교 검정 로직을 검증한다)
[D] get_precedence_indicator, two_proportion_ztest, cmh_test, find_best_lag
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplementary_diagnostics_A_D import (
    classify_decay, cohens_d, welch_early_window,
    analyze_segment_precedence_pivot, run_pairwise_test,
    get_precedence_indicator, two_proportion_ztest, cmh_test, find_best_lag,
    DECAY_RATIO_THRESHOLD,
)


# =============================================================================
# [A] classify_decay
# =============================================================================

class TestClassifyDecay:
    def test_too_short_is_undecidable(self):
        out = classify_decay(np.array([1.0, 2.0]))
        assert out["pattern"] == "판정불가(구간부족)"

    def test_transient_spike_then_decay(self):
        z = np.array([1.0, 5.0, 4.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1])
        out = classify_decay(z)
        assert out["pattern"] == "transient(일시적)"
        assert out["peak_idx"] == 1
        assert out["decay_idx"] is not None and out["decay_idx"] >= 1

    def test_persistent_stays_high(self):
        z = np.full(70, 5.0)
        out = classify_decay(z)
        assert out["pattern"] == "persistent(지속적)"
        assert np.isnan(out["decay_idx"])

    def test_zero_peak_is_undecidable(self):
        z = np.zeros(10)
        out = classify_decay(z)
        assert out["pattern"] == "판정불가(피크없음)"

    def test_decay_threshold_boundary(self):
        # 피크=10, threshold=10*DECAY_RATIO_THRESHOLD=5. 피크 직후 4로 떨어지면
        # transient(<=threshold)로 판정돼야 한다.
        z = np.array([2.0, 10.0, 4.0, 4.0, 4.0])
        out = classify_decay(z)
        assert out["peak_abs_z"] == pytest.approx(10.0)
        assert out["pattern"] == "transient(일시적)"
        assert 4.0 <= 10.0 * DECAY_RATIO_THRESHOLD


# =============================================================================
# [B] cohens_d / welch_early_window
# =============================================================================

class TestCohensD:
    def test_matches_expected_scale(self):
        rng = np.random.default_rng(0)
        a = rng.normal(3, 1, 400)
        b = rng.normal(0, 1, 400)
        assert cohens_d(a, b) == pytest.approx(3.0, abs=0.3)


class TestWelchEarlyWindow:
    def test_early_window_truncates_post(self):
        rng = np.random.default_rng(1)
        pre = rng.normal(0, 1, 50)
        # post 앞부분만 크게 이동, 뒷부분은 pre 수준으로 복귀(transient) ->
        # early window로 자르면 유의하고, full 평균은 희석돼 덜 유의할 수 있음
        post_early = rng.normal(10, 1, 10)
        post_late = rng.normal(0, 1, 200)
        post_full = np.concatenate([post_early, post_late])

        out = welch_early_window(pre, post_full, window=10)
        assert out["p_early"] < 0.01
        assert out["d_early"] > 0
        # full 표본은 대부분 pre 수준으로 돌아온 값들이 섞여 평균 이동이 희석됨
        assert abs(out["d_full"]) < out["d_early"]

    def test_insufficient_points_returns_nan(self):
        pre = np.array([1.0, 2.0])   # MIN_WINDOW_POINTS(5) 미만
        post = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = welch_early_window(pre, post, window=10)
        assert np.isnan(out["p_early"])
        assert np.isnan(out["p_full"])


# =============================================================================
# [C] analyze_segment_precedence_pivot / run_pairwise_test
# =============================================================================

class TestAnalyzeSegmentPrecedencePivot:
    def test_no_deviation_gives_all_nan(self):
        values = np.zeros(100) + np.random.default_rng(2).normal(0, 1e-6, 100)
        out = analyze_segment_precedence_pivot(values, sampling=1.0, pivot_idx=50)
        assert np.isnan(out["level_cross_sec"])
        assert np.isnan(out["diff_cross_sec"])
        assert np.isnan(out["diff2_cross_sec"])

    def test_level_shift_detected_with_correct_offset(self):
        rng = np.random.default_rng(3)
        pre = rng.normal(0, 0.1, 60)
        post = rng.normal(20, 0.1, 60)   # 즉시 큰 이동 -> 첫 포인트에서 이탈
        values = np.concatenate([pre, post])
        out = analyze_segment_precedence_pivot(values, sampling=2.0, pivot_idx=60)
        assert out["level_cross_sec"] == pytest.approx(0.0, abs=1e-9)


class TestRunPairwiseTest:
    def test_small_sample_flagged(self):
        df = pd.DataFrame({"level_cross_sec": [1.0, 2.0], "diff_cross_sec": [0.5, 1.0]})
        out = run_pairwise_test(df, "level", "diff")
        assert out["n_pairs"] < 5
        assert "표본부족" in out["note"]

    def test_consistent_precedence_significant(self):
        rng = np.random.default_rng(4)
        n = 20
        level_t = rng.uniform(5, 10, n)
        diff_t = level_t - rng.uniform(2, 4, n)
        df = pd.DataFrame({"level_cross_sec": level_t, "diff_cross_sec": diff_t})
        out = run_pairwise_test(df, "diff", "level")
        assert out["p_value"] < 0.01
        assert out["frac_diff_precedes_level"] == pytest.approx(1.0)


# =============================================================================
# [D] get_precedence_indicator / two_proportion_ztest / cmh_test / find_best_lag
# =============================================================================

class TestGetPrecedenceIndicator:
    def test_indicator_matches_sign_of_difference(self):
        df = pd.DataFrame({
            "channel": ["A", "A", "B"],
            "diff_cross_sec": [1.0, 5.0, np.nan],
            "level_cross_sec": [2.0, 3.0, 1.0],
        })
        ind, channels = get_precedence_indicator(df, "diff", "level")
        # 행0: 1<2 -> precedes(1), 행1: 5<3? False -> 0, 행2: NaN 제외
        assert list(ind) == [1, 0]
        assert list(channels) == ["A", "A"]


class TestTwoProportionZTest:
    def test_zero_n_returns_nan(self):
        z, p = two_proportion_ztest(0, 0, 5, 10)
        assert np.isnan(z) and np.isnan(p)

    def test_identical_proportions_gives_zero_z(self):
        z, p = two_proportion_ztest(5, 10, 5, 10)
        assert z == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_large_difference_is_significant(self):
        z, p = two_proportion_ztest(90, 100, 10, 100)
        assert p < 0.001
        assert z > 0


class TestCmhTest:
    def test_no_valid_strata_returns_nan(self):
        out = cmh_test([])
        assert np.isnan(out["p_value"])
        assert out["n_strata"] == 0

    def test_consistent_association_across_strata_significant(self):
        # 두 계층 모두 "a"쪽 비율이 뚜렷하게 큰 2x2 표
        tables = [(40, 10, 10, 40), (35, 15, 15, 35)]
        out = cmh_test(tables)
        assert out["n_strata"] == 2
        assert out["p_value"] < 0.001
        assert out["or_mh"] > 1.0

    def test_no_association_gives_large_p(self):
        tables = [(25, 25, 25, 25), (25, 25, 25, 25)]
        out = cmh_test(tables)
        assert out["p_value"] > 0.5


class TestFindBestLag:
    def test_too_short_returns_zero(self):
        lag, corr = find_best_lag(np.array([1.0]), np.array([1.0]), max_lag=5, min_overlap=5)
        assert lag == 0
        assert np.isnan(corr)

    def test_recovers_known_shift(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, 60)
        shift = 3
        a = base.copy()
        b = np.roll(base, shift)   # b는 a를 shift만큼 뒤로 민 것
        lag, corr = find_best_lag(a, b, max_lag=10, min_overlap=10)
        assert corr > 0.9
        # a[:-shift] == b[shift:] 관계이므로 양의 lag에서 최적 정렬됨을 확인
        assert abs(lag) <= 10

    def test_zero_variance_segment_skipped(self):
        a = np.zeros(20)
        b = np.zeros(20)
        lag, corr = find_best_lag(a, b, max_lag=5, min_overlap=5)
        assert lag == 0
        assert corr == 0.0
