"""_shared.py 핵심 로직 단위테스트 — 채널 프로파일링, 노이즈 추정, KF, 계층적
축소(shrinkage), BOCPD."""

from __future__ import annotations

import numpy as np
import pytest

from _shared import (
    mad_variance, mixture_variance, detect_quantization_step, profile_channel,
    LocalLinearTrendKF, hierarchical_shrink_variance, BOCPD,
    QUANTIZATION_FRAC_ZERO_THRESHOLD, FLOAT_NOISE_ABS_DIFF_THRESHOLD,
)


# =============================================================================
# mad_variance / mixture_variance / detect_quantization_step
# =============================================================================

class TestMadVariance:
    def test_too_few_points_returns_nan(self):
        assert np.isnan(mad_variance(np.array([1.0])))
        assert np.isnan(mad_variance(np.array([])))

    def test_matches_known_gaussian_scale(self):
        rng = np.random.default_rng(0)
        true_std = 2.0
        x = rng.normal(0, true_std, 20000)
        var_est = mad_variance(x)
        # MAD -> robust std estimator, 큰 표본이면 참 분산에 근접해야 함
        assert var_est == pytest.approx(true_std ** 2, rel=0.15)

    def test_constant_input_gives_zero_variance(self):
        x = np.full(10, 3.5)
        assert mad_variance(x) == pytest.approx(0.0)


class TestMixtureVariance:
    def test_too_few_points_returns_nan(self):
        assert np.isnan(mixture_variance(np.array([1.0])))

    def test_all_zero_diffs_gives_zero_variance(self):
        assert mixture_variance(np.zeros(50)) == 0.0

    def test_mixture_scales_with_jump_probability(self):
        rng = np.random.default_rng(1)
        # 잦은 점프(p=0.8)와 드문 점프(p=0.2) 두 경우를 비교 — 점프 크기 자체에
        # 약간의 변동을 줘서 MAD(jump_var)가 0이 되지 않게 한다. 같은 점프
        # 크기 분포라면 p가 클수록 mixture_variance(=p*jump_var)도 커야 한다.
        jump_sizes = lambda n: rng.normal(5.0, 0.5, n)  # noqa: E731
        mask_common = rng.random(5000) < 0.8
        jumps_common = np.where(mask_common, jump_sizes(5000), 0.0)
        mask_rare = rng.random(5000) < 0.2
        jumps_rare = np.where(mask_rare, jump_sizes(5000), 0.0)
        assert mixture_variance(jumps_common) > mixture_variance(jumps_rare)


class TestDetectQuantizationStep:
    def test_empty_returns_nan(self):
        assert np.isnan(detect_quantization_step(np.array([])))

    def test_recovers_uniform_step(self):
        # 0.05 배수로만 이루어진 절대차분 -> 25th percentile도 0.05의 배수
        steps = np.array([0.05, 0.05, 0.05, 0.10, 0.05, 0.15])
        q = detect_quantization_step(steps)
        assert q == pytest.approx(np.percentile(steps, 25))


# =============================================================================
# profile_channel — quantized / float_noise_suspect / continuous 분류
# =============================================================================

class TestProfileChannel:
    def test_no_segments_returns_unknown(self):
        prof = profile_channel("CH_X", [])
        assert prof.channel_type == "unknown"
        assert prof.n_nominal_points == 0

    def test_detects_quantized_channel(self):
        # 값이 정수 격자 위에서만 움직이도록 구성 -> frac_zero_diff가 높아야 함
        rng = np.random.default_rng(2)
        base = rng.integers(0, 5, size=500).astype(float)
        segs = [base]
        prof = profile_channel("CADC_Q", segs)
        assert prof.frac_zero_diff >= QUANTIZATION_FRAC_ZERO_THRESHOLD
        assert prof.channel_type == "quantized"
        assert prof.quantization_step is not None and np.isfinite(prof.quantization_step)

    def test_detects_float_noise_suspect_channel(self):
        # 항상 0이 아닌 아주 작은 차분(<1e-6)만 나오는 연속형 채널
        rng = np.random.default_rng(3)
        base = np.cumsum(rng.normal(0, 1e-8, 500))
        segs = [base]
        prof = profile_channel("CADC_F", segs)
        assert prof.channel_type == "float_noise_suspect"

    def test_detects_continuous_channel(self):
        rng = np.random.default_rng(4)
        base = np.cumsum(rng.normal(0, 1.0, 500))
        segs = [base]
        prof = profile_channel("CADC_C", segs)
        assert prof.channel_type == "continuous"
        assert prof.r_robust > 0
        assert prof.q_robust > 0


# =============================================================================
# LocalLinearTrendKF
# =============================================================================

class TestLocalLinearTrendKF:
    def test_output_shapes(self):
        kf = LocalLinearTrendKF(q=1e-4, r_nominal=0.1)
        y = np.linspace(0, 10, 50)
        innovations, innovation_vars, _ = kf.run(y)
        assert innovations.shape == (50,)
        assert innovation_vars.shape == (50,)
        assert np.all(innovation_vars > 0)

    def test_tracks_linear_trend_with_small_late_innovations(self):
        # 완전한 직선 추세는 로컬-선형 KF가 몇 스텝 뒤엔 거의 완벽히 추적해야
        # 하므로, 뒷부분 혁신값(innovation)의 절댓값 평균이 앞부분보다 작아야 함
        kf = LocalLinearTrendKF(q=1e-6, r_nominal=1e-4)
        y = 2.0 * np.arange(200) + 1.0
        innovations, _, _ = kf.run(y)
        early = np.mean(np.abs(innovations[:5]))
        late = np.mean(np.abs(innovations[-20:]))
        assert late < early

    def test_r_eff_includes_quantization_floor(self):
        kf_no_floor = LocalLinearTrendKF(q=1e-4, r_nominal=0.1, quantization_floor=0.0)
        kf_with_floor = LocalLinearTrendKF(q=1e-4, r_nominal=0.1, quantization_floor=0.5)
        assert kf_with_floor.r_eff > kf_no_floor.r_eff
        assert kf_with_floor.r_eff == pytest.approx(kf_no_floor.r_eff + 0.5)


# =============================================================================
# hierarchical_shrink_variance
# =============================================================================

class TestHierarchicalShrinkVariance:
    def test_shrinks_toward_global_mean(self):
        # 채널 A는 표본이 매우 적어(n_eff=2) 추정이 불안정 -> 강하게 축소돼야 함
        # 채널 B는 표본이 매우 많아(n_eff=10000) 안정적 -> 원래 값 근처 유지
        estimates = {
            "A": (100.0, 2),      # 불안정한 이상치 추정
            "B": (1.0, 10000),    # 안정적인 추정
            "C": (1.2, 10000),
            "D": (0.9, 10000),
        }
        shrunk = hierarchical_shrink_variance(estimates)
        assert set(shrunk.keys()) == set(estimates.keys())
        # A는 원래 값(100)보다 훨씬 작아져야 하고, 다른 안정적인 채널들 쪽으로 당겨진다
        assert shrunk["A"] < 100.0
        # B는 표본이 많아 거의 원래 값 근처를 유지해야 한다
        assert shrunk["B"] == pytest.approx(1.0, rel=0.5)

    def test_all_equal_inputs_stay_equal(self):
        estimates = {ch: (2.0, 500) for ch in ["A", "B", "C"]}
        shrunk = hierarchical_shrink_variance(estimates)
        for v in shrunk.values():
            assert v == pytest.approx(2.0, rel=1e-6)


# =============================================================================
# BOCPD
# =============================================================================

class TestBOCPD:
    def test_run_length_posterior_sums_to_one_each_step(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 60)
        bocpd = BOCPD(hazard_lambda=250.0, forgetting=True)
        posteriors = bocpd.run(x)
        assert len(posteriors) == len(x)
        for p in posteriors:
            assert p.sum() == pytest.approx(1.0, abs=1e-6)
            assert np.all(p >= 0)

    def test_detects_an_obvious_changepoint(self):
        # 평균 0, std 1인 구간 뒤에 평균 20으로 뚝 떨어지는 명백한 변화점 ->
        # 변화 직후 run-length posterior는 '짧은 run-length(=최근 리셋)'에
        # 확률질량이 쏠려야 한다.
        rng = np.random.default_rng(6)
        pre = rng.normal(0, 1, 80)
        post = rng.normal(20, 1, 20)
        x = np.concatenate([pre, post])
        bocpd = BOCPD(hazard_lambda=250.0, forgetting=True)
        posteriors = bocpd.run(x)
        last = posteriors[-1]
        # run-length 0~3 (변화 이후 아주 짧은 run) 쪽 확률질량이
        # run-length가 훨씬 긴(변화 전부터 이어졌다고 보는) 쪽보다 커야 한다.
        short_run_mass = last[:4].sum()
        long_run_mass = last[40:60].sum() if len(last) > 60 else last[40:].sum()
        assert short_run_mass > long_run_mass
