import numpy as np
import pytest

from layer1_signal_estimation.noise_estimation import (
    MIN_VARIANCE_FLOOR,
    compare_estimators,
    estimate_channel_noise,
    estimate_noise_mixture,
    estimate_noise_simple,
    mad_variance,
    mixture_variance,
)


def test_mad_variance_matches_normal_std_roughly():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, size=5000)
    var = mad_variance(x)
    # MAD-based sigma should recover the true std (2.0) reasonably closely
    # for a large Gaussian sample.
    assert np.sqrt(var) == pytest.approx(2.0, rel=0.1)


def test_mad_variance_too_few_points():
    assert np.isnan(mad_variance(np.array([1.0])))


def test_mixture_variance_all_zero():
    assert mixture_variance(np.zeros(10)) == 0.0


def test_mixture_variance_zero_inflated_recovers_jump_scale():
    rng = np.random.default_rng(1)
    n = 5000
    p_true = 0.3
    is_jump = rng.random(n) < p_true
    jumps = np.where(is_jump, rng.normal(0, 5.0, n), 0.0)
    var = mixture_variance(jumps)
    # E[Var(diff)] = p * Var(jump | jump) = 0.3 * 25 = 7.5
    assert var == pytest.approx(7.5, rel=0.25)


def test_mixture_variance_too_few_points():
    assert np.isnan(mixture_variance(np.array([1.0])))


def test_estimate_noise_simple_matches_manual_calc():
    diffs = np.array([1.0, -1.0, 2.0, -2.0, 1.0, -1.0])
    r, q = estimate_noise_simple(diffs)
    expected_r = max(np.var(diffs) / 2.0, MIN_VARIANCE_FLOOR)
    assert r == pytest.approx(expected_r)
    assert q >= MIN_VARIANCE_FLOOR


def test_estimate_noise_simple_floors_zero_variance():
    r, q = estimate_noise_simple(np.zeros(20))
    assert r == MIN_VARIANCE_FLOOR
    assert q == MIN_VARIANCE_FLOOR


def test_estimate_noise_mixture_floors_zero_variance():
    r, q = estimate_noise_mixture(np.zeros(20))
    assert r == MIN_VARIANCE_FLOOR
    assert q == MIN_VARIANCE_FLOOR


def test_estimate_channel_noise_simple_vs_mixture_differ_on_zero_inflated_data():
    rng = np.random.default_rng(2)
    # Zero-inflated: most segments are flat with occasional jumps.
    segments = []
    for _ in range(20):
        n = 40
        is_jump = rng.random(n - 1) < 0.15
        jumps = np.where(is_jump, rng.normal(0, 3.0, n - 1), 0.0)
        segments.append(np.concatenate([[0.0], np.cumsum(jumps)]))

    simple = estimate_channel_noise("ZI", segments, estimator="simple")
    mixture = estimate_channel_noise("ZI", segments, estimator="mixture")

    assert simple.estimator == "simple"
    assert mixture.estimator == "mixture"
    # The two estimators need not be numerically identical on zero-inflated data.
    assert simple.r != mixture.r or simple.q != mixture.q


def test_estimate_channel_noise_empty_segments():
    est = estimate_channel_noise("EMPTY", [])
    assert est.n_diff == 0
    assert est.r == MIN_VARIANCE_FLOOR
    assert est.q == MIN_VARIANCE_FLOOR


def test_estimate_channel_noise_unknown_estimator_raises():
    with pytest.raises(ValueError):
        estimate_channel_noise("X", [np.array([1.0, 2.0, 3.0])], estimator="bogus")  # type: ignore[arg-type]


def test_compare_estimators_shape():
    rng = np.random.default_rng(3)
    segments = [np.cumsum(rng.normal(0, 1, 40)) for _ in range(10)]
    result = compare_estimators("CMP", segments)
    assert result["channel"] == "CMP"
    assert "simple" in result and "mixture" in result
    assert np.isfinite(result["r_ratio_mixture_to_simple"])
