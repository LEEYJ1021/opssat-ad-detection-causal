import numpy as np
import pytest

from layer1_signal_estimation.kalman_filter import LocalLinearTrendKF, run_kalman_for_segment


def test_filter_output_shapes():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 30))
    kf = LocalLinearTrendKF(q=0.1, r_nominal=1.0)
    result = kf.run(y)
    assert result.innovations.shape == (30,)
    assert result.innovation_vars.shape == (30,)
    assert result.levels.shape == (30,)


def test_filter_denoises_constant_signal():
    rng = np.random.default_rng(1)
    true_level = 5.0
    y = true_level + rng.normal(0, 0.5, 200)
    kf = LocalLinearTrendKF(q=1e-6, r_nominal=0.25)
    result = kf.run(y)
    # After the filter settles, the level prediction should be much
    # closer to the true constant than the raw noisy observations are.
    raw_err = np.abs(y[50:] - true_level).mean()
    filt_err = np.abs(result.levels[50:] - true_level).mean()
    assert filt_err < raw_err


def test_filter_tracks_a_ramp():
    n = 100
    slope = 0.2
    y = np.arange(n) * slope
    kf = LocalLinearTrendKF(q=1e-3, r_nominal=1e-6, x0=np.array([0.0, slope]))
    result = kf.run(y)
    # With near-zero measurement noise and the correct initial slope, the
    # filter should track the ramp closely after the first couple steps.
    assert np.abs(result.levels[10:] - y[10:]).max() < 1.0


def test_negative_q_raises():
    with pytest.raises(ValueError):
        LocalLinearTrendKF(q=-1.0, r_nominal=1.0)


def test_negative_r_raises():
    with pytest.raises(ValueError):
        LocalLinearTrendKF(q=1.0, r_nominal=-1.0)


def test_run_kalman_for_segment_quantization_floor_increases_effective_noise():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(0, 0.01, 50)) + 1.0

    no_quant = LocalLinearTrendKF(q=1e-4, r_nominal=1e-4, quantization_floor=0.0)
    with_quant = LocalLinearTrendKF(q=1e-4, r_nominal=1e-4, quantization_floor=1e-3)

    assert with_quant.r_eff > no_quant.r_eff

    result = run_kalman_for_segment(y, r=1e-4, q=1e-4, quantization_step=0.1)
    # step**2/12 = 0.1**2/12 ~= 8.33e-4 added to r_eff -> innovation
    # variances should reflect the inflated effective measurement noise.
    assert result.innovation_vars[-1] > 1e-4


def test_run_kalman_for_segment_no_quantization_step():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 1, 20))
    result = run_kalman_for_segment(y, r=1.0, q=0.1, quantization_step=None)
    assert result.levels.shape == (20,)
