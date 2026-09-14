import numpy as np
import pytest

from layer1_signal_estimation.channel_classification import (
    QUANTIZATION_FRAC_ZERO_THRESHOLD,
    classify_channel,
    detect_quantization_step,
)


def _quantized_segments(n_segments=10, seg_len=50, step=0.01, frac_zero=0.4, seed=0):
    rng = np.random.default_rng(seed)
    segments = []
    for _ in range(n_segments):
        n_steps = seg_len - 1
        is_jump = rng.random(n_steps) > frac_zero
        jumps = np.where(is_jump, rng.choice([-1, 1], size=n_steps) * step, 0.0)
        segments.append(np.concatenate([[1.0], 1.0 + np.cumsum(jumps)]))
    return segments


def _float_noise_segments(n_segments=10, seg_len=50, base=1e-7, noise=1e-11, seed=0):
    rng = np.random.default_rng(seed)
    return [base + np.cumsum(rng.normal(0, noise, seg_len)) for _ in range(n_segments)]


def _continuous_segments(n_segments=10, seg_len=50, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(0, noise, seg_len)) for _ in range(n_segments)]


def test_quantized_channel_detected():
    profile = classify_channel("TEST_Q", _quantized_segments())
    assert profile.channel_type == "quantized"
    assert profile.frac_zero_diff >= QUANTIZATION_FRAC_ZERO_THRESHOLD
    assert np.isfinite(profile.quantization_step)


def test_float_noise_channel_detected():
    profile = classify_channel("TEST_F", _float_noise_segments())
    assert profile.channel_type == "float_noise_suspect"
    assert np.isnan(profile.quantization_step)


def test_continuous_channel_detected():
    profile = classify_channel("TEST_C", _continuous_segments())
    assert profile.channel_type == "continuous"
    assert np.isnan(profile.quantization_step)


def test_empty_input_is_unknown():
    profile = classify_channel("TEST_EMPTY", [])
    assert profile.channel_type == "unknown"
    assert profile.n_nominal_points == 0


def test_single_point_segments_are_unknown():
    # Segments with a single sample produce no diffs at all.
    profile = classify_channel("TEST_SINGLE", [np.array([1.0]), np.array([2.0])])
    assert profile.channel_type == "unknown"
    assert profile.n_nominal_points == 2


def test_detect_quantization_step_empty():
    assert np.isnan(detect_quantization_step(np.array([])))


def test_detect_quantization_step_basic():
    step = detect_quantization_step(np.array([0.01, 0.01, 0.02, 0.01]))
    assert step == pytest.approx(0.01, abs=1e-6)


def test_invalid_channel_type_raises():
    from layer1_signal_estimation.channel_classification import ChannelProfile

    with pytest.raises(ValueError):
        ChannelProfile(
            channel="BAD", channel_type="not_a_type", frac_zero_diff=0.0,
            min_nonzero_abs_diff=0.0, quantization_step=float("nan"), n_nominal_points=0,
        )


def test_to_dict_roundtrip():
    profile = classify_channel("TEST_Q", _quantized_segments())
    d = profile.to_dict()
    assert d["channel"] == "TEST_Q"
    assert d["channel_type"] == "quantized"
    assert set(d.keys()) == {
        "channel", "channel_type", "frac_zero_diff",
        "min_nonzero_abs_diff", "quantization_step", "n_nominal_points",
    }
