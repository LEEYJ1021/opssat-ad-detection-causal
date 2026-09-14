"""
tests/test_sequence_window.py
================================
Unit tests for representations.raw_sequence_features.build_3channel_window:
shape correctness, None-on-too-short-segment behavior, and per-channel
z-normalization (mean ~0, SD ~1 within each channel of the window).
"""

import numpy as np
import pytest

from model_cross_validation.representations.raw_sequence_features import (
    build_3channel_window, znorm, SEQ_WINDOW_HALF,
)


def test_window_shape_is_3_by_2half():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    win = build_3channel_window(values, pivot_idx=100, half=SEQ_WINDOW_HALF)
    assert win is not None
    assert win.shape == (3, 2 * SEQ_WINDOW_HALF)
    assert win.dtype == np.float32


def test_window_none_when_segment_too_short():
    values = np.arange(10, dtype=float)  # far too short for half=15
    win = build_3channel_window(values, pivot_idx=5, half=SEQ_WINDOW_HALF)
    assert win is None


def test_window_none_when_pivot_near_left_edge():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    # pivot_idx - half - 2 < 0
    win = build_3channel_window(values, pivot_idx=SEQ_WINDOW_HALF - 1, half=SEQ_WINDOW_HALF)
    assert win is None


def test_window_channels_are_per_window_znormalized():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=100, scale=20, size=300)  # far from zero mean/unit SD
    win = build_3channel_window(values, pivot_idx=150, half=SEQ_WINDOW_HALF)
    assert win is not None
    for ch in range(3):
        assert win[ch].mean() == pytest.approx(0.0, abs=1e-4)
        assert win[ch].std() == pytest.approx(1.0, abs=1e-3)


def test_znorm_handles_constant_input_without_nan():
    x = np.ones(10)
    z = znorm(x)
    assert np.all(np.isfinite(z))  # guarded by the 1e-8 epsilon in the denominator
