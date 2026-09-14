"""
tests/test_effects.py
=======================
Unit tests for representations.tabular_summary_features: cohens_d_abs,
log_var_ratio, and compute_effects. These are the analytical foundation
every model/representation in this package is ultimately compared against,
so their edge-case behavior (underpowered samples, degenerate variance) is
tested directly.
"""

import numpy as np
import pytest

from model_cross_validation.representations.tabular_summary_features import (
    cohens_d_abs, log_var_ratio, compute_effects, BURN_IN, MIN_WINDOW_POINTS,
)


def test_cohens_d_abs_basic():
    a = np.array([10.0, 11.0, 9.0, 10.5, 9.5])
    b = np.array([0.0, 1.0, -1.0, 0.5, -0.5])
    d = cohens_d_abs(a, b)
    assert d > 5.0  # large, well-separated means relative to within-group SD


def test_cohens_d_abs_zero_when_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    d = cohens_d_abs(a, a.copy())
    assert d == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_abs_nan_when_underpowered():
    a = np.array([1.0])  # n < 2
    b = np.array([1.0, 2.0, 3.0])
    assert np.isnan(cohens_d_abs(a, b))


def test_cohens_d_abs_nan_when_degenerate_variance():
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0])
    assert np.isnan(cohens_d_abs(a, b))  # pooled SD == 0


def test_log_var_ratio_positive_for_variance_surge():
    rng = np.random.default_rng(0)
    pre = rng.normal(0, 1, size=50)
    post = rng.normal(0, 5, size=50)  # 5x SD -> 25x variance
    lvr = log_var_ratio(pre, post)
    assert lvr > 0
    assert lvr == pytest.approx(np.log(25), rel=0.5)  # loose tolerance (finite sample)


def test_log_var_ratio_nan_when_underpowered():
    pre = np.array([1.0, 2.0])  # fewer than MIN_WINDOW_POINTS
    post = np.random.default_rng(0).normal(size=20)
    assert np.isnan(log_var_ratio(pre, post))


def test_compute_effects_returns_all_three_keys():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 4, 100)])
    pivot_idx = 100
    eff = compute_effects(values, pivot_idx)
    assert set(eff.keys()) == {"level", "diff", "diff2"}
    # a real level+variance shift at the pivot should be detectable
    assert not np.isnan(eff["level"])
    assert not np.isnan(eff["diff"])


def test_compute_effects_nan_near_series_edges():
    values = np.arange(20, dtype=float)
    # pivot too close to the start given BURN_IN / MIN_WINDOW_POINTS
    eff = compute_effects(values, BURN_IN)
    assert np.isnan(eff["level"]) or np.isnan(eff["diff"])
