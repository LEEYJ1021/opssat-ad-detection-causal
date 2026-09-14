"""
tests/test_shap_normalize.py
===============================
Regression tests for [PATCH 1]: attribution.shap_permutation._normalize_shap_values
must correctly absorb every shap_values() return shape observed in practice
(legacy list, modern 3D ndarray, already-2D ndarray, 1D edge case) and
return None (triggering the permutation-importance fallback) when the shape
cannot be reconciled with n_features.

This directly guards against the regression that originally crashed
GaussianNB with "TypeError: only 0-dimensional arrays can be converted to
Python scalars" when a 3D ndarray was returned unhandled.
"""

import numpy as np

from model_cross_validation.attribution.shap_permutation import _normalize_shap_values


N_SAMPLES, N_FEATURES = 20, 3


def test_normalize_legacy_list_format():
    class0 = np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES))
    class1 = np.random.default_rng(1).normal(size=(N_SAMPLES, N_FEATURES))
    sv = [class0, class1]
    out = _normalize_shap_values(sv, N_FEATURES)
    assert out is not None
    assert out.shape == (N_SAMPLES, N_FEATURES)
    np.testing.assert_array_equal(out, class1)  # positive-class array selected


def test_normalize_legacy_single_element_list():
    class0 = np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES))
    out = _normalize_shap_values([class0], N_FEATURES)
    assert out is not None
    assert out.shape == (N_SAMPLES, N_FEATURES)


def test_normalize_modern_3d_ndarray():
    """The exact case that crashed GaussianNB pre-patch: (n, features, classes)."""
    sv = np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES, 2))
    out = _normalize_shap_values(sv, N_FEATURES)
    assert out is not None
    assert out.shape == (N_SAMPLES, N_FEATURES)
    np.testing.assert_array_equal(out, sv[:, :, 1])  # positive class (last axis, index 1)


def test_normalize_modern_3d_ndarray_single_class():
    sv = np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES, 1))
    out = _normalize_shap_values(sv, N_FEATURES)
    assert out is not None
    assert out.shape == (N_SAMPLES, N_FEATURES)


def test_normalize_already_2d_passthrough():
    sv = np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES))
    out = _normalize_shap_values(sv, N_FEATURES)
    np.testing.assert_array_equal(out, sv)


def test_normalize_1d_edge_case_reshaped():
    sv = np.random.default_rng(0).normal(size=(N_FEATURES,))
    out = _normalize_shap_values(sv, N_FEATURES)
    assert out is not None
    assert out.shape == (1, N_FEATURES)


def test_normalize_returns_none_on_feature_mismatch():
    sv = np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES + 1))  # wrong feature count
    out = _normalize_shap_values(sv, N_FEATURES)
    assert out is None
