import numpy as np
import pytest

from layer1_signal_estimation.hierarchical_shrinkage import (
    hierarchical_shrink_variance,
    shrinkage_weights,
)


def test_single_channel_passthrough():
    estimates = {"A": (1e-3, 100)}
    shrunk = hierarchical_shrink_variance(estimates)
    assert shrunk == {"A": 1e-3}


def test_low_sample_channel_pulled_toward_consensus():
    # Five channels agree tightly around 1e-3 with lots of data; one
    # channel has very little data and a wildly different raw estimate.
    estimates = {
        "A": (1.0e-3, 5000),
        "B": (1.1e-3, 5000),
        "C": (0.9e-3, 5000),
        "D": (1.05e-3, 5000),
        "LOWSAMPLE": (50.0, 3),  # extreme outlier, almost no data
    }
    shrunk = hierarchical_shrink_variance(estimates)
    # The low-sample channel's shrunk estimate should move drastically
    # away from its raw 50.0 toward the ~1e-3 consensus.
    assert shrunk["LOWSAMPLE"] < 1.0
    # High-sample channels should stay close to their own raw estimates.
    assert shrunk["A"] == pytest.approx(1.0e-3, rel=0.5)


def test_shrinkage_weights_between_zero_and_one():
    estimates = {
        "A": (1.0e-3, 5000),
        "B": (1.1e-3, 5000),
        "LOWSAMPLE": (50.0, 3),
    }
    weights = shrinkage_weights(estimates)
    for c, w in weights.items():
        assert 0.0 <= w <= 1.0
    assert weights["LOWSAMPLE"] <= weights["A"]


def test_shrinkage_weights_single_channel_is_one():
    weights = shrinkage_weights({"A": (1e-3, 10)})
    assert weights == {"A": 1.0}


def test_homogeneous_channels_shrink_close_to_global_mean():
    # When all channels agree closely, tau^2 should be ~0, so every
    # channel's shrunk estimate should end up close to the (weighted)
    # global mean regardless of sample size.
    estimates = {c: (1.0e-3 * (1 + 0.01 * i), 1000) for i, c in enumerate("ABCDE")}
    shrunk = hierarchical_shrink_variance(estimates)
    values = np.array(list(shrunk.values()))
    assert values.std() < 1e-4


def test_shrunk_values_are_all_positive():
    estimates = {
        "A": (1e-9, 10000),
        "B": (1e-3, 2),
        "C": (1e2, 50),
    }
    shrunk = hierarchical_shrink_variance(estimates)
    for v in shrunk.values():
        assert v > 0
