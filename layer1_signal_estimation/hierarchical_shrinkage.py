"""
hierarchical_shrinkage.py
==========================
Hierarchical Bayesian shrinkage of per-channel noise-variance estimates.

Motivation (README, Layer 1, step 3): several OPS-SAT-AD channels have
very few nominal segments to estimate noise from (e.g. `CADC0886` has
only 8 nominal segments, `CADC0890` only 3). A per-channel noise-variance
estimate computed in isolation on that little data is dominated by
sampling noise. This module pools statistical strength across *all*
channels using a random-effects (DerSimonian-Laird-style) shrinkage
estimator on the log-variance scale, so that low-sample channels are
pulled toward the cross-channel consensus while high-sample channels
are left close to their own empirical estimate.

This is the same family of estimator used later, at the causal-analysis
stage (Layer 3), for the between-channel heterogeneity meta-analysis --
here it is applied one level earlier, to noise variances rather than
effect sizes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Random-effects variance (tau^2) cannot go negative; DerSimonian-Laird
# style estimators are truncated at this floor.
MIN_TAU2: float = 0.0


def hierarchical_shrink_variance(
    channel_estimates: Dict[str, Tuple[float, int]]
) -> Dict[str, float]:
    """Shrink each channel's noise-variance estimate toward the
    cross-channel consensus, weighted by how much data supports it.

    Parameters
    ----------
    channel_estimates : dict
        ``{channel: (variance_estimate, n_effective)}``, where
        ``variance_estimate`` is e.g. the ``r`` or ``q`` from
        `noise_estimation.py` for that channel, and ``n_effective`` is
        the number of difference observations it was computed over
        (larger = more trustworthy = less shrinkage).

    Returns
    -------
    dict
        ``{channel: shrunk_variance_estimate}``, same keys as input.

    Notes
    -----
    Operates on the log-variance scale (variances are strictly positive
    and typically span many orders of magnitude across OPS-SAT-AD
    channels -- magnetometer channels' variances are ~1e-13, quantized
    photodiode channels' are ~1e-5 -- so shrinkage on the raw scale would
    be dominated by the largest-variance channels; the log scale treats
    each channel's *relative* precision fairly).
    """
    channels = list(channel_estimates.keys())
    if len(channels) < 2:
        # Nothing to pool across -- return the input unchanged.
        return {c: float(v) for c, (v, _n) in channel_estimates.items()}

    log_v = np.array([np.log(max(channel_estimates[c][0], 1e-12)) for c in channels])
    n_eff = np.array([max(channel_estimates[c][1], 2) for c in channels])

    # Within-channel sampling-error SE of a log-variance estimate,
    # asymptotic approximation: SE[log(var)] ~= sqrt(2 / (n - 1)).
    se = np.sqrt(2.0 / (n_eff - 1))
    weights = 1.0 / (se ** 2)
    global_mean = float(np.sum(weights * log_v) / np.sum(weights))

    # DerSimonian-Laird between-channel variance (tau^2) estimate.
    weighted_ss = float(np.sum(weights * (log_v - global_mean) ** 2))
    df = len(channels) - 1
    denom = float(np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
    tau2 = max(MIN_TAU2, (weighted_ss - df) / denom) if denom > 0 else MIN_TAU2

    shrunk: Dict[str, float] = {}
    for c, lv, s in zip(channels, log_v, se):
        # Shrinkage weight: 1.0 = trust this channel's own estimate fully
        # (large tau^2 relative to its own sampling error), 0.0 = fall
        # back entirely to the cross-channel consensus (small tau^2 /
        # large sampling error, i.e. a low-sample channel).
        denom_c = tau2 + s ** 2
        w_c = tau2 / denom_c if denom_c > 0 else 0.0
        log_v_shrunk = w_c * lv + (1 - w_c) * global_mean
        shrunk[c] = float(np.exp(log_v_shrunk))
    return shrunk


def shrinkage_weights(
    channel_estimates: Dict[str, Tuple[float, int]]
) -> Dict[str, float]:
    """Report the shrinkage weight ``w_c`` applied to each channel by
    `hierarchical_shrink_variance`, without needing to re-derive it from
    the shrunk/raw ratio. Useful for diagnostics: ``w_c`` close to 1
    means "barely shrunk" (high-sample channel), close to 0 means
    "shrunk almost entirely to the cross-channel consensus" (low-sample
    channel).
    """
    channels = list(channel_estimates.keys())
    if len(channels) < 2:
        return {c: 1.0 for c in channels}

    log_v = np.array([np.log(max(channel_estimates[c][0], 1e-12)) for c in channels])
    n_eff = np.array([max(channel_estimates[c][1], 2) for c in channels])
    se = np.sqrt(2.0 / (n_eff - 1))
    weights = 1.0 / (se ** 2)
    global_mean = float(np.sum(weights * log_v) / np.sum(weights))

    weighted_ss = float(np.sum(weights * (log_v - global_mean) ** 2))
    df = len(channels) - 1
    denom = float(np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
    tau2 = max(MIN_TAU2, (weighted_ss - df) / denom) if denom > 0 else MIN_TAU2

    out: Dict[str, float] = {}
    for c, s in zip(channels, se):
        denom_c = tau2 + s ** 2
        out[c] = float(tau2 / denom_c) if denom_c > 0 else 0.0
    return out
