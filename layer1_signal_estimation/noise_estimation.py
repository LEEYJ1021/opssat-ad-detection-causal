"""
noise_estimation.py
====================
Per-channel measurement-noise (``r``) and process-noise (``q``) variance
estimation for the Layer-1 Kalman filter, plus the estimator-choice
ablation documented in the README:

    "Noise-estimator comparison: a simple var(diff)/2 estimator was
    compared against a Gaussian-mixture noise estimator. The simple
    estimator was adopted -- the mixture model did not improve
    calibration and added a source of instability for low-sample
    channels."

Both estimators are kept in this module (`estimate_noise_simple` and
`estimate_noise_mixture`) so the ablation in
``docs/dev-log/step1_signal_estimation.md`` remains fully reproducible,
but the pipeline default (`ESTIMATOR = "simple"`, see `run.py`) is locked
to the simple estimator per that decision.

Why the mixture estimator exists at all: several channels are strongly
zero-inflated (quantized photodiode channels can have >40% exactly-zero
first differences). Applying a robust MAD estimator naively to the *full*
diff distribution in that regime pushes the median to zero and collapses
the variance estimate to zero -- a real failure observed during
development. `mixture_variance` avoids this by explicitly separating
"probability a jump occurs" from "size of the jump given it occurs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EstimatorName = Literal["simple", "mixture"]

# Locked pipeline default -- see module docstring / README Layer-1 §2.
LOCKED_ESTIMATOR: EstimatorName = "simple"

# Numerical floor so downstream Kalman-filter division-by-variance never
# blows up on a channel with (near-)zero observed noise.
MIN_VARIANCE_FLOOR: float = 1e-14


@dataclass
class NoiseEstimate:
    """Measurement- and process-noise variance estimate for one channel.

    Attributes
    ----------
    channel : str
    estimator : str
        Which estimator produced this result (``"simple"`` or ``"mixture"``).
    r : float
        Measurement-noise variance, derived from ``Var(diff) / 2`` (the
        classical estimator for i.i.d. additive noise under a random-walk
        assumption: ``Var(y_t - y_{t-1}) = 2 * Var(noise)``).
    q : float
        Process-noise variance, derived analogously from the second
        difference: ``Var(diff(diff)) / 6``.
    n_diff : int
        Number of first-difference observations the estimate was computed
        over.
    """

    channel: str
    estimator: str
    r: float
    q: float
    n_diff: int

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "estimator": self.estimator,
            "r": self.r,
            "q": self.q,
            "n_diff": self.n_diff,
        }


def mad_variance(x: np.ndarray) -> float:
    """Median-Absolute-Deviation variance estimate (robust to outliers).

    Converts MAD to a standard-deviation estimate via the usual 1.4826
    consistency scale factor (exact for a Gaussian), then squares it.

    Note: if more than half of ``x`` shares the same value (e.g. many
    exact zeros), this returns 0 by construction -- callers with
    zero-inflated data should pre-filter zeros out first (see
    `mixture_variance`, which does this).
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return float(sigma ** 2)


def mixture_variance(diffs_all: np.ndarray) -> float:
    """Zero-inflated mixture variance estimator.

    Models the difference series as a mixture: with probability ``p`` a
    "jump" occurs (drawn from some jump-size distribution), and with
    probability ``1 - p`` the difference is exactly zero. Under this
    model, ``Var(diff) = p * Var(jump | jump occurred)``, which is what
    this function computes (using a robust MAD estimate of the jump-size
    variance so a handful of extreme jumps don't dominate).
    """
    diffs_all = np.asarray(diffs_all, dtype=float)
    if len(diffs_all) < 2:
        return float("nan")
    p = float(np.mean(diffs_all != 0))
    if p == 0:
        return 0.0
    nonzero = diffs_all[diffs_all != 0]
    if len(nonzero) >= 2:
        jump_var = mad_variance(nonzero)
    else:
        jump_var = float(nonzero[0] ** 2)
    return p * jump_var


def _diff_based_variance(x: np.ndarray, estimator: EstimatorName) -> float:
    if estimator == "simple":
        return float(np.var(x))
    if estimator == "mixture":
        return mixture_variance(x)
    raise ValueError(f"Unknown estimator: {estimator!r}")


def estimate_noise_simple(diffs_all: np.ndarray) -> tuple[float, float]:
    """Locked pipeline estimator: plain ``Var(diff) / 2`` and
    ``Var(diff-of-diff) / 6``, floored at `MIN_VARIANCE_FLOOR`.

    Returns
    -------
    (r, q) : tuple of float
    """
    diffs_all = np.asarray(diffs_all, dtype=float)
    r = max(float(np.var(diffs_all)) / 2.0, MIN_VARIANCE_FLOOR)
    d2 = np.diff(diffs_all)
    q = max(float(np.var(d2)) / 6.0, MIN_VARIANCE_FLOOR) if len(d2) > 1 else MIN_VARIANCE_FLOOR
    return r, q


def estimate_noise_mixture(diffs_all: np.ndarray) -> tuple[float, float]:
    """Ablation-only estimator: same ``/2`` and ``/6`` scaling as
    `estimate_noise_simple`, but using the zero-inflated
    `mixture_variance` in place of the plain sample variance.

    Kept for reproducing the README's noise-estimator ablation; not used
    by the locked pipeline (see `LOCKED_ESTIMATOR`).
    """
    diffs_all = np.asarray(diffs_all, dtype=float)
    r = max(mixture_variance(diffs_all) / 2.0, MIN_VARIANCE_FLOOR)
    d2 = np.diff(diffs_all)
    q = max(mixture_variance(d2) / 6.0, MIN_VARIANCE_FLOOR) if len(d2) > 1 else MIN_VARIANCE_FLOOR
    return r, q


def estimate_channel_noise(
    channel: str,
    nominal_values_by_segment,
    estimator: EstimatorName = LOCKED_ESTIMATOR,
) -> NoiseEstimate:
    """Compute a `NoiseEstimate` for one channel, pooling first
    differences across all of its nominal segments.

    Parameters
    ----------
    channel : str
    nominal_values_by_segment : sequence of 1D arrays
        Raw values per nominal segment (see `channel_classification`).
    estimator : {"simple", "mixture"}
        Defaults to the locked pipeline choice, `LOCKED_ESTIMATOR`.
    """
    diffs_all = [
        np.diff(np.asarray(v, dtype=float))
        for v in nominal_values_by_segment
        if len(v) > 1
    ]
    if not diffs_all:
        return NoiseEstimate(channel=channel, estimator=estimator, r=MIN_VARIANCE_FLOOR,
                              q=MIN_VARIANCE_FLOOR, n_diff=0)
    diffs = np.concatenate(diffs_all)

    if estimator == "simple":
        r, q = estimate_noise_simple(diffs)
    elif estimator == "mixture":
        r, q = estimate_noise_mixture(diffs)
    else:
        raise ValueError(f"Unknown estimator: {estimator!r}")

    return NoiseEstimate(channel=channel, estimator=estimator, r=r, q=q, n_diff=len(diffs))


def compare_estimators(channel: str, nominal_values_by_segment) -> dict:
    """Run both estimators on the same data and report both, for the
    calibration ablation in the README ("Noise-estimator comparison").

    Returns
    -------
    dict with keys ``channel``, ``simple`` (NoiseEstimate.to_dict()),
    ``mixture`` (NoiseEstimate.to_dict()), and ``r_ratio_mixture_to_simple``
    (a quick eyeball metric for how much the two estimators disagree).
    """
    simple = estimate_channel_noise(channel, nominal_values_by_segment, estimator="simple")
    mixture = estimate_channel_noise(channel, nominal_values_by_segment, estimator="mixture")
    ratio = (mixture.r / simple.r) if simple.r > 0 else float("nan")
    return {
        "channel": channel,
        "simple": simple.to_dict(),
        "mixture": mixture.to_dict(),
        "r_ratio_mixture_to_simple": ratio,
    }
