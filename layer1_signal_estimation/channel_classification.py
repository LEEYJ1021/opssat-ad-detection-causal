"""
channel_classification.py
==========================
Automatic per-channel regime classification: `quantized`, `float_noise_suspect`,
or `continuous`.

Rationale (README, Layer 1, step 1): telemetry channels in OPS-SAT-AD fall
into two very different noise regimes that must be handled differently by
the noise estimator and the Kalman filter:

  * **Photodiode-like ("quantized")**: a large fraction of consecutive
    first differences are exactly zero, and the non-zero differences
    cluster on a discrete step size (ADC quantization).
  * **Magnetometer-like ("float_noise_suspect")**: differences are almost
    never exactly zero, and the smallest non-zero |diff| is many orders
    of magnitude below any physically meaningful signal change --
    symptomatic of floating-point representation noise rather than a
    true discrete step.
  * **Continuous**: neither of the above -- differences are non-zero and
    of a normal, non-quantized magnitude.

The classification thresholds below were fixed from empirical
``diff``-distribution diagnostics run across all 9 channels (see
``docs/dev-log/step1_signal_estimation.md``) and are then locked for the
whole pipeline -- they are not re-tuned per channel or per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

# ---------------------------------------------------------------------
# Locked classification thresholds (empirically fixed; see module
# docstring). Do not silently change these without re-running the
# threshold-diagnostics notebook and updating docs/dev-log/.
# ---------------------------------------------------------------------
QUANTIZATION_FRAC_ZERO_THRESHOLD: float = 0.10
FLOAT_NOISE_ABS_DIFF_THRESHOLD: float = 1e-6

VALID_CHANNEL_TYPES = ("quantized", "float_noise_suspect", "continuous", "unknown")


@dataclass
class ChannelProfile:
    """Classification result for a single channel.

    Attributes
    ----------
    channel : str
        Channel code, e.g. ``"CADC0872"``.
    channel_type : str
        One of ``VALID_CHANNEL_TYPES``.
    frac_zero_diff : float
        Fraction of first-differences (pooled across all nominal, i.e.
        non-anomalous, segments of this channel) that are exactly zero.
    min_nonzero_abs_diff : float
        Smallest non-zero |diff| observed. Used to flag float-noise
        channels whose "quantization" is actually representation noise.
    quantization_step : float
        Estimated ADC step size (25th percentile of non-zero |diff|),
        ``NaN`` if the channel was not classified as ``quantized``.
    n_nominal_points : int
        Total number of nominal (non-anomalous) sample points the
        classification was computed over, pooled across segments.
    """

    channel: str
    channel_type: str
    frac_zero_diff: float
    min_nonzero_abs_diff: float
    quantization_step: float
    n_nominal_points: int

    def __post_init__(self) -> None:
        if self.channel_type not in VALID_CHANNEL_TYPES:
            raise ValueError(
                f"channel_type={self.channel_type!r} not in {VALID_CHANNEL_TYPES}"
            )

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "channel_type": self.channel_type,
            "frac_zero_diff": self.frac_zero_diff,
            "min_nonzero_abs_diff": self.min_nonzero_abs_diff,
            "quantization_step": self.quantization_step,
            "n_nominal_points": self.n_nominal_points,
        }


def detect_quantization_step(nonzero_abs_diffs: np.ndarray) -> float:
    """Estimate the ADC quantization step as the 25th percentile of the
    non-zero |diff| distribution.

    The 25th percentile (rather than e.g. the minimum) is used because
    the minimum non-zero |diff| is fragile to single-sample multi-step
    jumps; the lower quartile is a more robust estimate of "one ADC
    count" for genuinely quantized channels.
    """
    if len(nonzero_abs_diffs) == 0:
        return float("nan")
    return float(np.percentile(nonzero_abs_diffs, 25))


def classify_channel(
    channel: str, nominal_values_by_segment: Sequence[np.ndarray]
) -> ChannelProfile:
    """Classify a single channel's noise/quantization regime.

    Parameters
    ----------
    channel : str
        Channel code.
    nominal_values_by_segment : sequence of 1D arrays
        Raw values for each **nominal (non-anomalous)** segment of this
        channel. Only nominal segments are used so that the classification
        reflects normal operating behavior, not anomaly-time dynamics.

    Returns
    -------
    ChannelProfile
    """
    diffs_all: List[np.ndarray] = []
    n_points = 0
    for values in nominal_values_by_segment:
        values = np.asarray(values, dtype=float)
        n_points += len(values)
        if len(values) > 1:
            diffs_all.append(np.diff(values))

    if not diffs_all:
        return ChannelProfile(
            channel=channel,
            channel_type="unknown",
            frac_zero_diff=float("nan"),
            min_nonzero_abs_diff=float("nan"),
            quantization_step=float("nan"),
            n_nominal_points=n_points,
        )

    diffs = np.concatenate(diffs_all)
    frac_zero = float(np.mean(diffs == 0))
    nonzero = diffs[diffs != 0]
    min_nonzero_abs = float(np.min(np.abs(nonzero))) if len(nonzero) else float("nan")

    if frac_zero >= QUANTIZATION_FRAC_ZERO_THRESHOLD:
        channel_type = "quantized"
        q_step = detect_quantization_step(np.abs(nonzero))
    elif np.isfinite(min_nonzero_abs) and min_nonzero_abs < FLOAT_NOISE_ABS_DIFF_THRESHOLD:
        channel_type = "float_noise_suspect"
        q_step = float("nan")
    else:
        channel_type = "continuous"
        q_step = float("nan")

    return ChannelProfile(
        channel=channel,
        channel_type=channel_type,
        frac_zero_diff=frac_zero,
        min_nonzero_abs_diff=min_nonzero_abs,
        quantization_step=q_step,
        n_nominal_points=n_points,
    )


def classify_all_channels(
    nominal_values_by_channel: dict[str, Sequence[np.ndarray]],
) -> List[ChannelProfile]:
    """Classify every channel in ``nominal_values_by_channel``.

    Parameters
    ----------
    nominal_values_by_channel : dict
        ``{channel: [segment_values, ...]}`` for nominal segments only.

    Returns
    -------
    list of ChannelProfile, in the same iteration order as the input dict.
    """
    return [
        classify_channel(channel, segments)
        for channel, segments in nominal_values_by_channel.items()
    ]


def profiles_to_records(profiles: Iterable[ChannelProfile]) -> List[dict]:
    """Flatten a collection of ChannelProfile into plain dicts, e.g. for
    ``json.dump`` or ``pandas.DataFrame.from_records``."""
    return [p.to_dict() for p in profiles]
