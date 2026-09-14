"""
kalman_filter.py
=================
Local-linear-trend Kalman filter used to obtain a denoised state estimate
(level + slope) for each telemetry segment before Layer-2 change-point
detection is applied.

State-space model
------------------
State vector ``x_t = [level_t, slope_t]``, evolving as a local linear
trend (a.k.a. "integrated random walk"):

    level_t = level_{t-1} + slope_{t-1}
    slope_t = slope_{t-1} + w_t,           w_t ~ N(0, q)

Observation model:

    y_t = level_t + v_t,                    v_t ~ N(0, r_eff)

where ``r_eff = r_nominal + theta + quantization_floor`` lets the caller
inflate the nominal measurement-noise variance to account for known
extra sources of observation noise (``theta``) or a quantization floor
(``quantization_step**2 / 12``, the variance of a uniform quantization
error), on top of the `noise_estimation.py` estimate.

This is a textbook scalar-observation Kalman filter; it is implemented
by hand here (rather than via a library) so that every intermediate
quantity needed downstream -- the innovation sequence and its variance --
is directly exposed for Layer-2's BOCPD to consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class KalmanFilterResult(NamedTuple):
    """Output of `LocalLinearTrendKF.run`.

    Attributes
    ----------
    innovations : np.ndarray, shape (n,)
        One-step-ahead prediction errors ``y_t - H @ x_pred_t``. This is
        the primary signal consumed by Layer 2's BOCPD.
    innovation_vars : np.ndarray, shape (n,)
        Predicted innovation variance ``S_t`` at each step (i.e. total
        uncertainty of the one-step-ahead prediction, filter + noise).
    levels : np.ndarray, shape (n,)
        Predicted level (pre-update) at each step -- the denoised state
        estimate.
    """

    innovations: np.ndarray
    innovation_vars: np.ndarray
    levels: np.ndarray


@dataclass
class LocalLinearTrendKF:
    """Scalar-observation local-linear-trend Kalman filter.

    Parameters
    ----------
    q : float
        Process-noise variance (slope innovation variance), typically
        from `noise_estimation.estimate_channel_noise(...).q`.
    r_nominal : float
        Baseline measurement-noise variance, typically from
        `noise_estimation.estimate_channel_noise(...).r`.
    theta : float, default 0.0
        Additional measurement-noise variance to add on top of
        ``r_nominal`` (e.g. a per-segment inflation term).
    quantization_floor : float, default 0.0
        Extra measurement-noise variance from ADC quantization
        (``step**2 / 12`` for a channel classified ``"quantized"`` by
        `channel_classification.py`; 0 otherwise).
    x0 : np.ndarray, shape (2,), default [0, 0]
        Initial state ``[level_0, slope_0]``.
    P0_scale : float, default 10.0
        Initial state covariance is ``P0_scale * I`` -- deliberately
        large/diffuse so the filter is not overconfident at segment start.
    """

    q: float
    r_nominal: float
    theta: float = 0.0
    quantization_floor: float = 0.0
    x0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P0_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.q < 0 or self.r_nominal < 0:
            raise ValueError("q and r_nominal must be non-negative")
        self.r_eff = self.r_nominal + self.theta + self.quantization_floor
        # State transition: local linear trend (integrated random walk).
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]])
        # Observation: we observe the level only, not the slope.
        self.H = np.array([[1.0, 0.0]])
        # Process noise enters only through the slope component.
        self.Q = np.array([[0.0, 0.0], [0.0, self.q]])

    def run(self, y: np.ndarray) -> KalmanFilterResult:
        """Run the filter forward over one segment's observations.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Raw (noisy) segment values, in time order.

        Returns
        -------
        KalmanFilterResult
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        x = np.asarray(self.x0, dtype=float).copy()
        P = np.eye(2) * self.P0_scale

        innovations = np.empty(n)
        innovation_vars = np.empty(n)
        levels = np.empty(n)

        for t in range(n):
            # --- predict ---
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            y_pred = (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T)[0, 0] + self.r_eff
            nu = y[t] - y_pred

            innovations[t] = nu
            innovation_vars[t] = S
            levels[t] = y_pred

            # --- update ---
            K = (P_pred @ self.H.T) / S
            x = x_pred + (K.flatten() * nu)
            P = P_pred - K @ self.H @ P_pred

        return KalmanFilterResult(innovations=innovations, innovation_vars=innovation_vars, levels=levels)


def run_kalman_for_segment(
    y: np.ndarray,
    r: float,
    q: float,
    quantization_step: float | None = None,
) -> KalmanFilterResult:
    """Convenience wrapper: build a `LocalLinearTrendKF` from noise
    estimates and a channel's quantization step (if any), then filter one
    segment.

    Parameters
    ----------
    y : np.ndarray
        Raw segment values.
    r : float
        Measurement-noise variance (from `noise_estimation.py`).
    q : float
        Process-noise variance (from `noise_estimation.py`).
    quantization_step : float, optional
        If the channel was classified ``"quantized"``
        (`channel_classification.py`), pass its estimated step size to
        add the corresponding uniform-quantization-error floor
        (``step**2 / 12``) to the effective measurement noise.
    """
    quant_floor = 0.0
    if quantization_step is not None and np.isfinite(quantization_step):
        quant_floor = float(quantization_step) ** 2 / 12.0
    kf = LocalLinearTrendKF(q=q, r_nominal=r, quantization_floor=quant_floor)
    return kf.run(y)
