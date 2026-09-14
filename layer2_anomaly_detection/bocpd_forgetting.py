"""
layer2_anomaly_detection.bocpd_forgetting
==========================================
Bayesian Online Change-Point Detection (Adams & MacKay 2007)에
"forgetting" 확장을 추가한 구현.

왜 forgetting이 필요한가
------------------------
바닐라 BOCPD는 정상 구간이 길게 이어지면 run-length 사후분포가 점점 더 큰 kappa로
쏠리면서 과신(overconfidence)에 빠지는 실패 모드가 있다 — 관측잡음 추정치가
표본이 쌓일수록 지나치게 확신을 갖게 되어, 진짜 변화가 왔을 때도 둔감해진다.
forgetting 확장은 sufficient statistics (kappa, alpha, beta)가 kappa_max를
넘으면 비례적으로 깎아내려(clip-and-rescale) run-length가 아무리 길어져도
"최근 kappa_max개 관측치만큼의 확신"으로 상한을 둔다.

1단계 ablation(본 폴더의 ablation_mixture_forgetting.py) 결론:
    forgetting=True가 float_noise_suspect 채널(872/873/874)의 recall 유지에 필수적
    (끄면 874: recall 0.725 -> 0.319로 거의 반토막).

onset 확정 규칙
---------------
MAP run-length가 ONSET_RUN_LENGTH_HIGH(5) 초과 상태에서 ONSET_RUN_LENGTH_LOW(2)
이하로 떨어지는 시점을 onset으로 확정한다. 이 시점의 run-length 사후분포
(posterior_over_onset)를 함께 저장해 두면, 층3(causal_analysis)의 몬테카를로
onset 불확실성 전파에서 재사용할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _shared import (
    BOCPD_HAZARD_LAMBDA, BOCPD_KAPPA_MAX,
    ONSET_RUN_LENGTH_HIGH, ONSET_RUN_LENGTH_LOW,
    LocalLinearTrendKF, fit_channel_profiles, innovations_to_z, section,
)


class BOCPD:
    """forgetting 확장이 포함된 BOCPD. 입력 x는 (근사적으로) 단위분산 잔차 시계열
    — LocalLinearTrendKF.run()의 innovation/sqrt(innovation_var)를 그대로 넣는다."""

    def __init__(self, hazard_lambda: float = BOCPD_HAZARD_LAMBDA,
                 mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0,
                 kappa_max: float = BOCPD_KAPPA_MAX, use_forgetting: bool = True):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.kappa_max = kappa_max
        self.use_forgetting = use_forgetting

    def _cap_sufficient_stats(self, kappa, alpha, beta):
        """forgetting 핵심: kappa가 kappa_max를 넘는 run들만 골라 (kappa, alpha, beta)를
        동일 비율로 깎는다 — run이 길어질수록 "확신"이 무한정 커지는 것을 막는다."""
        if not self.use_forgetting:
            return kappa, alpha, beta
        over = kappa > self.kappa_max
        if not np.any(over):
            return kappa, alpha, beta
        scale = np.where(over, self.kappa_max / kappa, 1.0)
        kappa2 = np.where(over, self.kappa_max, kappa)
        alpha2 = np.where(over, alpha * scale, alpha)
        beta2 = np.where(over, beta * scale, beta)
        alpha2 = np.maximum(alpha2, 1e-3)
        return kappa2, alpha2, beta2

    def run(self, x: np.ndarray) -> dict:
        n = len(x)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0

        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        map_run_length = np.zeros(n, dtype=int)

        for t in range(n):
            xt = x[t]
            dof = 2 * alpha
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            pred_probs = stats.t.pdf(xt, df=dof, loc=mu, scale=scale)
            pred_probs = np.nan_to_num(pred_probs, nan=1e-12, posinf=1e-12)

            run_probs = R[t, : t + 1]
            growth = run_probs * pred_probs * (1 - self.hazard)
            cp_prob = np.sum(run_probs * pred_probs * self.hazard)

            new_R = np.zeros(t + 2)
            new_R[0] = cp_prob
            new_R[1:] = growth
            total = new_R.sum()
            if total <= 0 or not np.isfinite(total):
                new_R[:] = 0
                new_R[0] = 1.0
            else:
                new_R /= total
            R[t + 1, : t + 2] = new_R

            map_run_length[t] = int(np.argmax(new_R))

            new_mu = np.empty(t + 2)
            new_kappa = np.empty(t + 2)
            new_alpha = np.empty(t + 2)
            new_beta = np.empty(t + 2)

            new_mu[0] = self.mu0
            new_kappa[0] = self.kappa0
            new_alpha[0] = self.alpha0
            new_beta[0] = self.beta0

            new_mu[1:] = (kappa * mu + xt) / (kappa + 1)
            new_kappa[1:] = kappa + 1
            new_alpha[1:] = alpha + 0.5
            new_beta[1:] = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))

            new_kappa, new_alpha, new_beta = self._cap_sufficient_stats(new_kappa, new_alpha, new_beta)
            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        onset_idx = None
        onset_posterior = None
        confirm_idx = None
        posterior_over_onset = None
        for t in range(1, n):
            if map_run_length[t] <= ONSET_RUN_LENGTH_LOW and map_run_length[t - 1] > ONSET_RUN_LENGTH_HIGH:
                onset_idx = t - map_run_length[t]
                confirm_idx = t
                row = R[t + 1, : t + 2]
                k = min(len(row), 30)
                posterior_over_onset = {int(r): float(row[r]) for r in range(k) if row[r] > 1e-6}
                onset_posterior = float(np.sum(row[:3]))
                break

        return {
            "run_length_posterior": R, "map_run_length": map_run_length,
            "onset_idx": onset_idx, "onset_confidence": onset_posterior,
            "confirm_idx": confirm_idx, "posterior_over_onset": posterior_over_onset,
        }


def detect_all_channels(seg: pd.DataFrame, channels: list,
                         r_shrunk: dict, q_shrunk: dict, quant_floor: dict,
                         use_forgetting: bool = True) -> dict:
    """채널별로 KF -> BOCPD를 돌려 세그먼트 단위 recall/false-alarm-rate 요약(calibration),
    세그먼트별 onset 판정 결과, onset run-length 사후분포(long format)를 만든다.

    반환 키: "calibration" (DataFrame), "onset_results" (DataFrame),
             "onset_posteriors" (DataFrame)
    """
    calibration_rows = []
    onset_result_rows = []
    onset_posterior_rows = []

    for ch in channels:
        ch_segments = seg[seg["channel"] == ch].sort_values(["segment", "timestamp"])
        seg_ids = ch_segments["segment"].unique()

        kf = LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch], theta=0.0,
                                 quantization_floor=quant_floor[ch])
        bocpd = BOCPD(hazard_lambda=BOCPD_HAZARD_LAMBDA, kappa_max=BOCPD_KAPPA_MAX,
                      use_forgetting=use_forgetting)

        n_anom, n_anom_det, n_norm, n_norm_fa = 0, 0, 0, 0
        for sid in seg_ids:
            s = ch_segments[ch_segments["segment"] == sid].sort_values("timestamp")
            values = s["value"].values
            is_anomaly = bool(s["anomaly"].iloc[0])
            if len(values) < 5:
                continue

            z = innovations_to_z(kf, values)
            result = bocpd.run(z)
            detected = result["onset_idx"] is not None

            onset_result_rows.append({
                "channel": ch, "segment": sid, "is_anomaly": int(is_anomaly),
                "n_points": len(values), "onset_idx": result["onset_idx"],
                "onset_confidence": result["onset_confidence"],
                "confirm_idx": result["confirm_idx"], "detected": int(detected),
            })

            if result["posterior_over_onset"] is not None:
                confirm_idx = result["confirm_idx"]
                for run_length, prob in result["posterior_over_onset"].items():
                    onset_posterior_rows.append({
                        "channel": ch, "segment": sid, "confirm_idx": confirm_idx,
                        "run_length_at_confirm": run_length,
                        "candidate_onset_idx": (confirm_idx - run_length) if confirm_idx is not None else np.nan,
                        "posterior_prob": prob,
                    })

            if is_anomaly:
                n_anom += 1
                n_anom_det += int(detected)
            else:
                n_norm += 1
                n_norm_fa += int(detected)

        calibration_rows.append({
            "channel": ch,
            "n_anomaly_segments": n_anom, "n_anomaly_detected": n_anom_det,
            "recall": n_anom_det / n_anom if n_anom else np.nan,
            "n_normal_segments": n_norm, "n_normal_false_alarm": n_norm_fa,
            "false_alarm_rate": n_norm_fa / n_norm if n_norm else np.nan,
        })

    return {
        "calibration": pd.DataFrame(calibration_rows),
        "onset_results": pd.DataFrame(onset_result_rows),
        "onset_posteriors": pd.DataFrame(onset_posterior_rows),
    }


if __name__ == "__main__":
    from _shared import load_segments

    section("bocpd_forgetting — standalone smoke test")
    seg, channels, is_synthetic = load_segments(Path("data/raw/segments.csv"))
    print(f"{'[합성 데이터]' if is_synthetic else '[실 데이터]'} segments={seg.shape}, channels={channels}")

    profiles, r_shrunk, q_shrunk, quant_floor = fit_channel_profiles(seg, channels)
    out = detect_all_channels(seg, channels, r_shrunk, q_shrunk, quant_floor, use_forgetting=True)

    print(out["calibration"].to_string(index=False))
