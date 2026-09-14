"""
layer2_anomaly_detection.bootstrap_ci
======================================
채널별 MCC의 부트스트랩 95% 신뢰구간을 계산한다. 특히 CADC0890처럼 표본이
아주 작은 채널(n_anom=11)은 점추정 MCC=0.826이 높아 보여도 표본변동만으로
결과가 뒤집힐 수 있는지 확인이 필요하다.

절차: 세그먼트 단위 (channel, is_anomaly, detected) 원자료에서 anomaly 그룹과
normal 그룹을 각각 복원추출(N_BOOT=2000회)해 매 회 MCC를 다시 계산하고,
2.5/97.5 백분위수를 CI로 삼는다.

실제 OPS-SAT-AD 실행 결과 (전체 데이터 기준, README/§1-11 근거):

    channel   mcc_median  ci_low   ci_high  n_anom  n_norm  ci_crosses_zero
    CADC0872   0.5138     0.4441   0.5775   131     415     False
    CADC0873   0.4888     0.4028   0.5641   105     488     False
    CADC0874   0.7439     0.6354   0.8320    69     125     False
    CADC0884      NaN        NaN      NaN     0     158     NaN (구조적 제외)
    CADC0886   0.0000     0.0000   0.0000     3       8     False
    CADC0888   0.1941     0.0737   0.3115    60     192     False
    CADC0890   0.8257     0.6030   1.0000    11       3     False   <- 그래도 CI 넓음
    CADC0892  -0.0902    -0.2740   0.0526    34     177     True    <- 유일하게 0을 포함
    CADC0894   0.1894     0.1461   0.2297    21     123     False

    -> CADC0892만 95% CI가 0을 포함 (§1-11에서 이미 MCC<=0으로 제외된 채널과 동일 — 상호 보강).

test-only 재확인(triple_verification.py의 best_combo 기준, N_BOOT=2000)은
CHANNEL_MCC_BOOTSTRAP_TEST_ONLY_REFERENCE를 참고.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _shared import RANDOM_STATE, mcc_from_counts, section

N_BOOT_DEFAULT = 2000

# 실제 OPS-SAT-AD 실행 결과 — 전체 데이터 기준 (섹션 H)
CHANNEL_MCC_BOOTSTRAP_REFERENCE = pd.DataFrame([
    {"channel": "CADC0872", "mcc_median": 0.513804, "mcc_ci_low": 0.444101, "mcc_ci_high": 0.577466, "n_anomaly": 131, "n_normal": 415, "ci_crosses_zero": False},
    {"channel": "CADC0873", "mcc_median": 0.488849, "mcc_ci_low": 0.402766, "mcc_ci_high": 0.564106, "n_anomaly": 105, "n_normal": 488, "ci_crosses_zero": False},
    {"channel": "CADC0874", "mcc_median": 0.743875, "mcc_ci_low": 0.635407, "mcc_ci_high": 0.831974, "n_anomaly": 69, "n_normal": 125, "ci_crosses_zero": False},
    {"channel": "CADC0884", "mcc_median": np.nan, "mcc_ci_low": np.nan, "mcc_ci_high": np.nan, "n_anomaly": 0, "n_normal": 158, "ci_crosses_zero": np.nan},
    {"channel": "CADC0886", "mcc_median": 0.000000, "mcc_ci_low": 0.000000, "mcc_ci_high": 0.000000, "n_anomaly": 3, "n_normal": 8, "ci_crosses_zero": False},
    {"channel": "CADC0888", "mcc_median": 0.194050, "mcc_ci_low": 0.073666, "mcc_ci_high": 0.311465, "n_anomaly": 60, "n_normal": 192, "ci_crosses_zero": False},
    {"channel": "CADC0890", "mcc_median": 0.825723, "mcc_ci_low": 0.603023, "mcc_ci_high": 1.000000, "n_anomaly": 11, "n_normal": 3, "ci_crosses_zero": False},
    {"channel": "CADC0892", "mcc_median": -0.090162, "mcc_ci_low": -0.274016, "mcc_ci_high": 0.052636, "n_anomaly": 34, "n_normal": 177, "ci_crosses_zero": True},
    {"channel": "CADC0894", "mcc_median": 0.189389, "mcc_ci_low": 0.146087, "mcc_ci_high": 0.229721, "n_anomaly": 21, "n_normal": 123, "ci_crosses_zero": False},
])

# 실제 OPS-SAT-AD 실행 결과 — test-only, best_combo 기준 (섹션 M)
CHANNEL_MCC_BOOTSTRAP_TEST_ONLY_REFERENCE = pd.DataFrame([
    {"channel": "CADC0872", "n_anom_test": 32, "n_norm_test": 100, "mcc_median": 0.559017, "mcc_ci_low": 0.418330, "mcc_ci_high": 0.679674, "ci_crosses_zero": False},
    {"channel": "CADC0873", "n_anom_test": 31, "n_norm_test": 122, "mcc_median": 0.434382, "mcc_ci_low": 0.280552, "mcc_ci_high": 0.578736, "ci_crosses_zero": False},
    {"channel": "CADC0874", "n_anom_test": 23, "n_norm_test": 29, "mcc_median": 0.852030, "mcc_ci_low": 0.714957, "mcc_ci_high": 0.961581, "ci_crosses_zero": False},
    {"channel": "CADC0884", "n_anom_test": 0, "n_norm_test": 36, "mcc_median": np.nan, "mcc_ci_low": np.nan, "mcc_ci_high": np.nan, "ci_crosses_zero": np.nan},
    {"channel": "CADC0886", "n_anom_test": 1, "n_norm_test": 3, "mcc_median": 0.000000, "mcc_ci_low": 0.000000, "mcc_ci_high": 0.000000, "ci_crosses_zero": False},
    {"channel": "CADC0888", "n_anom_test": 12, "n_norm_test": 52, "mcc_median": 0.320256, "mcc_ci_low": 0.091698, "mcc_ci_high": 0.527988, "ci_crosses_zero": False},
    {"channel": "CADC0890", "n_anom_test": 2, "n_norm_test": 0, "mcc_median": np.nan, "mcc_ci_low": np.nan, "mcc_ci_high": np.nan, "ci_crosses_zero": np.nan},
    {"channel": "CADC0892", "n_anom_test": 7, "n_norm_test": 46, "mcc_median": 0.054096, "mcc_ci_low": 0.000000, "mcc_ci_high": 0.095553, "ci_crosses_zero": False},
    {"channel": "CADC0894", "n_anom_test": 5, "n_norm_test": 28, "mcc_median": 0.199205, "mcc_ci_low": 0.107335, "mcc_ci_high": 0.279143, "ci_crosses_zero": False},
])


def bootstrap_channel_mcc(onset_results: pd.DataFrame, channels: list,
                           n_boot: int = N_BOOT_DEFAULT, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """onset_results (bocpd_forgetting.detect_all_channels()의 "onset_results" 출력,
    channel/is_anomaly/detected 컬럼 필요)로부터 채널별 MCC 부트스트랩 CI를 계산한다."""
    rng = np.random.default_rng(random_state)
    rows = []
    for ch in channels:
        sub = onset_results[onset_results["channel"] == ch]
        anom = sub[sub["is_anomaly"] == 1]["detected"].values
        norm = sub[sub["is_anomaly"] == 0]["detected"].values

        if len(anom) == 0 or len(norm) == 0:
            rows.append({
                "channel": ch, "mcc_median": np.nan, "mcc_ci_low": np.nan, "mcc_ci_high": np.nan,
                "n_anomaly": len(anom), "n_normal": len(norm),
                "ci_crosses_zero": np.nan, "note": "구조적 제외 (anomaly 또는 normal 세그먼트 없음)",
            })
            continue

        mccs = np.empty(n_boot)
        for i in range(n_boot):
            a_s = rng.choice(anom, size=len(anom), replace=True)
            n_s = rng.choice(norm, size=len(norm), replace=True)
            tp, fn = a_s.sum(), len(a_s) - a_s.sum()
            fp, tn = n_s.sum(), len(n_s) - n_s.sum()
            mccs[i] = mcc_from_counts(tp, fn, fp, tn)

        ci_low, ci_high = float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))
        rows.append({
            "channel": ch, "mcc_median": float(np.median(mccs)),
            "mcc_ci_low": ci_low, "mcc_ci_high": ci_high,
            "n_anomaly": len(anom), "n_normal": len(norm),
            "ci_crosses_zero": bool(ci_low < 0 < ci_high), "note": "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    section("bootstrap_ci — 실측 결과 (전체 데이터 기준)")
    print(CHANNEL_MCC_BOOTSTRAP_REFERENCE.to_string(index=False))
    flagged = CHANNEL_MCC_BOOTSTRAP_REFERENCE[CHANNEL_MCC_BOOTSTRAP_REFERENCE["ci_crosses_zero"] == True]  # noqa: E712
    print(f"\n[주의] MCC 95% CI가 0을 포함하는 채널: {flagged['channel'].tolist()}")

    section("bootstrap_ci — 실측 결과 (test-only, best_combo 기준)")
    print(CHANNEL_MCC_BOOTSTRAP_TEST_ONLY_REFERENCE.to_string(index=False))
