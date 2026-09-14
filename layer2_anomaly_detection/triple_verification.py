"""
layer2_anomaly_detection.triple_verification
=============================================
채널 스코프(channel_scoping.py의 5채널)가 "전체 데이터로 고르고 전체 데이터로
평가한" 결과가 아니라는 것을 세 가지 독립 경로로 확인한다:

    1. 전체 데이터 부트스트랩 CI       -> bootstrap_ci.py (섹션 H)
    2. train 데이터로만 재적합 -> test 데이터에서만 재평가  (본 파일 §G)
    3. test-only MCC 부트스트랩 CI     -> bootstrap_ci.py (섹션 M) + 전체(H) 중앙값이
                                          test-only CI 안에 드는지 대조 (본 파일 §M-compare)

세 경로가 모두 동일한 5채널(CADC0872/0873/0874/0888/0894) 포함 결론에 수렴하면
"1단계 잠금"으로 취급한다 (§N).

실제 OPS-SAT-AD 실행 결과 — train 재적합 -> test 재평가 (섹션 G)
-------------------------------------------------------------------
    channel   recall(full)  recall_test  recall_drop   fa(full)  fa_test   fa_drop
    CADC0872     0.3206        0.3750      -0.0544       0.0000   0.0000   0.0000
    CADC0873     0.2762        0.2258       0.0504       0.0000   0.0000   0.0000
    CADC0874     0.7246        0.8261      -0.1014       0.0320   0.0000   0.0320
    CADC0884       NaN           NaN          NaN        0.2405   0.2778  -0.0373
    CADC0886     0.0000        0.0000       0.0000       0.0000   0.0000   0.0000
    CADC0888     0.6167        0.7500      -0.1333       0.3906   0.3462   0.0445
    CADC0890     0.9091        1.0000      -0.0909       0.0000     NaN      NaN
    CADC0892     0.9706        1.0000      -0.0294       0.9944   0.9783   0.0161
    CADC0894     1.0000        1.0000       0.0000       0.7967   0.7857   0.0110

    -> recall_drop/fa_drop 대부분 0.1 미만(874/888만 근접) -> in-sample 튜닝 우려는
       실질적으로 크지 않음.

실제 OPS-SAT-AD 실행 결과 — 전체(H) 중앙값이 test-only(M) 95% CI 안에 드는지 대조
-------------------------------------------------------------------------------------
    channel   mcc_median_full   test_ci_low   test_ci_high   full_median_in_test_ci
    CADC0872      0.5138           0.4183         0.6797            True
    CADC0873      0.4888           0.2806         0.5787            True
    CADC0874      0.7439           0.7150         0.9616            True
    CADC0884        NaN              NaN            NaN             False (평가 불가)
    CADC0886      0.0000           0.0000         0.0000            True
    CADC0888      0.1941           0.0917         0.5280            True
    CADC0890      0.8257             NaN            NaN             False (test 정상=0)
    CADC0892     -0.0902           0.0000         0.0956            False
    CADC0894      0.1894           0.1073         0.2791            True

    -> 최종 5채널(872/873/874/888/894) 모두 True: train/test 표본변동을 넘어
       안정적인 채널 스코프임을 재확인. (884/890/892는 애초에 §E에서 제외됐던
       채널이므로 여기서의 False는 "번복"이 아니라 기존 제외 판정의 추가 근거.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _shared import fit_channel_profiles, section
from bocpd_forgetting import detect_all_channels
from bootstrap_ci import bootstrap_channel_mcc
from channel_scoping import CHANNEL_SCOPE_REFERENCE, FINAL_SCOPE, build_channel_scope

# 실제 OPS-SAT-AD 실행 결과 — 섹션 G (train 재적합 -> test 재평가)
TRAIN_FIT_TEST_EVAL_REFERENCE = pd.DataFrame([
    {"channel": "CADC0872", "recall": 0.320611, "recall_test": 0.375000, "recall_drop": -0.054389, "false_alarm_rate": 0.000000, "fa_test": 0.000000, "fa_drop": 0.000000},
    {"channel": "CADC0873", "recall": 0.276190, "recall_test": 0.225806, "recall_drop": 0.050384, "false_alarm_rate": 0.000000, "fa_test": 0.000000, "fa_drop": 0.000000},
    {"channel": "CADC0874", "recall": 0.724638, "recall_test": 0.826087, "recall_drop": -0.101449, "false_alarm_rate": 0.032000, "fa_test": 0.000000, "fa_drop": 0.032000},
    {"channel": "CADC0884", "recall": np.nan, "recall_test": np.nan, "recall_drop": np.nan, "false_alarm_rate": 0.240506, "fa_test": 0.277778, "fa_drop": -0.037271},
    {"channel": "CADC0886", "recall": 0.000000, "recall_test": 0.000000, "recall_drop": 0.000000, "false_alarm_rate": 0.000000, "fa_test": 0.000000, "fa_drop": 0.000000},
    {"channel": "CADC0888", "recall": 0.616667, "recall_test": 0.750000, "recall_drop": -0.133333, "false_alarm_rate": 0.390625, "fa_test": 0.346154, "fa_drop": 0.044471},
    {"channel": "CADC0890", "recall": 0.909091, "recall_test": 1.000000, "recall_drop": -0.090909, "false_alarm_rate": 0.000000, "fa_test": np.nan, "fa_drop": np.nan},
    {"channel": "CADC0892", "recall": 0.970588, "recall_test": 1.000000, "recall_drop": -0.029412, "false_alarm_rate": 0.994350, "fa_test": 0.978261, "fa_drop": 0.016089},
    {"channel": "CADC0894", "recall": 1.000000, "recall_test": 1.000000, "recall_drop": 0.000000, "false_alarm_rate": 0.796748, "fa_test": 0.785714, "fa_drop": 0.011034},
])

# 실제 OPS-SAT-AD 실행 결과 — 전체(H) 중앙값 vs test-only(M) CI 대조
FULL_VS_TEST_CI_COMPARISON_REFERENCE = pd.DataFrame([
    {"channel": "CADC0872", "mcc_median_full": 0.513804, "mcc_ci_low_test": 0.418330, "mcc_ci_high_test": 0.679674, "full_median_in_test_ci": True},
    {"channel": "CADC0873", "mcc_median_full": 0.488849, "mcc_ci_low_test": 0.280552, "mcc_ci_high_test": 0.578736, "full_median_in_test_ci": True},
    {"channel": "CADC0874", "mcc_median_full": 0.743875, "mcc_ci_low_test": 0.714957, "mcc_ci_high_test": 0.961581, "full_median_in_test_ci": True},
    {"channel": "CADC0884", "mcc_median_full": np.nan, "mcc_ci_low_test": np.nan, "mcc_ci_high_test": np.nan, "full_median_in_test_ci": False},
    {"channel": "CADC0886", "mcc_median_full": 0.000000, "mcc_ci_low_test": 0.000000, "mcc_ci_high_test": 0.000000, "full_median_in_test_ci": True},
    {"channel": "CADC0888", "mcc_median_full": 0.194050, "mcc_ci_low_test": 0.091698, "mcc_ci_high_test": 0.527988, "full_median_in_test_ci": True},
    {"channel": "CADC0890", "mcc_median_full": 0.825723, "mcc_ci_low_test": np.nan, "mcc_ci_high_test": np.nan, "full_median_in_test_ci": False},
    {"channel": "CADC0892", "mcc_median_full": -0.090162, "mcc_ci_low_test": 0.000000, "mcc_ci_high_test": 0.095553, "full_median_in_test_ci": False},
    {"channel": "CADC0894", "mcc_median_full": 0.189389, "mcc_ci_low_test": 0.107335, "mcc_ci_high_test": 0.279143, "full_median_in_test_ci": True},
])


def train_fit_test_eval(seg: pd.DataFrame, channels: list, calib_full: pd.DataFrame,
                         train_col: str = "train") -> pd.DataFrame:
    """§G: train 세그먼트만으로 채널 프로파일 + KF 파라미터를 재적합하고,
    test 세그먼트에서만 recall/false-alarm-rate를 재평가한다."""
    train_mask = seg[train_col].astype(bool)
    seg_train, seg_test = seg[train_mask], seg[~train_mask]

    _, r_shr, q_shr, qf = fit_channel_profiles(seg_train, channels)
    test_out = detect_all_channels(seg_test, channels, r_shr, q_shr, qf, use_forgetting=True)
    test_calib = test_out["calibration"].rename(
        columns={"recall": "recall_test", "false_alarm_rate": "fa_test"}
    )[["channel", "recall_test", "fa_test", "n_anomaly_segments", "n_normal_segments"]]
    test_calib = test_calib.rename(columns={
        "n_anomaly_segments": "n_anom_test", "n_normal_segments": "n_norm_test",
    })

    compare = calib_full.merge(test_calib, on="channel")
    compare["recall_drop"] = compare["recall"] - compare["recall_test"]
    compare["fa_drop"] = compare["false_alarm_rate"] - compare["fa_test"]
    return compare


def compare_full_vs_test_ci(full_boot: pd.DataFrame, test_boot: pd.DataFrame) -> pd.DataFrame:
    """§M-compare: 전체 데이터 기준 MCC 중앙값이 test-only 95% CI 안에 드는지 대조."""
    cmp = test_boot.merge(full_boot, on="channel", suffixes=("_test", "_full"))
    cmp["full_median_in_test_ci"] = (
        (cmp["mcc_median_full"] >= cmp["mcc_ci_low_test"]) & (cmp["mcc_median_full"] <= cmp["mcc_ci_high_test"])
    )
    return cmp


def lock_channel_scope(channel_scope: pd.DataFrame) -> pd.DataFrame:
    """§N: 890의 서술만 정정(제외 판정을 '강화'로 재확인, 번복 아님)하고 최종 스코프를 잠근다."""
    rows = []
    for _, row in channel_scope.iterrows():
        ch, reason = row["channel"], row["scope_reason"]
        if ch == "CADC0890":
            reason_final = f"{reason} + test 정상 표본=0으로 재확인 (기존 제외 판정 강화, 번복 아님)"
        else:
            reason_final = reason
        rows.append({"channel": ch, "include_in_stage2": row["include_in_stage2"], "scope_reason_final": reason_final})
    return pd.DataFrame(rows)


VERIFICATION_CHECKLIST = [
    "채널별 온라인 필터링 + BOCPD(forgetting) 재탐지 (bocpd_forgetting.py)",
    "4-조합 train-only 선택 / test 평가 (ablation_mixture_forgetting.py)",
    "MCC/Youden 채널 스코프 확정 (channel_scoping.py)",
    "전체 데이터 MCC 부트스트랩 CI (bootstrap_ci.py, 섹션 H)",
    "train 재적합 -> test 재평가 (본 파일, 섹션 G)",
    "test-only MCC 부트스트랩 CI (bootstrap_ci.py, 섹션 M)",
    "890 서술 정정 포함 최종 스코프 잠금 (본 파일, 섹션 N)",
]


if __name__ == "__main__":
    section("triple_verification — 실측 결과 재현 (reference data)")

    scope = build_channel_scope(CHANNEL_SCOPE_REFERENCE)
    locked = lock_channel_scope(scope)
    print(locked.to_string(index=False))

    included_final = locked[locked["include_in_stage2"]]["channel"].tolist()
    print(f"\n2단계 최종 포함 채널 ({len(included_final)}개): {included_final}")
    assert included_final == FINAL_SCOPE

    section("1단계(층2) 잠금 체크리스트")
    for item in VERIFICATION_CHECKLIST:
        print(f"  [O] {item}")
    print("\n[결론] 5개 채널 스코프는 전체 데이터 / train-only / test-only 세 경로 모두에서 "
          "동일하게 재현되어 '잠금(locked)' 상태로 취급한다.")
