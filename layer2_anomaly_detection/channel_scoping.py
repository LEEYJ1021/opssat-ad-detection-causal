"""
layer2_anomaly_detection.channel_scoping
=========================================
채널별 recall/false-alarm-rate로부터 MCC, Youden's J를 계산하고,
아래 세 가지 기준으로 "층3(causal_analysis)에 포함할 채널"을 확정한다:

    1. 구조적 제외   — n_anom == 0 (평가 자체가 불가능; CADC0884)
    2. 표본부족 제외 — n_anom < 5 또는 n_norm < 5 (추정이 불안정; CADC0886, CADC0890)
    3. 우연수준 제외 — MCC <= 0 (탐지기가 무작위보다 못함; CADC0892)
    4. 그 외 -> 포함

실제 OPS-SAT-AD 실행 결과 (README Table 1과 동일, 이 스크립트가 실 데이터 없이
실행되면 재현되지 않으므로 CHANNEL_SCOPE_REFERENCE에 실측치를 보존해 둔다):

    channel   type                  recall   fa        mcc      youden_j  reason                          include
    CADC0872  float_noise_suspect   0.3206   0.0000    0.5138    0.3206   포함                             True
    CADC0873  float_noise_suspect   0.2762   0.0000    0.4888    0.2762   포함                             True
    CADC0874  float_noise_suspect   0.7246   0.0320    0.7398    0.6926   포함                             True
    CADC0884  quantized             NaN      0.2405    NaN       NaN      구조적 제외 (anomaly=0)          False
    CADC0886  quantized             0.0000   0.0000    0.0000    0.0000   표본부족 (n_anom=3, n_norm=8)    False
    CADC0888  quantized             0.6167   0.3906    0.1938    0.2260   포함                             True
    CADC0890  continuous            0.9091   0.0000    0.8257    0.9091   표본부족 (n_anom=11, n_norm=3)   False
    CADC0892  quantized             0.9706   0.9944   -0.0902   -0.0238   우연수준 이하 (MCC<=0)           False
    CADC0894  quantized             1.0000   0.7967    0.1894    0.2033   포함                             True

    -> 최종 스코프 (5개): CADC0872, CADC0873, CADC0874, CADC0888, CADC0894

주의: CADC0890은 MCC=0.826으로 매우 높지만 표본이 n_anom=11로 작아 "포함"으로
채택하지 않는다 — 이는 보수적 선택이며, triple_verification.py의 test-only
재확인에서 test 정상 표본=0으로 추가 확인(번복이 아니라 강화)된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _shared import MIN_SAMPLES_PER_CLASS, section

# 실제 OPS-SAT-AD 실행 결과 (재현 없이도 그대로 인용 가능하도록 보존)
CHANNEL_SCOPE_REFERENCE = pd.DataFrame([
    {"channel": "CADC0872", "channel_type": "float_noise_suspect", "recall": 0.320611, "false_alarm_rate": 0.000000},
    {"channel": "CADC0873", "channel_type": "float_noise_suspect", "recall": 0.276190, "false_alarm_rate": 0.000000},
    {"channel": "CADC0874", "channel_type": "float_noise_suspect", "recall": 0.724638, "false_alarm_rate": 0.032000},
    {"channel": "CADC0884", "channel_type": "quantized", "recall": np.nan, "false_alarm_rate": 0.240506},
    {"channel": "CADC0886", "channel_type": "quantized", "recall": 0.000000, "false_alarm_rate": 0.000000},
    {"channel": "CADC0888", "channel_type": "quantized", "recall": 0.616667, "false_alarm_rate": 0.390625},
    {"channel": "CADC0890", "channel_type": "continuous", "recall": 0.909091, "false_alarm_rate": 0.000000},
    {"channel": "CADC0892", "channel_type": "quantized", "recall": 0.970588, "false_alarm_rate": 0.994350},
    {"channel": "CADC0894", "channel_type": "quantized", "recall": 1.000000, "false_alarm_rate": 0.796748},
])
CHANNEL_SCOPE_REFERENCE["n_anomaly_segments"] = [131, 105, 69, 0, 3, 60, 11, 34, 21]
CHANNEL_SCOPE_REFERENCE["n_normal_segments"] = [415, 488, 125, 158, 8, 192, 3, 177, 123]

FINAL_SCOPE = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]


def channel_mcc_youden(row: pd.Series) -> dict:
    n_anom, n_norm = row["n_anomaly_segments"], row["n_normal_segments"]
    recall, fa = row["recall"], row["false_alarm_rate"]

    if n_anom == 0 or pd.isna(recall):
        return {"mcc": np.nan, "youden_j": np.nan, "scope_reason": "구조적 제외 (anomaly=0)"}

    tp = recall * n_anom
    fn = n_anom - tp
    fp = fa * n_norm
    tn = n_norm - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    youden_j = recall - fa

    if n_anom < MIN_SAMPLES_PER_CLASS or n_norm < MIN_SAMPLES_PER_CLASS:
        reason = f"표본부족 (n_anom={int(n_anom)}, n_norm={int(n_norm)})"
    elif mcc <= 0:
        reason = "우연수준 이하 (MCC<=0)"
    else:
        reason = "포함"
    return {"mcc": mcc, "youden_j": youden_j, "scope_reason": reason}


def build_channel_scope(calibration: pd.DataFrame) -> pd.DataFrame:
    """detect_all_channels()가 반환하는 calibration DataFrame
    (channel, recall, false_alarm_rate, n_anomaly_segments, n_normal_segments 필요)
    으로부터 채널 스코프 결정표를 만든다."""
    scope_rows = []
    for _, row in calibration.iterrows():
        stats_row = channel_mcc_youden(row)
        scope_rows.append({
            "channel": row["channel"],
            "channel_type": row.get("channel_type", ""),
            "recall": row["recall"],
            "false_alarm_rate": row["false_alarm_rate"],
            **stats_row,
            "include_in_stage2": stats_row["scope_reason"] == "포함",
        })
    return pd.DataFrame(scope_rows)


def to_json_summary(channel_scope: pd.DataFrame) -> dict:
    included = channel_scope[channel_scope["include_in_stage2"]]["channel"].tolist()
    excluded = {
        row["channel"]: row["scope_reason"]
        for _, row in channel_scope[~channel_scope["include_in_stage2"]].iterrows()
    }
    return {
        "included_channels": included,
        "excluded_channels": excluded,
        "n_included": len(included),
        "n_total": len(channel_scope),
        "decision_table": channel_scope.to_dict(orient="records"),
    }


if __name__ == "__main__":
    section("channel_scoping — 실측 결과 기준 재현 (reference data)")
    scope = build_channel_scope(CHANNEL_SCOPE_REFERENCE)
    print(scope.to_string(index=False))

    included = scope[scope["include_in_stage2"]]["channel"].tolist()
    print(f"\n2단계(층3) 포함 채널 ({len(included)}개): {included}")
    assert included == FINAL_SCOPE, "reference 계산 결과가 잠금된 FINAL_SCOPE와 다릅니다."
    print("[검증 통과] FINAL_SCOPE와 일치")
