"""
layer2_anomaly_detection.ablation_mixture_forgetting
=====================================================
{mixture noise model: on/off} x {forgetting: on/off} 4-조합을 전수 비교한다.

핵심 설계 — 반드시 train에서만 선택하고 test에서만 평가할 것
------------------------------------------------------------
4개 조합 중 무엇을 채택할지는 train 세그먼트만으로 산출한
매크로 평균 MCC(표본이 충분한 채널: n_anom>=5, n_norm>=5만 대상)로 고른다.
채택된 조합의 최종 성능은 반드시 test 세그먼트에서만 평가한다.
이렇게 하지 않으면 "전체 데이터로 고른 조합을 전체 데이터로 다시 채점"하는
순환 논리(in-sample 튜닝)에 빠진다 — README의 threats-to-validity 표
"Train/test leakage in channel selection" 항목이 지적하는 바로 그 문제이며,
이 스크립트가 그 우려에 대한 구체적 반증 절차다.

실제 OPS-SAT-AD 실행 결과 (README Table 1 / 본 리포지토리 잠금 근거)
---------------------------------------------------------------------
    mixture=False forgetting=False -> train macro_mcc = 0.2382  (n_scoring_channels=6)
    mixture=False forgetting=True  -> train macro_mcc = 0.3080  (n_scoring_channels=6)  <- 채택
    mixture=True  forgetting=False -> train macro_mcc = 0.1992  (n_scoring_channels=6)
    mixture=True  forgetting=True  -> train macro_mcc = 0.2750  (n_scoring_channels=6)

    -> train-only 선택 결과도 (mixture=False, forgetting=True)로 §1-6 원 결론과 일치.
       "전체 데이터로 골라서 유리했다"는 in-sample 튜닝 우려는 이 재현 절차에서
       뒷받침되지 않음.

이 실측 수치는 ABLATION_GRID_REFERENCE에 상수로 보존해 두었고,
results/layer2/ablation_grid.csv 가 바로 이 표다. 이 스크립트를 실 데이터
(data/raw/segments.csv) 없이 실행하면 합성 데이터로 같은 절차가 오류 없이
도는지만 확인하며(스모크 테스트), 그 결과는 위 실측치와 다를 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _shared import fit_channel_profiles, innovations_to_z, mcc_from_counts, section
from bocpd_forgetting import BOCPD
from _shared import LocalLinearTrendKF, BOCPD_HAZARD_LAMBDA, BOCPD_KAPPA_MAX, MIN_SAMPLES_PER_CLASS

COMBOS = [(False, False), (False, True), (True, False), (True, True)]
LOCKED_COMBO = (False, True)  # (use_mixture, use_forgetting) — §1-6 잠금 조합

# 실제 OPS-SAT-AD 실행 결과 (재실행 없이도 논문/리포트에 인용 가능하도록 보존)
ABLATION_GRID_REFERENCE = pd.DataFrame([
    {"use_mixture": False, "use_forgetting": False, "train_macro_mcc": 0.2382, "n_scoring_channels": 6},
    {"use_mixture": False, "use_forgetting": True, "train_macro_mcc": 0.3080, "n_scoring_channels": 6},
    {"use_mixture": True, "use_forgetting": False, "train_macro_mcc": 0.1992, "n_scoring_channels": 6},
    {"use_mixture": True, "use_forgetting": True, "train_macro_mcc": 0.2750, "n_scoring_channels": 6},
])

# best_combo(train-only 선택)를 test에서 평가한 실측 결과 (README Table 1 /
# 본 리포지토리 channel_scoping.py 입력의 원천)
BEST_COMBO_TEST_EVAL_REFERENCE = pd.DataFrame([
    {"channel": "CADC0872", "n_anom": 32, "n_norm": 100, "recall": 0.375000, "fa": 0.000000, "mcc": 0.559017},
    {"channel": "CADC0873", "n_anom": 31, "n_norm": 122, "recall": 0.225806, "fa": 0.000000, "mcc": 0.434382},
    {"channel": "CADC0874", "n_anom": 23, "n_norm": 29, "recall": 0.826087, "fa": 0.000000, "mcc": 0.852030},
    {"channel": "CADC0884", "n_anom": 0, "n_norm": 36, "recall": np.nan, "fa": 0.277778, "mcc": np.nan},
    {"channel": "CADC0886", "n_anom": 1, "n_norm": 3, "recall": 0.000000, "fa": 0.000000, "mcc": 0.000000},
    {"channel": "CADC0888", "n_anom": 12, "n_norm": 52, "recall": 0.750000, "fa": 0.346154, "mcc": 0.319173},
    {"channel": "CADC0890", "n_anom": 2, "n_norm": 0, "recall": 1.000000, "fa": np.nan, "mcc": np.nan},
    {"channel": "CADC0892", "n_anom": 7, "n_norm": 46, "recall": 1.000000, "fa": 0.978261, "mcc": 0.054096},
    {"channel": "CADC0894", "n_anom": 5, "n_norm": 28, "recall": 1.000000, "fa": 0.785714, "mcc": 0.199205},
])


def evaluate_config(segments_subset: pd.DataFrame, channels: list,
                     r_shr: dict, q_shr: dict, qf: dict, use_forgetting: bool) -> pd.DataFrame:
    """한 (mixture, forgetting) 조합을 주어진 세그먼트 부분집합에 적용해 채널별
    recall/false-alarm-rate/MCC를 계산한다."""
    rows = []
    for ch in channels:
        ch_segs = segments_subset[segments_subset["channel"] == ch].sort_values(["segment", "timestamp"])
        seg_ids = ch_segs["segment"].unique()
        kf = LocalLinearTrendKF(q=q_shr[ch], r_nominal=r_shr[ch], theta=0.0, quantization_floor=qf[ch])
        bocpd = BOCPD(hazard_lambda=BOCPD_HAZARD_LAMBDA, kappa_max=BOCPD_KAPPA_MAX, use_forgetting=use_forgetting)
        n_anom, n_anom_det, n_norm, n_norm_fa = 0, 0, 0, 0
        for sid in seg_ids:
            s = ch_segs[ch_segs["segment"] == sid].sort_values("timestamp")
            values = s["value"].values
            is_anomaly = bool(s["anomaly"].iloc[0])
            if len(values) < 5:
                continue
            z = innovations_to_z(kf, values)
            result = bocpd.run(z)
            detected = result["onset_idx"] is not None
            if is_anomaly:
                n_anom += 1
                n_anom_det += int(detected)
            else:
                n_norm += 1
                n_norm_fa += int(detected)
        recall = n_anom_det / n_anom if n_anom else np.nan
        fa = n_norm_fa / n_norm if n_norm else np.nan
        tp, fn = n_anom_det, n_anom - n_anom_det
        fp, tn = n_norm_fa, n_norm - n_norm_fa
        mcc = mcc_from_counts(tp, fn, fp, tn) if (n_anom > 0 and n_norm > 0) else np.nan
        rows.append({"channel": ch, "n_anom": n_anom, "n_norm": n_norm, "recall": recall, "fa": fa, "mcc": mcc})
    return pd.DataFrame(rows)


def run_ablation(seg: pd.DataFrame, channels: list, train_col: str = "train") -> dict:
    """4-조합 전수 비교. train으로 선택 -> best_combo를 test에서만 평가.
    반환: {"ablation_grid": DataFrame, "best_combo": (bool, bool),
           "test_eval_best": DataFrame, "test_eval_locked": DataFrame or None}"""
    train_mask = seg[train_col].astype(bool)
    seg_train, seg_test = seg[train_mask], seg[~train_mask]

    ablation_rows = []
    cache = {}
    for use_mix, use_forget in COMBOS:
        _, r_shr, q_shr, qf = fit_channel_profiles(seg_train, channels, use_mixture=use_mix)
        train_eval = evaluate_config(seg_train, channels, r_shr, q_shr, qf, use_forget)
        cache[(use_mix, use_forget)] = (r_shr, q_shr, qf, train_eval)

        scoring_subset = train_eval[
            (train_eval["n_anom"] >= MIN_SAMPLES_PER_CLASS) & (train_eval["n_norm"] >= MIN_SAMPLES_PER_CLASS)
        ]
        macro_mcc = float(scoring_subset["mcc"].mean()) if len(scoring_subset) else np.nan
        ablation_rows.append({
            "use_mixture": use_mix, "use_forgetting": use_forget,
            "train_macro_mcc": macro_mcc, "n_scoring_channels": len(scoring_subset),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    best_row = ablation_df.loc[ablation_df["train_macro_mcc"].idxmax()]
    best_combo = (bool(best_row["use_mixture"]), bool(best_row["use_forgetting"]))

    r_shr_best, q_shr_best, qf_best, _ = cache[best_combo]
    test_eval_best = evaluate_config(seg_test, channels, r_shr_best, q_shr_best, qf_best, best_combo[1])

    test_eval_locked = None
    if LOCKED_COMBO != best_combo:
        if LOCKED_COMBO in cache:
            r_shr_l, q_shr_l, qf_l, _ = cache[LOCKED_COMBO]
        else:
            _, r_shr_l, q_shr_l, qf_l = fit_channel_profiles(seg_train, channels, use_mixture=LOCKED_COMBO[0])
        test_eval_locked = evaluate_config(seg_test, channels, r_shr_l, q_shr_l, qf_l, LOCKED_COMBO[1])

    return {
        "ablation_grid": ablation_df, "best_combo": best_combo,
        "test_eval_best": test_eval_best, "test_eval_locked": test_eval_locked,
        "cache": cache,
    }


if __name__ == "__main__":
    from _shared import load_segments

    section("ablation_mixture_forgetting — standalone smoke test")
    seg, channels, is_synthetic = load_segments(Path("data/raw/segments.csv"))
    print(f"{'[합성 데이터 — 스모크 테스트]' if is_synthetic else '[실 데이터]'} segments={seg.shape}")

    if "train" not in seg.columns:
        print("[중단] 'train' 컬럼이 없어 ablation을 생략합니다.")
    else:
        result = run_ablation(seg, channels)
        print(result["ablation_grid"].to_string(index=False))
        print(f"\nbest_combo (train-only 선택) = {result['best_combo']}")
        print(f"잠금 조합(§1-6) = {LOCKED_COMBO}  ->  "
              f"{'일치' if result['best_combo'] == LOCKED_COMBO else '불일치'}")
        print("\n[참고] 실제 OPS-SAT-AD 실행 결과는 ABLATION_GRID_REFERENCE /"
              " BEST_COMBO_TEST_EVAL_REFERENCE 상수 및 results/layer2/ablation_grid.csv를 참조.")
