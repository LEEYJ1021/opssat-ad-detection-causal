"""
layer3_causal_analysis/meta_analysis_random_effects.py
=============================================================================
DerSimonian-Laird random-effects meta-analysis across the 5 scoped channels
(README §Layer 3 (d), Fig. 4b, Table 4).

Layer 2/3에서 나온 채널별 효과(effect size |d|, frac_significant)를 하나의
"5채널 공통 효과"로 뭉뚱그리지 않고, 랜덤효과 모델로 풀링하면서 채널 간
이질성(I², Q-test)을 함께 정량화한다.

입력: onset_mc_propagation.py의 stage2_mc_segment_summary_corrected.csv
(reliability_tier가 high/medium인 세그먼트만 사용).

출력:
  - stage2_meta_effect_size_by_channel.csv        (|d| 메타분석 study table)
  - stage2_meta_effect_size_pooled.csv            (|d| pooled 결과, feature별)
  - stage2_meta_frac_significant_by_channel.csv   (frac_significant study table)
  - stage2_meta_frac_significant_pooled.csv       (frac_significant pooled 결과)
  - heterogeneity_summary.csv                     (README results/layer3 명명과 맞춘 통합본)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _shared import OUT_DIR, FINAL_CHANNELS, FEATURES, section

SEG_CORRECTED_PATH = OUT_DIR / "stage2_mc_segment_summary_corrected.csv"

MIN_SEGMENTS_PER_CHANNEL = 3
ALPHA = 0.05


def dl_meta_analysis(yi: np.ndarray, vi: np.ndarray, labels: list) -> dict:
    """DerSimonian & Laird (1986) 랜덤효과 메타분석."""
    valid = np.isfinite(yi) & np.isfinite(vi) & (vi > 0)
    yi, vi, labels = yi[valid], vi[valid], [l for l, v in zip(labels, valid) if v]
    k = len(yi)

    if k < 2:
        return {"k": k, "pooled": np.nan, "se_pooled": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "z": np.nan, "p_value": np.nan, "Q": np.nan, "df": np.nan, "p_Q": np.nan,
                "tau2": np.nan, "I2": np.nan, "study_table": pd.DataFrame(), "note": f"study 수 부족(k={k})"}

    wi_fixed = 1.0 / vi
    pooled_fixed = np.sum(wi_fixed * yi) / np.sum(wi_fixed)
    Q = float(np.sum(wi_fixed * (yi - pooled_fixed) ** 2))
    df = k - 1
    p_Q = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else np.nan

    C = np.sum(wi_fixed) - np.sum(wi_fixed ** 2) / np.sum(wi_fixed)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    wi_random = 1.0 / (vi + tau2)
    pooled = float(np.sum(wi_random * yi) / np.sum(wi_random))
    se_pooled = float(np.sqrt(1.0 / np.sum(wi_random)))
    z = pooled / se_pooled if se_pooled > 0 else np.nan
    p_value = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    ci_z = stats.norm.ppf(1 - ALPHA / 2)
    ci_low, ci_high = pooled - ci_z * se_pooled, pooled + ci_z * se_pooled

    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    study_table = pd.DataFrame({
        "channel": labels, "yi": yi, "vi": vi, "se": np.sqrt(vi),
        "weight_random_raw": wi_random, "weight_random_pct": wi_random / wi_random.sum() * 100,
        "ci_low": yi - ci_z * np.sqrt(vi), "ci_high": yi + ci_z * np.sqrt(vi),
    })

    return {"k": k, "pooled": pooled, "se_pooled": se_pooled, "ci_low": ci_low, "ci_high": ci_high,
            "z": z, "p_value": p_value, "Q": Q, "df": df, "p_Q": p_Q, "tau2": tau2, "I2": I2,
            "study_table": study_table, "note": ""}


def interpret_I2(i2: float) -> str:
    if not np.isfinite(i2):
        return "판정불가"
    if i2 < 25:
        return "낮음(채널 간 일관됨)"
    if i2 < 50:
        return "중간"
    if i2 < 75:
        return "높음"
    return "매우 높음(채널마다 메커니즘이 다를 가능성)"


def arcsine_transform(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-6, 1 - 1e-6)
    return np.arcsin(np.sqrt(p_c))


def arcsine_backtransform(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.sin(np.clip(x, 0, np.pi / 2)) ** 2)


def channel_effect_and_variance(df: pd.DataFrame, value_col: str, channels: list = FINAL_CHANNELS) -> tuple[np.ndarray, np.ndarray, list]:
    """채널별로 세그먼트 값들의 평균(yi)과 표준오차 제곱(vi=SE^2)을 계산."""
    yi_list, vi_list, labels = [], [], []
    for ch in channels:
        vals = df.loc[df["channel"] == ch, value_col].dropna().values
        n = len(vals)
        if n < 2:
            yi_list.append(np.nan)
            vi_list.append(np.nan)
        else:
            mean_val = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(n))
            yi_list.append(mean_val)
            vi_list.append(se ** 2 if se > 0 else np.nan)
        labels.append(ch)
    return np.array(yi_list), np.array(vi_list), labels


def run_effect_size_meta(reliable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("1. 효과크기(|Cohen's d|) 랜덤효과 메타분석 (level / diff / diff2)")
    pooled_rows, study_tables = [], []
    for feat in FEATURES:
        col = f"{feat}_d_abs_median"
        yi, vi, labels = channel_effect_and_variance(reliable, col)
        result = dl_meta_analysis(yi, vi, labels)
        print(f"\n--- {feat} (|d|) --- pooled={result['pooled']}, I2={result['I2']}, p_Q={result['p_Q']}")
        if len(result["study_table"]):
            st = result["study_table"].copy()
            st["feature"] = feat
            study_tables.append(st)
        pooled_rows.append({"feature": feat, "metric": "d_abs_median", "k": result["k"],
                             "pooled": result["pooled"], "se_pooled": result["se_pooled"],
                             "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                             "z": result["z"], "p_value": result["p_value"], "Q": result["Q"],
                             "df": result["df"], "p_Q": result["p_Q"], "tau2": result["tau2"],
                             "I2": result["I2"], "I2_interpretation": interpret_I2(result["I2"]),
                             "note": result["note"]})
    pooled_df = pd.DataFrame(pooled_rows)
    n_tests = int(pooled_df["p_value"].notna().sum())
    pooled_df["p_value_bonferroni"] = (pooled_df["p_value"] * max(n_tests, 1)).clip(upper=1.0)
    study_df = pd.concat(study_tables, ignore_index=True) if study_tables else pd.DataFrame()

    study_df.to_csv(OUT_DIR / "stage2_meta_effect_size_by_channel.csv", index=False)
    pooled_df.to_csv(OUT_DIR / "stage2_meta_effect_size_pooled.csv", index=False)
    return study_df, pooled_df


def run_frac_significant_meta(reliable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("2. 유의미 비율(frac_significant) 랜덤효과 메타분석 (arcsine 제곱근 변환)")
    pooled_rows, study_tables = [], []
    for feat in FEATURES:
        col = f"{feat}_frac_significant"
        raw = reliable[["channel", col]].copy()
        raw["transformed"] = arcsine_transform(raw[col].values)
        yi, vi, labels = channel_effect_and_variance(raw, "transformed")
        result = dl_meta_analysis(yi, vi, labels)
        pooled_back = arcsine_backtransform(result["pooled"])
        ci_low_back = arcsine_backtransform(result["ci_low"])
        ci_high_back = arcsine_backtransform(result["ci_high"])
        print(f"\n--- {feat} (frac_significant) --- pooled_prop={pooled_back}, I2={result['I2']}")

        if len(result["study_table"]):
            st = result["study_table"].copy()
            st["yi_prop"] = st["yi"].apply(arcsine_backtransform)
            st["feature"] = feat
            study_tables.append(st)

        pooled_rows.append({"feature": feat, "metric": "frac_significant", "k": result["k"],
                             "pooled_arcsine_scale": result["pooled"], "pooled_proportion": pooled_back,
                             "ci_low_proportion": ci_low_back, "ci_high_proportion": ci_high_back,
                             "z": result["z"], "p_value": result["p_value"], "Q": result["Q"],
                             "df": result["df"], "p_Q": result["p_Q"], "tau2": result["tau2"],
                             "I2": result["I2"], "I2_interpretation": interpret_I2(result["I2"]),
                             "note": result["note"]})
    pooled_df = pd.DataFrame(pooled_rows)
    n_tests = int(pooled_df["p_value"].notna().sum())
    pooled_df["p_value_bonferroni"] = (pooled_df["p_value"] * max(n_tests, 1)).clip(upper=1.0)
    study_df = pd.concat(study_tables, ignore_index=True) if study_tables else pd.DataFrame()

    study_df.to_csv(OUT_DIR / "stage2_meta_frac_significant_by_channel.csv", index=False)
    pooled_df.to_csv(OUT_DIR / "stage2_meta_frac_significant_pooled.csv", index=False)
    return study_df, pooled_df


def main() -> None:
    section("0. 데이터 로드 및 채널별 효과크기(yi) / 분산(vi) 산출")
    seg_corrected = pd.read_csv(SEG_CORRECTED_PATH)
    reliable = seg_corrected[
        seg_corrected["reliability_tier"].isin(["high", "medium"])
        & seg_corrected["channel"].isin(FINAL_CHANNELS)
    ].copy()
    n_per_channel = reliable.groupby("channel").size()
    print(n_per_channel.reindex(FINAL_CHANNELS).to_string())
    thin = n_per_channel[n_per_channel < MIN_SEGMENTS_PER_CHANNEL].index.tolist()
    if thin:
        print(f"[주의] 세그먼트 {MIN_SEGMENTS_PER_CHANNEL}개 미만 채널: {thin} — SE 추정 불안정할 수 있음")

    _, effect_pooled = run_effect_size_meta(reliable)
    _, frac_pooled = run_frac_significant_meta(reliable)

    # README results/layer3/heterogeneity_summary.csv 명명과 맞춘 통합 요약
    het_rows = []
    for _, r in effect_pooled.iterrows():
        het_rows.append({"feature": r["feature"], "metric": r["metric"], "I2": r["I2"], "p_Q": r["p_Q"],
                          "I2_interpretation": r["I2_interpretation"]})
    for _, r in frac_pooled.iterrows():
        het_rows.append({"feature": r["feature"], "metric": r["metric"], "I2": r["I2"], "p_Q": r["p_Q"],
                          "I2_interpretation": r["I2_interpretation"]})
    pd.DataFrame(het_rows).to_csv(OUT_DIR / "heterogeneity_summary.csv", index=False)


if __name__ == "__main__":
    main()
