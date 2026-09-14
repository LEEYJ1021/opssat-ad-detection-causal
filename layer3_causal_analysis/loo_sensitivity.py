"""
layer3_causal_analysis/loo_sensitivity.py
=============================================================================
Leave-one-channel-out (LOO) sensitivity analysis for the random-effects
meta-analysis (README §Layer 3 (d), "Threats to validity" table).

meta_analysis_random_effects.py에서 diff/diff2/level 모두 I²가 65~97%로 매우
높게 나왔다. 채널을 하나씩 빼고 DL 메타분석을 재실행해 pooled 값과 I²가
특정 채널의 존재/부재에 얼마나 좌우되는지를 정량화한다.

출력:
  - stage2_meta_loo_effect_size.csv        (LOO 효과크기 결과, feature x 제외채널)
  - stage2_meta_loo_frac_significant.csv   (LOO 유의미비율 결과, feature x 제외채널)
  - stage2_meta_loo_influential_flags.csv  (영향력 있는 채널 판정 요약)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _shared import OUT_DIR, FINAL_CHANNELS, FEATURES, section
from meta_analysis_random_effects import (
    dl_meta_analysis, arcsine_transform, arcsine_backtransform,
    channel_effect_and_variance, SEG_CORRECTED_PATH,
)

I2_DROP_THRESHOLD = 30.0
CI_SHIFT_FLAG = True


def run_loo_effect_size(reliable: pd.DataFrame) -> pd.DataFrame:
    section("1. 효과크기(|Cohen's d|) Leave-One-Out 민감도 분석")
    rows = []
    for feat in FEATURES:
        col = f"{feat}_d_abs_median"
        yi_full, vi_full, labels_full = channel_effect_and_variance(reliable, col, list(FINAL_CHANNELS))
        full_result = dl_meta_analysis(yi_full, vi_full, labels_full)
        rows.append({"feature": feat, "excluded_channel": "(none, 전체포함)", "k": full_result["k"],
                     "pooled": full_result["pooled"], "ci_low": full_result["ci_low"],
                     "ci_high": full_result["ci_high"], "p_value": full_result["p_value"],
                     "I2": full_result["I2"], "tau2": full_result["tau2"],
                     "pooled_shift_from_full": 0.0, "I2_drop_from_full": 0.0})

        for excl_ch in FINAL_CHANNELS:
            remaining = [c for c in FINAL_CHANNELS if c != excl_ch]
            yi, vi, labels = channel_effect_and_variance(reliable, col, remaining)
            result = dl_meta_analysis(yi, vi, labels)
            pooled_shift = result["pooled"] - full_result["pooled"] if np.isfinite(result["pooled"]) else np.nan
            i2_drop = full_result["I2"] - result["I2"] if np.isfinite(result["I2"]) else np.nan
            rows.append({"feature": feat, "excluded_channel": excl_ch, "k": result["k"],
                         "pooled": result["pooled"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                         "p_value": result["p_value"], "I2": result["I2"], "tau2": result["tau2"],
                         "pooled_shift_from_full": pooled_shift, "I2_drop_from_full": i2_drop})

    loo_effect_df = pd.DataFrame(rows)
    loo_effect_df.to_csv(OUT_DIR / "stage2_meta_loo_effect_size.csv", index=False)
    print(loo_effect_df.to_string(index=False))
    return loo_effect_df


def run_loo_frac_significant(reliable: pd.DataFrame) -> pd.DataFrame:
    section("2. 유의미 비율(frac_significant) Leave-One-Out 민감도 분석")
    rows = []
    for feat in FEATURES:
        col = f"{feat}_frac_significant"
        raw = reliable[["channel", col]].copy()
        raw["transformed"] = arcsine_transform(raw[col].values)

        yi_full, vi_full, labels_full = channel_effect_and_variance(raw, "transformed", list(FINAL_CHANNELS))
        full_result = dl_meta_analysis(yi_full, vi_full, labels_full)
        full_pooled_prop = arcsine_backtransform(full_result["pooled"])
        full_ci_low_prop = arcsine_backtransform(full_result["ci_low"])
        full_ci_high_prop = arcsine_backtransform(full_result["ci_high"])
        rows.append({"feature": feat, "excluded_channel": "(none, 전체포함)", "k": full_result["k"],
                     "pooled_proportion": full_pooled_prop, "ci_low_proportion": full_ci_low_prop,
                     "ci_high_proportion": full_ci_high_prop, "p_value": full_result["p_value"],
                     "I2": full_result["I2"], "pooled_shift_from_full": 0.0, "I2_drop_from_full": 0.0})

        for excl_ch in FINAL_CHANNELS:
            remaining = [c for c in FINAL_CHANNELS if c != excl_ch]
            yi, vi, labels = channel_effect_and_variance(raw, "transformed", remaining)
            result = dl_meta_analysis(yi, vi, labels)
            pooled_prop = arcsine_backtransform(result["pooled"])
            ci_low_prop = arcsine_backtransform(result["ci_low"])
            ci_high_prop = arcsine_backtransform(result["ci_high"])
            pooled_shift = pooled_prop - full_pooled_prop if np.isfinite(pooled_prop) else np.nan
            i2_drop = full_result["I2"] - result["I2"] if np.isfinite(result["I2"]) else np.nan
            rows.append({"feature": feat, "excluded_channel": excl_ch, "k": result["k"],
                         "pooled_proportion": pooled_prop, "ci_low_proportion": ci_low_prop,
                         "ci_high_proportion": ci_high_prop, "p_value": result["p_value"], "I2": result["I2"],
                         "pooled_shift_from_full": pooled_shift, "I2_drop_from_full": i2_drop})

    loo_frac_df = pd.DataFrame(rows)
    loo_frac_df.to_csv(OUT_DIR / "stage2_meta_loo_frac_significant.csv", index=False)
    print(loo_frac_df.to_string(index=False))
    return loo_frac_df


def flag_influential_channels(loo_effect_df: pd.DataFrame, loo_frac_df: pd.DataFrame) -> pd.DataFrame:
    section("3. 영향력 있는 채널 판정 (pooled 값 이동 + I^2 급락 기준)")
    rows = []
    for metric_name, df, pooled_col in [
        ("effect_size(|d|)", loo_effect_df, "pooled"),
        ("frac_significant", loo_frac_df, "pooled_proportion"),
    ]:
        for feat in FEATURES:
            sub = df[(df["feature"] == feat) & (df["excluded_channel"] != "(none, 전체포함)")]
            full_row = df[(df["feature"] == feat) & (df["excluded_channel"] == "(none, 전체포함)")].iloc[0]
            for _, row in sub.iterrows():
                flag_i2 = bool(row["I2_drop_from_full"] >= I2_DROP_THRESHOLD) if np.isfinite(row["I2_drop_from_full"]) else False
                flag_ci = False
                if CI_SHIFT_FLAG and np.isfinite(row[pooled_col]):
                    if pooled_col == "pooled":
                        flag_ci = not (full_row["ci_low"] <= row[pooled_col] <= full_row["ci_high"])
                    else:
                        flag_ci = not (full_row["ci_low_proportion"] <= row[pooled_col] <= full_row["ci_high_proportion"])
                rows.append({"metric": metric_name, "feature": feat, "excluded_channel": row["excluded_channel"],
                             "I2_drop_from_full": row["I2_drop_from_full"],
                             "pooled_shift_from_full": row["pooled_shift_from_full"],
                             f"flag_I2_drop(>={I2_DROP_THRESHOLD:.0f}%p)": flag_i2,
                             "flag_pooled_outside_full_CI": flag_ci, "influential": bool(flag_i2 or flag_ci)})

    influential_df = pd.DataFrame(rows)
    influential_df.to_csv(OUT_DIR / "stage2_meta_loo_influential_flags.csv", index=False)
    print(influential_df.to_string(index=False))

    summary = influential_df.groupby("excluded_channel")["influential"].agg(["sum", "count"])
    summary.columns = ["influential_count", "total_tests"]
    summary["influential_ratio"] = summary["influential_count"] / summary["total_tests"]
    print("\n[채널별 영향력 요약]")
    print(summary.sort_values("influential_count", ascending=False).to_string())
    return influential_df


def main() -> None:
    section("0. 데이터 로드")
    seg_corrected = pd.read_csv(SEG_CORRECTED_PATH)
    reliable = seg_corrected[
        seg_corrected["reliability_tier"].isin(["high", "medium"])
        & seg_corrected["channel"].isin(FINAL_CHANNELS)
    ].copy()

    loo_effect_df = run_loo_effect_size(reliable)
    loo_frac_df = run_loo_frac_significant(reliable)
    flag_influential_channels(loo_effect_df, loo_frac_df)


if __name__ == "__main__":
    main()
