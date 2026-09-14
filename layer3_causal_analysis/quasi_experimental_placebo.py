"""
layer3_causal_analysis/quasi_experimental_placebo.py
=============================================================================
Quasi-experimental control-group design: anomalous segments vs. a same-channel
placebo pool of normal-operation segments (README §Layer 3 (c), Fig. 3a).
This is the single most decisive test in the paper.

배경
-----------------------------------------------------------------------------
지금까지(onset_mc_propagation.py §2-2, temporal_precedence_test.py §2-3,
supplementary_diagnostics_A_D.py)는 전부 "같은 이상 세그먼트 내부"에서 onset
전(pre) vs 후(post)를 비교했다. 이건 대조군이 없는 비교라서, "이 정도 변화가
이상 신호 특유의 것인지, 아니면 같은 채널의 아무 세그먼트나 가져다 비슷하게
반으로 잘라도 원래 그 정도는 자연스럽게 흔들리는 것인지"를 구분하지 못한다.

이 스크립트는 각 이상 세그먼트의 pre/post 효과크기를, **같은 채널의 정상
세그먼트들을 무작위 시점에서 갈라 계산한 "위약(placebo)" 효과크기 분포**와
비교한다. 정상 세그먼트에는 실제 onset이 없으므로, 이상 세그먼트들의 "onset
위치비율(onset_idx/n_points)" 분포에서 pivot 비율을 뽑아 그 위치에서 정상
세그먼트를 가르는 방식(supplementary_diagnostics_A_D.py [C]와 같은 발상)을
그대로 재사용하되, 이번엔 세그먼트당 여러 번(K회) 뽑아 채널별로 충분히 큰
"정상 변동 폭 풀"을 만든다.

각 이상 세그먼트의 관측 효과크기가 이 위약 풀 대비 상위 몇 %에 해당하는지
(경험적 p-value)를 계산하고, 채널·피처 단위로 Mann-Whitney U 검정까지
수행한다. 이걸로 "이상 세그먼트의 변화가 정상적인 세그먼트 간 변동 폭으로는
설명되지 않는다"는 훨씬 강한 근거를 만드는 것이 목적이다.

주의: 이것도 여전히 "준실험적(quasi-experimental)" 비교다. 무작위 배정(RCT)이
아니므로 완벽한 인과 증명은 아니며, "정상적 변동보다 유의하게 크다"는 근거를
제공하는 층3의 한 축일 뿐이다.

입력 (재계산 없이 최대한 재사용):
  - segments.csv
  - stage2_mc_draws_raw.csv                  (canonical onset 산출용)
  - stage2_mc_segment_summary_corrected.csv  (①②보정, tier 포함)

출력:
  - stage3_qexp_placebo_pool.csv           (채널×피처별 위약 풀 원자료)
  - stage3_qexp_segment_pvalues.csv        (이상 세그먼트별 경험적 p-value)
  - stage3_qexp_channel_summary.csv        (채널별 유의 비율 요약)
  - stage3_qexp_mannwhitney.csv            (채널×피처 Mann-Whitney U 검정)
  - placebo_comparison.csv                 (README results/layer3 명명과 맞춘 사본)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _shared import (
    SEGMENTS_PATH, OUT_DIR, FINAL_CHANNELS, BURN_IN, MIN_WINDOW_POINTS,
    RANDOM_STATE, section,
)

DRAWS_PATH = OUT_DIR / "stage2_mc_draws_raw.csv"
SEG_CORRECTED_PATH = OUT_DIR / "stage2_mc_segment_summary_corrected.csv"

K_PLACEBO_DRAWS_PER_NORMAL = 5      # 정상 세그먼트 1개당 무작위 pivot 추출 횟수
MAX_NORMAL_PER_CHANNEL = 300         # 계산량 관리용 상한
FEATURES = ["level_d_abs", "diff_logvar", "diff2_logvar"]

rng = np.random.default_rng(RANDOM_STATE)


# =============================================================================
# 효과크기 계산 함수
# =============================================================================

def cohens_d_abs(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt((va + vb) / 2.0)
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float(abs(np.mean(a) - np.mean(b)) / pooled_sd)


def log_var_ratio(pre: np.ndarray, post: np.ndarray) -> float:
    """post 구간 변동성이 pre 대비 얼마나 커졌는지: log(var_post/var_pre).
    0이면 변화 없음, 양수면 post에서 변동성 증가."""
    if len(pre) < MIN_WINDOW_POINTS or len(post) < MIN_WINDOW_POINTS:
        return np.nan
    vpre, vpost = np.var(pre, ddof=1), np.var(post, ddof=1)
    if vpre <= 0 or vpost <= 0 or not np.isfinite(vpre) or not np.isfinite(vpost):
        return np.nan
    return float(np.log(vpost / vpre))


def compute_effects(values: np.ndarray, pivot_idx: int) -> dict:
    """pivot_idx를 기준으로 level(|d|)과 diff/diff2(log 분산비)를 계산."""
    out = {"level_d_abs": np.nan, "diff_logvar": np.nan, "diff2_logvar": np.nan}

    pre_level = values[BURN_IN:pivot_idx]
    post_level = values[pivot_idx:]
    out["level_d_abs"] = cohens_d_abs(post_level, pre_level)

    diff_all = np.diff(values)
    split1 = max(pivot_idx - 1, 0)
    pre_diff = diff_all[max(BURN_IN - 1, 0):split1]
    post_diff = diff_all[split1:]
    out["diff_logvar"] = log_var_ratio(pre_diff, post_diff)

    diff2_all = np.diff(diff_all)
    split2 = max(pivot_idx - 2, 0)
    pre_diff2 = diff2_all[max(BURN_IN - 2, 0):split2]
    post_diff2 = diff2_all[split2:]
    out["diff2_logvar"] = log_var_ratio(pre_diff2, post_diff2)

    return out


# =============================================================================
# 파이프라인
# =============================================================================

def build_canonical_onset(seg: pd.DataFrame, seg_filter: pd.DataFrame | None = None) -> pd.DataFrame:
    """§2-2 draw posterior의 중앙값을 canonical onset으로 삼는다.
    seg_filter가 주어지면(예: train=1 세그먼트) 그 (channel, segment)만 포함한다."""
    draws = pd.read_csv(DRAWS_PATH)
    seg_corrected = pd.read_csv(SEG_CORRECTED_PATH)

    reliable_ids = seg_corrected.loc[
        seg_corrected["reliability_tier"].isin(["high", "medium"])
        & seg_corrected["channel"].isin(FINAL_CHANNELS),
        ["channel", "segment"]
    ]
    if seg_filter is not None:
        reliable_ids = reliable_ids.merge(seg_filter, on=["channel", "segment"], how="inner")

    canonical_onset = (
        draws.merge(reliable_ids, on=["channel", "segment"], how="inner")
        .groupby(["channel", "segment"])["onset_idx_sampled"]
        .median().round().astype(int).rename("canonical_onset_idx").reset_index()
    )
    return canonical_onset


def compute_observed_effects(seg: pd.DataFrame, canonical_onset: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """이상 세그먼트의 관측 효과크기(canonical onset 기준)를 계산하고,
    채널별 onset 위치비율 풀도 함께 반환한다(위약 pivot 추출용)."""
    observed_rows = []
    onset_ratio_pool_by_channel = {ch: [] for ch in FINAL_CHANNELS}

    for _, row in canonical_onset.iterrows():
        ch, sid, onset_idx = row["channel"], int(row["segment"]), int(row["canonical_onset_idx"])
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 1)]
        if len(s) == 0:
            continue
        s = s.sort_values("timestamp")
        values = s["value"].values
        n = len(values)

        if onset_idx < BURN_IN + MIN_WINDOW_POINTS or onset_idx > n - MIN_WINDOW_POINTS:
            continue

        eff = compute_effects(values, onset_idx)
        observed_rows.append({"channel": ch, "segment": sid, "n_points": n, "onset_idx": onset_idx, **eff})
        onset_ratio_pool_by_channel[ch].append(onset_idx / n)

    return pd.DataFrame(observed_rows), onset_ratio_pool_by_channel


def build_placebo_pool(seg: pd.DataFrame, onset_ratio_pool_by_channel: dict) -> pd.DataFrame:
    """채널별로 정상 세그먼트를 무작위 pivot(이상 세그먼트 onset 위치비율 분포에서
    추출)으로 갈라 위약 효과크기 풀을 만든다."""
    placebo_rows = []

    for ch in FINAL_CHANNELS:
        ratio_pool = np.array(onset_ratio_pool_by_channel[ch])
        if len(ratio_pool) == 0:
            print(f"[{ch}] onset 위치비율 풀이 비어 있어 스킵")
            continue

        normal_meta = seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment").size().rename("n_points").reset_index()
        if len(normal_meta) > MAX_NORMAL_PER_CHANNEL:
            normal_meta = normal_meta.sample(n=MAX_NORMAL_PER_CHANNEL, random_state=RANDOM_STATE)

        n_generated = 0
        for _, nrow in normal_meta.iterrows():
            sid, n = int(nrow["segment"]), int(nrow["n_points"])
            s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 0)].sort_values("timestamp")
            values = s["value"].values

            for _ in range(K_PLACEBO_DRAWS_PER_NORMAL):
                ratio = rng.choice(ratio_pool)
                pivot_idx = int(round(ratio * n))
                pivot_idx = int(np.clip(pivot_idx, BURN_IN + MIN_WINDOW_POINTS, max(n - MIN_WINDOW_POINTS, BURN_IN + MIN_WINDOW_POINTS)))
                if pivot_idx < BURN_IN + MIN_WINDOW_POINTS or pivot_idx > n - MIN_WINDOW_POINTS:
                    continue
                eff = compute_effects(values, pivot_idx)
                placebo_rows.append({"channel": ch, "segment": sid, "pivot_idx": pivot_idx, **eff})
                n_generated += 1

        print(f"[{ch}] 정상 세그먼트 {len(normal_meta)}개 -> 위약 표본 {n_generated}개 생성")

    return pd.DataFrame(placebo_rows)


def compute_empirical_pvalues(observed_df: pd.DataFrame, placebo_df: pd.DataFrame) -> pd.DataFrame:
    """이상 세그먼트별로, 위약 풀 대비 관측 효과크기의 경험적 단측 p-value를 계산."""
    pval_rows = []
    for _, row in observed_df.iterrows():
        ch = row["channel"]
        placebo_ch = placebo_df[placebo_df["channel"] == ch]
        out_row = {"channel": ch, "segment": row["segment"], "n_points": row["n_points"]}

        for feat in FEATURES:
            obs_val = row[feat]
            placebo_vals = placebo_ch[feat].dropna().values
            if not np.isfinite(obs_val) or len(placebo_vals) < 20:
                out_row[f"{feat}_pvalue"] = np.nan
                out_row[f"{feat}_placebo_n"] = len(placebo_vals)
                continue
            p_emp = (np.sum(placebo_vals >= obs_val) + 1) / (len(placebo_vals) + 1)
            out_row[f"{feat}_pvalue"] = float(p_emp)
            out_row[f"{feat}_placebo_n"] = len(placebo_vals)
            out_row[f"{feat}_observed"] = float(obs_val)
            out_row[f"{feat}_placebo_median"] = float(np.median(placebo_vals))
            out_row[f"{feat}_placebo_p95"] = float(np.percentile(placebo_vals, 95))

        pval_rows.append(out_row)

    return pd.DataFrame(pval_rows)


def summarize_channel_significance(pval_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for ch in FINAL_CHANNELS:
        sub = pval_df[pval_df["channel"] == ch]
        row = {"channel": ch, "n_segments": len(sub)}
        for feat in FEATURES:
            pcol = f"{feat}_pvalue"
            p_vals = sub[pcol].dropna().values
            row[f"{feat}_frac_significant"] = float(np.mean(p_vals < 0.05)) if len(p_vals) else np.nan
            row[f"{feat}_n_valid"] = len(p_vals)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def run_mannwhitney(observed_df: pd.DataFrame, placebo_df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U (alternative='greater'): 이상 세그먼트 효과크기 분포가
    위약 풀 분포보다 체계적으로 더 큰지 채널×피처 단위로 검정."""
    mw_rows = []
    for ch in FINAL_CHANNELS:
        obs_ch = observed_df[observed_df["channel"] == ch]
        placebo_ch = placebo_df[placebo_df["channel"] == ch]
        for feat in FEATURES:
            a = obs_ch[feat].dropna().values
            b = placebo_ch[feat].dropna().values
            if len(a) < 5 or len(b) < 5:
                mw_rows.append({
                    "channel": ch, "feature": feat, "n_anomaly": len(a), "n_placebo": len(b),
                    "u_stat": np.nan, "p_value": np.nan, "median_anomaly": np.nan, "median_placebo": np.nan,
                    "note": "표본부족 (n<5)",
                })
                continue
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="greater")
            mw_rows.append({
                "channel": ch, "feature": feat, "n_anomaly": len(a), "n_placebo": len(b),
                "u_stat": float(u_stat), "p_value": float(p_val),
                "median_anomaly": float(np.median(a)), "median_placebo": float(np.median(b)),
                "note": "",
            })

    mw_df = pd.DataFrame(mw_rows)
    n_tests = int(mw_df["p_value"].notna().sum())
    mw_df["p_value_bonferroni"] = (mw_df["p_value"] * max(n_tests, 1)).clip(upper=1.0)
    return mw_df


def main() -> None:
    section("0. 데이터 로드 및 canonical onset 재구성 (§2-2/§2-3와 동일 절차 재사용)")

    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

    canonical_onset = build_canonical_onset(seg)
    print(f"신뢰도 high+medium 이상 세그먼트: {len(canonical_onset)}개")

    section("1. 이상 세그먼트 관측 효과크기 계산")
    observed_df, onset_ratio_pool_by_channel = compute_observed_effects(seg, canonical_onset)
    print(f"관측 효과크기 계산된 이상 세그먼트: {len(observed_df)}개")
    print(observed_df.groupby("channel").size().reindex(FINAL_CHANNELS).rename("n_segments").to_string())

    section(f"2. 채널별 위약(placebo) 풀 생성 (정상 세그먼트당 {K_PLACEBO_DRAWS_PER_NORMAL}회 무작위 pivot)")
    placebo_df = build_placebo_pool(seg, onset_ratio_pool_by_channel)
    placebo_df.to_csv(OUT_DIR / "stage3_qexp_placebo_pool.csv", index=False)
    print(f"\n[저장] stage3_qexp_placebo_pool.csv  ({len(placebo_df)} rows)")

    section("3. 이상 세그먼트별 경험적 p-value 계산 (위약 풀 대비 상위 % 위치)")
    pval_df = compute_empirical_pvalues(observed_df, placebo_df)
    pval_df.to_csv(OUT_DIR / "stage3_qexp_segment_pvalues.csv", index=False)
    print(f"[저장] stage3_qexp_segment_pvalues.csv  ({len(pval_df)} rows)")

    section("4. 채널별 요약: 위약 풀 대비 유의미(p<0.05)한 이상 세그먼트 비율")
    channel_summary_df = summarize_channel_significance(pval_df)
    channel_summary_df.to_csv(OUT_DIR / "stage3_qexp_channel_summary.csv", index=False)
    print(channel_summary_df.to_string(index=False))

    section("5. Mann-Whitney U 검정: 이상 세그먼트 효과크기 분포 vs 위약 풀 분포 (채널별)")
    mw_df = run_mannwhitney(observed_df, placebo_df)
    mw_df.to_csv(OUT_DIR / "stage3_qexp_mannwhitney.csv", index=False)
    print(mw_df.to_string(index=False))

    # README results/layer3 명명과 맞춘 사본 (Fig. 3a 소스)
    mw_df.to_csv(OUT_DIR / "placebo_comparison.csv", index=False)

    section("완료 — 해석 가이드")
    print(
        "1) segment_pvalues의 '_pvalue'는 '이 이상 세그먼트의 변화가 같은 채널 정상\n"
        "   세그먼트들의 자연스러운 변동 폭보다 얼마나 극단적인가'를 나타내는 경험적\n"
        "   단측 p-value다. 낮을수록(0에 가까울수록) 정상적 변동으로는 설명 안 되는\n"
        "   변화라는 뜻이다.\n"
        "2) Mann-Whitney 결과의 p_value_bonferroni를 주 지표로 쓴다(15개 검정=5채널\n"
        "   x3피처 다중비교 보정). median_anomaly가 median_placebo보다 뚜렷이 크면서\n"
        "   p_value_bonferroni가 유의하면, 채널 전체 수준에서 '이상 세그먼트 그룹이\n"
        "   정상 세그먼트 그룹보다 체계적으로 더 크게 움직인다'는 근거다.\n"
        "3) 이 결과 역시 무작위 배정 실험이 아닌 준실험(quasi-experimental) 설계이므로,\n"
        "   '정상적 변동보다 유의하게 크다'까지만 주장하고 완전한 인과 증명으로\n"
        "   과장하지 않는다. scm_skeleton.py에서 이 결과를 SCM 뼈대 확정의 결정적\n"
        "   근거로 사용한다."
    )


if __name__ == "__main__":
    main()
