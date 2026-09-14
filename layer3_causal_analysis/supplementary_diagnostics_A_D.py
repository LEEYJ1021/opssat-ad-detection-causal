"""
layer3_causal_analysis/supplementary_diagnostics_A_D.py
=============================================================================
Supplementary diagnostics ruling out that the diff/diff2-precedes-level
result is a labeling or detector artifact (README §Layer 3 (e); "Threats to
validity" table).

네 개의 독립적인 진단을 하나로 묶는다:

  [A] Persistence classification — onset 이후 |z| 궤적이 피크를 찍고
      가라앉는지(transient) 끝까지 유지되는지(persistent)를 채널별로 분류.
  [B] Early-window re-test — "post 구간 전체 평균" Welch t-test 대신
      "post 구간 앞부분 EARLY_WINDOW_POINTS(기본 10포인트)"만으로 재검정.
  [C] Normal-segment control — temporal_precedence_test.py와 동일한 절차를
      anomaly=0 정상 세그먼트에도 적용(pivot은 이상 세그먼트들의 onset 위치비율
      분포에서 샘플링).
  [D] Group-difference significance — [C]에서 나온 "이상군 vs 정상군" 선행성
      비율 차이를 정식으로 검정: 풀링 2비율 z-검정 + Fisher's exact test,
      채널별 Fisher's exact test, CMH 검정. 이어서 diff vs diff2의 초 단위
      동률(tie) 문제를 우회하기 위해 교차상관 기반 연속 lag 추정을 추가.

출력:
  - stage2_diagA_diff_temporal_profile.csv
  - stage2_diagB_early_window_mc.csv
  - stage2_diagC_normal_control_crossing_times.csv
  - stage2_diagC_precedence_test_anomaly_vs_normal.csv
  - stage2_diagD_group_proportion_tests_pooled.csv
  - stage2_diagD_group_proportion_tests_by_channel.csv
  - stage2_diagD_group_proportion_tests_cmh.csv
  - stage2_diagD_diff_diff2_lag_segments.csv
  - stage2_diagD_lag_group_comparison.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _shared import (
    SEGMENTS_PATH, OUT_DIR, FINAL_CHANNELS, BURN_IN,
    MIN_WINDOW_POINTS, RANDOM_STATE, section, LocalLinearTrendKF,
)

DRAWS_PATH = OUT_DIR / "stage2_mc_draws_raw.csv"
SEG_CORRECTED_PATH = OUT_DIR / "stage2_mc_segment_summary_corrected.csv"
ANOMALY_CROSSING_PATH = OUT_DIR / "stage2_precedence_crossing_times.csv"

Z_THRESHOLD = 3.0
DECAY_LOOKAHEAD = 60
DECAY_RATIO_THRESHOLD = 0.5
EARLY_WINDOW_POINTS = 10
MAX_NORMAL_PER_CHANNEL = 200
MAX_LAG = 15
MIN_OVERLAP = 5
CORR_THRESHOLD = 0.3
POST_WINDOW = 40
PAIRS = [("level", "diff"), ("level", "diff2"), ("diff", "diff2")]

_kf_cache: dict[str, LocalLinearTrendKF] = {}
rng = np.random.default_rng(RANDOM_STATE)


def get_kf(ch: str) -> LocalLinearTrendKF:
    """캐시된 채널별 Kalman filter를 반환한다. main(kf_by_channel=...)로
    onset_mc_propagation.py가 적합한 KF를 주입받는 것을 권장."""
    return _kf_cache[ch]


# =============================================================================
# [A] Persistence classification (transient vs persistent)
# =============================================================================

def classify_decay(z_post: np.ndarray) -> dict:
    az = np.abs(z_post[:DECAY_LOOKAHEAD])
    if len(az) < 3:
        return {"pattern": "판정불가(구간부족)", "peak_idx": np.nan, "decay_idx": np.nan, "peak_abs_z": np.nan}
    peak_idx = int(np.argmax(az))
    peak_val = float(az[peak_idx])
    if peak_val <= 0 or not np.isfinite(peak_val):
        return {"pattern": "판정불가(피크없음)", "peak_idx": peak_idx, "decay_idx": np.nan, "peak_abs_z": peak_val}
    threshold = peak_val * DECAY_RATIO_THRESHOLD
    after_peak = az[peak_idx:]
    below = np.where(after_peak <= threshold)[0]
    if len(below) > 0:
        decay_idx = peak_idx + int(below[0])
        return {"pattern": "transient(일시적)", "peak_idx": peak_idx, "decay_idx": decay_idx, "peak_abs_z": peak_val}
    return {"pattern": "persistent(지속적)", "peak_idx": peak_idx, "decay_idx": np.nan, "peak_abs_z": peak_val}


def run_diag_A(seg: pd.DataFrame, canonical_onset: pd.DataFrame) -> pd.DataFrame:
    section("[A] diff/diff2 시간 프로파일 진단: onset 이후 |z|가 가라앉는가, 유지되는가")
    rows = []
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

        kf = get_kf(ch)
        innovations, innovation_vars, _ = kf.run(values)
        z_level = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))

        diff_all = np.diff(values)
        diff2_all = np.diff(diff_all)

        def zscore_post(raw: np.ndarray, pre_end: int, onset_pos: int) -> np.ndarray:
            pre = raw[max(BURN_IN, 0):pre_end]
            if len(pre) < MIN_WINDOW_POINTS:
                return np.array([])
            mu, sd = np.mean(pre), np.std(pre, ddof=1)
            if sd <= 0 or not np.isfinite(sd):
                return np.array([])
            return (raw[onset_pos:] - mu) / sd

        z_level_post = zscore_post(values, onset_idx, onset_idx)
        diff_pre_end = max(onset_idx - 1, 0)
        z_diff_post = zscore_post(diff_all, diff_pre_end, diff_pre_end)
        diff2_pre_end = max(onset_idx - 2, 0)
        z_diff2_post = zscore_post(diff2_all, diff2_pre_end, diff2_pre_end)

        row_out = {"channel": ch, "segment": sid, "n_points": n, "onset_idx": onset_idx}
        for name, zpost in [("level", z_level_post), ("diff", z_diff_post), ("diff2", z_diff2_post)]:
            cls = classify_decay(zpost)
            row_out[f"{name}_pattern"] = cls["pattern"]
            row_out[f"{name}_peak_abs_z"] = cls["peak_abs_z"]
            row_out[f"{name}_decay_idx"] = cls["decay_idx"]
        rows.append(row_out)

    diagA_df = pd.DataFrame(rows)
    diagA_df.to_csv(OUT_DIR / "stage2_diagA_diff_temporal_profile.csv", index=False)
    for name in ["level", "diff", "diff2"]:
        print(f"\n--- {name} pattern ---")
        print(diagA_df[f"{name}_pattern"].value_counts().to_string())
    return diagA_df


# =============================================================================
# [B] Early-window re-test
# =============================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt((va + vb) / 2.0)
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def welch_early_window(pre: np.ndarray, post_full: np.ndarray, window: int) -> dict:
    post_early = post_full[:window]
    out = {}
    if len(pre) >= MIN_WINDOW_POINTS and len(post_early) >= MIN_WINDOW_POINTS:
        _, p = stats.ttest_ind(post_early, pre, equal_var=False)
        out["p_early"], out["d_early"] = float(p), cohens_d(post_early, pre)
    else:
        out["p_early"] = out["d_early"] = np.nan
    if len(pre) >= MIN_WINDOW_POINTS and len(post_full) >= MIN_WINDOW_POINTS:
        _, p = stats.ttest_ind(post_full, pre, equal_var=False)
        out["p_full"], out["d_full"] = float(p), cohens_d(post_full, pre)
    else:
        out["p_full"] = out["d_full"] = np.nan
    return out


def run_diag_B(seg: pd.DataFrame, canonical_onset: pd.DataFrame) -> pd.DataFrame:
    section(f"[B] 조기창(early window={EARLY_WINDOW_POINTS}pt) 재검정: post 구간 전체 대신 앞부분만 비교")
    rows = []
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

        row_out = {"channel": ch, "segment": sid, "n_points": n}
        pre_level = values[BURN_IN:onset_idx]
        post_level = values[onset_idx:]
        for k, v in welch_early_window(pre_level, post_level, EARLY_WINDOW_POINTS).items():
            row_out[f"level_{k}"] = v

        diff_all = np.diff(values)
        split1 = max(onset_idx - 1, 0)
        pre_diff_sq = diff_all[max(BURN_IN - 1, 0):split1] ** 2
        post_diff_sq = diff_all[split1:] ** 2
        for k, v in welch_early_window(pre_diff_sq, post_diff_sq, EARLY_WINDOW_POINTS).items():
            row_out[f"diff_{k}"] = v

        diff2_all = np.diff(diff_all)
        split2 = max(onset_idx - 2, 0)
        pre_diff2_sq = diff2_all[max(BURN_IN - 2, 0):split2] ** 2
        post_diff2_sq = diff2_all[split2:] ** 2
        for k, v in welch_early_window(pre_diff2_sq, post_diff2_sq, EARLY_WINDOW_POINTS).items():
            row_out[f"diff2_{k}"] = v
        rows.append(row_out)

    diagB_df = pd.DataFrame(rows)
    diagB_df.to_csv(OUT_DIR / "stage2_diagB_early_window_mc.csv", index=False)

    summary_rows = []
    for name in ["level", "diff", "diff2"]:
        p_early = diagB_df[f"{name}_p_early"].dropna().values
        p_full = diagB_df[f"{name}_p_full"].dropna().values
        summary_rows.append({
            "variable": name,
            "n_early": len(p_early), "frac_sig_early": float(np.mean(p_early < 0.05)) if len(p_early) else np.nan,
            "n_full": len(p_full), "frac_sig_full": float(np.mean(p_full < 0.05)) if len(p_full) else np.nan,
        })
    summary_B = pd.DataFrame(summary_rows)
    summary_B["gain(early-full)"] = summary_B["frac_sig_early"] - summary_B["frac_sig_full"]
    print(summary_B.to_string(index=False))
    return diagB_df


# =============================================================================
# [C] Normal-segment control (placebo pivot from anomaly onset-ratio distribution)
# =============================================================================

def analyze_segment_precedence_pivot(values: np.ndarray, sampling: float, pivot_idx: int) -> dict:
    result = {"level_cross_sec": np.nan, "diff_cross_sec": np.nan, "diff2_cross_sec": np.nan}

    pre_level = values[BURN_IN:pivot_idx]
    post_level = values[pivot_idx:]
    if len(pre_level) >= MIN_WINDOW_POINTS and len(post_level) >= MIN_WINDOW_POINTS:
        mean_pre, std_pre = np.mean(pre_level), np.std(pre_level, ddof=1)
        if std_pre > 0 and np.isfinite(std_pre):
            z = np.abs(post_level - mean_pre) / std_pre
            hit = np.where(z > Z_THRESHOLD)[0]
            if len(hit) > 0:
                result["level_cross_sec"] = float((pivot_idx + hit[0] - pivot_idx) * sampling)

    diff_all = np.diff(values)
    diff_pre_end = max(pivot_idx - 1, 0)
    diff_pre = diff_all[max(BURN_IN - 1, 0):diff_pre_end]
    diff_post = diff_all[diff_pre_end:]
    if len(diff_pre) >= MIN_WINDOW_POINTS and len(diff_post) >= MIN_WINDOW_POINTS:
        mean_pre, std_pre = np.mean(diff_pre), np.std(diff_pre, ddof=1)
        if std_pre > 0 and np.isfinite(std_pre):
            z = np.abs(diff_post - mean_pre) / std_pre
            hit = np.where(z > Z_THRESHOLD)[0]
            if len(hit) > 0:
                k = diff_pre_end + hit[0]
                result["diff_cross_sec"] = float((k + 1 - pivot_idx) * sampling)

    diff2_all = np.diff(diff_all)
    diff2_pre_end = max(pivot_idx - 2, 0)
    diff2_pre = diff2_all[max(BURN_IN - 2, 0):diff2_pre_end]
    diff2_post = diff2_all[diff2_pre_end:]
    if len(diff2_pre) >= MIN_WINDOW_POINTS and len(diff2_post) >= MIN_WINDOW_POINTS:
        mean_pre, std_pre = np.mean(diff2_pre), np.std(diff2_pre, ddof=1)
        if std_pre > 0 and np.isfinite(std_pre):
            z = np.abs(diff2_post - mean_pre) / std_pre
            hit = np.where(z > Z_THRESHOLD)[0]
            if len(hit) > 0:
                m = diff2_pre_end + hit[0]
                result["diff2_cross_sec"] = float((m + 2 - pivot_idx) * sampling)

    return result


def run_pairwise_test(df: pd.DataFrame, a: str, b: str) -> dict:
    ta, tb = df[f"{a}_cross_sec"].values, df[f"{b}_cross_sec"].values
    mask = ~np.isnan(ta) & ~np.isnan(tb)
    n_pairs = int(mask.sum())
    base = {"pair": f"{a} vs {b}", "n_pairs": n_pairs, "median_diff_sec": np.nan,
            "wilcoxon_stat": np.nan, "p_value": np.nan, f"frac_{a}_precedes_{b}": np.nan, "note": ""}
    if n_pairs < 5:
        base["note"] = "표본부족 (n<5)"
        return base
    diff_time = ta[mask] - tb[mask]
    nonzero = diff_time[diff_time != 0]
    if len(nonzero) < 5:
        base["note"] = "동률 제외 후 표본부족 (n<5)"
        base["median_diff_sec"] = float(np.median(diff_time))
        return base
    stat, p = stats.wilcoxon(nonzero)
    base.update(median_diff_sec=float(np.median(diff_time)), wilcoxon_stat=float(stat), p_value=float(p))
    base[f"frac_{a}_precedes_{b}"] = float(np.mean(diff_time < 0))
    return base


def run_diag_C(seg: pd.DataFrame, canonical_onset: pd.DataFrame, precedence_df: pd.DataFrame) -> pd.DataFrame:
    section("[C] 정상 세그먼트 대조검정: diff-먼저 패턴이 이상 고유의 특성인가, 차분 연산의 일반적 아티팩트인가")

    onset_ratio = canonical_onset.merge(
        seg[["channel", "segment"]].drop_duplicates(), on=["channel", "segment"], how="inner"
    )
    n_points_map = seg.groupby(["channel", "segment"]).size().rename("n_points_actual")
    onset_ratio = onset_ratio.merge(n_points_map, on=["channel", "segment"], how="left")
    onset_ratio["ratio"] = onset_ratio["canonical_onset_idx"] / onset_ratio["n_points_actual"]
    ratio_pool = onset_ratio["ratio"].dropna().values
    print(f"이상 세그먼트 onset 위치비율 분포: mean={ratio_pool.mean():.3f}, median={np.median(ratio_pool):.3f}")

    normal_seg_meta = seg[seg["anomaly"] == 0].groupby(["channel", "segment"]).size().rename("n_points").reset_index()
    sampled_normal_rows = []
    for ch in FINAL_CHANNELS:
        sub = normal_seg_meta[normal_seg_meta["channel"] == ch]
        if len(sub) > MAX_NORMAL_PER_CHANNEL:
            sub = sub.sample(n=MAX_NORMAL_PER_CHANNEL, random_state=RANDOM_STATE)
        sampled_normal_rows.append(sub)
    sampled_normal = pd.concat(sampled_normal_rows, ignore_index=True)
    print(f"실제 대조검정 대상: {len(sampled_normal)}개 정상 세그먼트")

    rows = []
    for _, row in sampled_normal.iterrows():
        ch, sid, n = row["channel"], int(row["segment"]), int(row["n_points"])
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 0)].sort_values("timestamp")
        values = s["value"].values
        sampling = float(s["sampling"].iloc[0])
        pivot_ratio = rng.choice(ratio_pool)
        pivot_idx = int(round(pivot_ratio * n))
        pivot_idx = int(np.clip(pivot_idx, BURN_IN + MIN_WINDOW_POINTS, max(n - MIN_WINDOW_POINTS, BURN_IN + MIN_WINDOW_POINTS)))
        if pivot_idx < BURN_IN + MIN_WINDOW_POINTS or pivot_idx > n - MIN_WINDOW_POINTS:
            rows.append({"channel": ch, "segment": sid, "n_points": n, "pivot_idx": pivot_idx,
                         "skip_reason": "pivot이 경계에 가까워 pre/post 확보 불가",
                         "level_cross_sec": np.nan, "diff_cross_sec": np.nan, "diff2_cross_sec": np.nan})
            continue
        cross = analyze_segment_precedence_pivot(values, sampling, pivot_idx)
        rows.append({"channel": ch, "segment": sid, "n_points": n, "pivot_idx": pivot_idx, "skip_reason": "", **cross})

    diagC_df = pd.DataFrame(rows)
    diagC_df.to_csv(OUT_DIR / "stage2_diagC_normal_control_crossing_times.csv", index=False)

    valid_normal = diagC_df[diagC_df["skip_reason"] == ""]
    valid_anomaly = precedence_df[precedence_df["skip_reason"] == ""]
    normal_results = [dict(run_pairwise_test(valid_normal, a, b), group="normal") for a, b in PAIRS]
    anomaly_results = [dict(run_pairwise_test(valid_anomaly, a, b), group="anomaly") for a, b in PAIRS]
    compare_df = pd.DataFrame(normal_results + anomaly_results)
    compare_df.to_csv(OUT_DIR / "stage2_diagC_precedence_test_anomaly_vs_normal.csv", index=False)
    print(compare_df.to_string(index=False))
    return diagC_df


# =============================================================================
# [D] Group-difference significance: proportion tests + CMH + cross-corr lag
# =============================================================================

def get_precedence_indicator(df: pd.DataFrame, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    ta, tb = df[f"{a}_cross_sec"].values, df[f"{b}_cross_sec"].values
    mask = ~np.isnan(ta) & ~np.isnan(tb)
    indicator = (ta[mask] - tb[mask] < 0).astype(int)
    return indicator, df["channel"].values[mask]


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def cmh_test(tables: list[tuple[int, int, int, int]]) -> dict:
    num_sum, var_sum, or_num, or_den, used_strata = 0.0, 0.0, 0.0, 0.0, 0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        used_strata += 1
        e_a = (a + b) * (a + c) / n
        var_a = ((a + b) * (c + d) * (a + c) * (b + d)) / (n ** 2 * (n - 1)) if n > 1 else 0.0
        num_sum += (a - e_a)
        var_sum += var_a
        or_num += (a * d) / n
        or_den += (b * c) / n
    if var_sum == 0 or used_strata == 0:
        return {"chi2": np.nan, "p_value": np.nan, "or_mh": np.nan, "n_strata": used_strata}
    chi2 = max((abs(num_sum) - 0.5) ** 2 / var_sum, 0.0)
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    or_mh = float(or_num / or_den) if or_den > 0 else np.nan
    return {"chi2": float(chi2), "p_value": p_value, "or_mh": or_mh, "n_strata": used_strata}


def run_diag_D_proportions(anomaly_valid: pd.DataFrame, normal_valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section("[D-1a] 풀링 기준 2비율 z-검정 + Fisher's exact test")
    pooled_rows = []
    for a, b in PAIRS:
        ind_anom, _ = get_precedence_indicator(anomaly_valid, a, b)
        ind_norm, _ = get_precedence_indicator(normal_valid, a, b)
        x1, n1, x2, n2 = int(ind_anom.sum()), len(ind_anom), int(ind_norm.sum()), len(ind_norm)
        z, p_z = two_proportion_ztest(x1, n1, x2, n2)
        odds_ratio, p_fisher = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
        pooled_rows.append({"pair": f"{a} precedes {b}", "anomaly_frac": x1 / n1 if n1 else np.nan,
                             "anomaly_n": n1, "normal_frac": x2 / n2 if n2 else np.nan, "normal_n": n2,
                             "z_stat": z, "p_ztest": p_z, "odds_ratio": float(odds_ratio), "p_fisher": float(p_fisher)})
    pooled_df = pd.DataFrame(pooled_rows)
    n_tests = len(pooled_df)
    pooled_df["p_ztest_bonferroni"] = (pooled_df["p_ztest"] * n_tests).clip(upper=1.0)
    pooled_df["p_fisher_bonferroni"] = (pooled_df["p_fisher"] * n_tests).clip(upper=1.0)
    pooled_df.to_csv(OUT_DIR / "stage2_diagD_group_proportion_tests_pooled.csv", index=False)
    print(pooled_df.to_string(index=False))

    section("[D-1b] 채널별 Fisher's exact test")
    by_channel_rows = []
    for a, b in PAIRS:
        for ch in FINAL_CHANNELS:
            ind_anom, _ = get_precedence_indicator(anomaly_valid[anomaly_valid["channel"] == ch], a, b)
            ind_norm, _ = get_precedence_indicator(normal_valid[normal_valid["channel"] == ch], a, b)
            x1, n1, x2, n2 = int(ind_anom.sum()), len(ind_anom), int(ind_norm.sum()), len(ind_norm)
            if n1 == 0 or n2 == 0:
                by_channel_rows.append({"pair": f"{a} precedes {b}", "channel": ch, "anomaly_frac": np.nan,
                                         "anomaly_n": n1, "normal_frac": np.nan, "normal_n": n2,
                                         "odds_ratio": np.nan, "p_fisher": np.nan, "note": "표본 없음"})
                continue
            odds_ratio, p_fisher = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
            by_channel_rows.append({"pair": f"{a} precedes {b}", "channel": ch, "anomaly_frac": x1 / n1,
                                     "anomaly_n": n1, "normal_frac": x2 / n2, "normal_n": n2,
                                     "odds_ratio": float(odds_ratio), "p_fisher": float(p_fisher),
                                     "note": "" if min(n1, n2) >= 5 else "표본부족(n<5)"})
    by_channel_df = pd.DataFrame(by_channel_rows)
    by_channel_df.to_csv(OUT_DIR / "stage2_diagD_group_proportion_tests_by_channel.csv", index=False)
    print(by_channel_df.to_string(index=False))

    section("[D-1c] 채널 층화 Cochran-Mantel-Haenszel 검정")
    cmh_rows = []
    for a, b in PAIRS:
        tables = []
        for ch in FINAL_CHANNELS:
            ind_anom, _ = get_precedence_indicator(anomaly_valid[anomaly_valid["channel"] == ch], a, b)
            ind_norm, _ = get_precedence_indicator(normal_valid[normal_valid["channel"] == ch], a, b)
            a_cnt, n1, c_cnt, n2 = int(ind_anom.sum()), len(ind_anom), int(ind_norm.sum()), len(ind_norm)
            if n1 == 0 or n2 == 0:
                continue
            tables.append((a_cnt, n1 - a_cnt, c_cnt, n2 - c_cnt))
        result = cmh_test(tables)
        result["pair"] = f"{a} precedes {b}"
        cmh_rows.append(result)
    cmh_df = pd.DataFrame(cmh_rows)[["pair", "n_strata", "chi2", "p_value", "or_mh"]]
    cmh_df["p_value_bonferroni"] = (cmh_df["p_value"] * len(cmh_df)).clip(upper=1.0)
    cmh_df.to_csv(OUT_DIR / "stage2_diagD_group_proportion_tests_cmh.csv", index=False)
    print(cmh_df.to_string(index=False))
    return pooled_df, by_channel_df, cmh_df


def compute_z_diff_diff2_post(values: np.ndarray, pivot_idx: int) -> tuple[np.ndarray, np.ndarray]:
    diff_all = np.diff(values)
    diff_pre_end = max(pivot_idx - 1, 0)
    diff_pre = diff_all[max(BURN_IN - 1, 0):diff_pre_end]
    diff_post = diff_all[diff_pre_end:]

    diff2_all = np.diff(diff_all)
    diff2_pre_end = max(pivot_idx - 2, 0)
    diff2_pre = diff2_all[max(BURN_IN - 2, 0):diff2_pre_end]
    diff2_post = diff2_all[diff2_pre_end:]

    z_diff_post = np.array([])
    if len(diff_pre) >= MIN_WINDOW_POINTS:
        mu, sd = np.mean(diff_pre), np.std(diff_pre, ddof=1)
        if sd > 0 and np.isfinite(sd):
            z_diff_post = np.abs((diff_post - mu) / sd)

    z_diff2_post = np.array([])
    if len(diff2_pre) >= MIN_WINDOW_POINTS:
        mu, sd = np.mean(diff2_pre), np.std(diff2_pre, ddof=1)
        if sd > 0 and np.isfinite(sd):
            z_diff2_post = np.abs((diff2_post - mu) / sd)

    return z_diff_post[:POST_WINDOW], z_diff2_post[:POST_WINDOW]


def find_best_lag(a: np.ndarray, b: np.ndarray, max_lag: int, min_overlap: int) -> tuple[int, float]:
    n = min(len(a), len(b))
    if n < min_overlap:
        return 0, np.nan
    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a_seg, b_seg = a[: n - lag], b[lag:n]
        else:
            k = -lag
            a_seg, b_seg = a[k:n], b[: n - k]
        if len(a_seg) < min_overlap or len(a_seg) != len(b_seg):
            continue
        if np.std(a_seg) == 0 or np.std(b_seg) == 0:
            continue
        corr = np.corrcoef(a_seg, b_seg)[0, 1]
        if np.isfinite(corr) and abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, float(corr)
    return best_lag, best_corr


def run_diag_D_lag(seg: pd.DataFrame, anomaly_valid: pd.DataFrame, normal_valid: pd.DataFrame) -> pd.DataFrame:
    section("[D-2] diff vs diff2 교차상관 기반 lag 추정 (초 단위 동률 문제 우회)")

    anom_pivot_src = anomaly_valid[["channel", "segment", "onset_idx"]].rename(columns={"onset_idx": "pivot_idx"})
    anom_pivot_src["group"], anom_pivot_src["is_anomaly"] = "anomaly", 1
    norm_pivot_src = normal_valid[["channel", "segment", "pivot_idx"]].copy()
    norm_pivot_src["group"], norm_pivot_src["is_anomaly"] = "normal", 0
    all_pivots = pd.concat([anom_pivot_src, norm_pivot_src], ignore_index=True)

    rows = []
    for _, row in all_pivots.iterrows():
        ch, sid, pivot_idx, group, is_anom = (
            row["channel"], int(row["segment"]), int(row["pivot_idx"]), row["group"], int(row["is_anomaly"])
        )
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == is_anom)]
        if len(s) == 0:
            continue
        s = s.sort_values("timestamp")
        values = s["value"].values
        n = len(values)
        if pivot_idx < BURN_IN + MIN_WINDOW_POINTS or pivot_idx > n - MIN_WINDOW_POINTS:
            continue
        z_diff_post, z_diff2_post = compute_z_diff_diff2_post(values, pivot_idx)
        lag, corr = find_best_lag(z_diff_post, z_diff2_post, MAX_LAG, MIN_OVERLAP)
        rows.append({"channel": ch, "segment": sid, "group": group, "n_points": n,
                      "lag_diff_minus_diff2": lag, "best_corr": corr,
                      "reliable": bool(np.isfinite(corr) and abs(corr) >= CORR_THRESHOLD)})

    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(OUT_DIR / "stage2_diagD_diff_diff2_lag_segments.csv", index=False)
    print(lag_df.groupby("group")["reliable"].value_counts().to_string())

    section("[D-2b] lag 분포 검정: 그룹 내부(0 대비) + 그룹 간 비교")
    reliable_lag = lag_df[lag_df["reliable"]].copy()
    group_test_rows = []
    for group in ["anomaly", "normal"]:
        lags = reliable_lag.loc[reliable_lag["group"] == group, "lag_diff_minus_diff2"].values
        n = len(lags)
        if n >= 5 and np.any(lags != 0):
            stat, p = stats.wilcoxon(lags[lags != 0])
        else:
            stat, p = np.nan, np.nan
        group_test_rows.append({"test": f"{group}: lag vs 0 (Wilcoxon)", "n": n,
                                 "median_lag": float(np.median(lags)) if n else np.nan,
                                 "frac_diff_leads(lag>0)": float(np.mean(lags > 0)) if n else np.nan,
                                 "frac_diff2_leads(lag<0)": float(np.mean(lags < 0)) if n else np.nan,
                                 "stat": stat, "p_value": p})

    lags_anom = reliable_lag.loc[reliable_lag["group"] == "anomaly", "lag_diff_minus_diff2"].values
    lags_norm = reliable_lag.loc[reliable_lag["group"] == "normal", "lag_diff_minus_diff2"].values
    if len(lags_anom) >= 5 and len(lags_norm) >= 5:
        u_stat, u_p = stats.mannwhitneyu(lags_anom, lags_norm, alternative="two-sided")
    else:
        u_stat, u_p = np.nan, np.nan
    group_test_rows.append({"test": "anomaly vs normal (Mann-Whitney U)",
                             "n": f"{len(lags_anom)} vs {len(lags_norm)}", "median_lag": np.nan,
                             "frac_diff_leads(lag>0)": np.nan, "frac_diff2_leads(lag<0)": np.nan,
                             "stat": u_stat, "p_value": u_p})
    group_test_df = pd.DataFrame(group_test_rows)
    n_group_tests = int(group_test_df["p_value"].notna().sum())
    group_test_df["p_value_bonferroni"] = (group_test_df["p_value"] * max(n_group_tests, 1)).clip(upper=1.0)
    group_test_df.to_csv(OUT_DIR / "stage2_diagD_lag_group_comparison.csv", index=False)
    print(group_test_df.to_string(index=False))
    return group_test_df


def main(kf_by_channel: dict[str, LocalLinearTrendKF] | None = None) -> None:
    if kf_by_channel:
        _kf_cache.update(kf_by_channel)

    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

    draws = pd.read_csv(DRAWS_PATH)
    seg_corrected = pd.read_csv(SEG_CORRECTED_PATH)
    reliable_ids = seg_corrected.loc[seg_corrected["reliability_tier"].isin(["high", "medium"]), ["channel", "segment"]]
    canonical_onset = (
        draws.merge(reliable_ids, on=["channel", "segment"], how="inner")
        .groupby(["channel", "segment"])["onset_idx_sampled"].median().round().astype(int)
        .rename("canonical_onset_idx").reset_index()
    )

    run_diag_A(seg, canonical_onset)
    run_diag_B(seg, canonical_onset)

    precedence_df = pd.read_csv(ANOMALY_CROSSING_PATH)
    precedence_df["skip_reason"] = precedence_df["skip_reason"].fillna("")
    diagC_df = run_diag_C(seg, canonical_onset, precedence_df)

    anomaly_df = pd.read_csv(ANOMALY_CROSSING_PATH)
    normal_df = diagC_df
    anomaly_df["skip_reason"] = anomaly_df["skip_reason"].fillna("")
    normal_df["skip_reason"] = normal_df["skip_reason"].fillna("")
    anomaly_valid = anomaly_df[anomaly_df["skip_reason"] == ""].copy()
    normal_valid = normal_df[normal_df["skip_reason"] == ""].copy()

    run_diag_D_proportions(anomaly_valid, normal_valid)
    run_diag_D_lag(seg, anomaly_valid, normal_valid)


if __name__ == "__main__":
    main()
