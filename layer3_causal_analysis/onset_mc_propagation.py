"""
layer3_causal_analysis/onset_mc_propagation.py
=============================================================================
Monte Carlo propagation of BOCPD onset-time uncertainty (README §Layer 3 (a)).

Layer 1에서 3중 검증까지 마치고 확정한 5개 채널(CADC0872/0873/0874/0888/0894),
잠금 파라미터(mixture=False, forgetting=True)를 그대로 재사용합니다.

이 스크립트가 하는 일:
  1) 채널별로 onset이 검출된 각 이상(anomaly) 세그먼트에 대해 Kalman filter ->
     정규화 혁신값(z) -> BOCPD를 실행해 onset run-length 사후분포를 얻는다.
  2) 그 사후분포에서 확률에 비례해 onset 후보를 N_MC_DRAWS(기본 200)회
     가중샘플링한다 — "onset이 정확히 어디인지 100% 확신할 수 없다"는 사실을
     결과에 반영하기 위함이다.
  3) 매 draw마다 pre-onset vs post-onset 구간에서 level(원신호) / diff(1차
     변화) / diff2(2차 변화)의 '제곱값'(분산 대리 지표, level은 원값)을
     Welch t-test + Cohen's d로 비교한다.
  4) 세그먼트 단위로 p-value/d의 중앙값 · 95% CI · 유의미한 draw 비율을 집계한다.
  5) ①(세그먼트 간 부호 상쇄) / ②(draw 수가 적어 신뢰도가 낮은 세그먼트)를
     분리해서 보정한 채널 단위 요약을 한 번 더 만든다.

주의 — 이 스크립트가 "하지 않는" 것:
  - level→diff→diff² 선행성 검정(Wilcoxon)은 temporal_precedence_test.py
  - 채널 간 랜덤효과 메타분석(DerSimonian-Laird)은 meta_analysis_random_effects.py
  - 이 스크립트의 출력은 "이상 전후 통계적으로 유의미한 차이가 있는가"까지만
    답하며, "무엇이 원인이다"라는 인과 주장을 하지 않는다.

출력:
  - stage2_mc_draws_raw.csv                  (draw 단위 원자료)
  - stage2_mc_segment_summary.csv            (세그먼트 단위 1차 집계)
  - stage2_mc_channel_summary.csv            (채널 단위 단순평균, 참고용)
  - stage2_mc_segment_summary_corrected.csv  (①②보정 + reliability_tier)
  - stage2_mc_channel_summary_corrected.csv  (①②보정 채널 요약, |d| 가중평균)
  - stage2_mc_reliability_report.csv         (채널 x tier 교차표)
  - onset_posteriors.parquet                 (onset posterior 원자료, 파생 산출물)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from _shared import (
    BASE_DIR, SEGMENTS_PATH, OUT_DIR, FINAL_CHANNELS, FEATURES,
    USE_BOCPD_FORGETTING, BURN_IN, MIN_WINDOW_POINTS, RANDOM_STATE,
    section, profile_channel, hierarchical_shrink_variance,
    LocalLinearTrendKF, BOCPD,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 몬테카를로 설정 (설계서 §2-2 기준값) ---
N_MC_DRAWS = 200          # onset posterior에서 뽑을 표본 수
MIN_DRAWS_HIGH = 100      # reliability tier 경계값
MIN_DRAWS_MEDIUM = 30

rng = np.random.default_rng(RANDOM_STATE)


# =============================================================================
# 1. 채널 단위 Kalman filter -> BOCPD 실행, 세그먼트별 onset run-length 사후분포
# =============================================================================

def get_channel_kf(profile, r_shrunk, q_shrunk, quant_floor, ch) -> LocalLinearTrendKF:
    return LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch],
                               quantization_floor=quant_floor[ch])


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt((va + vb) / 2.0)
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def welch_test_features(values: np.ndarray, onset_idx: int) -> dict:
    """onset_idx 기준 pre/post 구간에서 level / diff^2 / diff2^2 의 Welch
    t-test p-value와 Cohen's d를 계산한다. diff/diff2는 제곱값으로 분산 대리
    지표를 사용(§2-2 원 설계)."""
    out = {"level_p": np.nan, "level_d": np.nan, "diff_p": np.nan, "diff_d": np.nan,
           "diff2_p": np.nan, "diff2_d": np.nan}

    pre_level = values[BURN_IN:onset_idx]
    post_level = values[onset_idx:]
    if len(pre_level) >= MIN_WINDOW_POINTS and len(post_level) >= MIN_WINDOW_POINTS:
        _, p = stats.ttest_ind(post_level, pre_level, equal_var=False)
        out["level_p"], out["level_d"] = float(p), cohens_d(post_level, pre_level)

    diff_all = np.diff(values)
    split1 = max(onset_idx - 1, 0)
    pre_diff_sq = diff_all[max(BURN_IN - 1, 0):split1] ** 2
    post_diff_sq = diff_all[split1:] ** 2
    if len(pre_diff_sq) >= MIN_WINDOW_POINTS and len(post_diff_sq) >= MIN_WINDOW_POINTS:
        _, p = stats.ttest_ind(post_diff_sq, pre_diff_sq, equal_var=False)
        out["diff_p"], out["diff_d"] = float(p), cohens_d(post_diff_sq, pre_diff_sq)

    diff2_all = np.diff(diff_all)
    split2 = max(onset_idx - 2, 0)
    pre_diff2_sq = diff2_all[max(BURN_IN - 2, 0):split2] ** 2
    post_diff2_sq = diff2_all[split2:] ** 2
    if len(pre_diff2_sq) >= MIN_WINDOW_POINTS and len(post_diff2_sq) >= MIN_WINDOW_POINTS:
        _, p = stats.ttest_ind(post_diff2_sq, pre_diff2_sq, equal_var=False)
        out["diff2_p"], out["diff2_d"] = float(p), cohens_d(post_diff2_sq, pre_diff2_sq)

    return out


def run_monte_carlo_event_study(seg: pd.DataFrame, r_shrunk: dict, q_shrunk: dict,
                                 quant_floor: dict, channels: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """채널별로 이상 세그먼트마다 KF->BOCPD 실행, run-length 사후분포에서
    N_MC_DRAWS회 가중샘플링, 매 draw마다 Welch t-test/Cohen's d 계산."""
    draw_rows, segment_rows = [], []

    for ch in channels:
        kf = LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch],
                                 quantization_floor=quant_floor[ch])
        bocpd = BOCPD(forgetting=USE_BOCPD_FORGETTING)

        anomalous = seg[(seg["channel"] == ch) & (seg["anomaly"] == 1)]
        for sid, s in anomalous.groupby("segment"):
            s = s.sort_values("timestamp")
            values = s["value"].values
            n = len(values)

            if n < BURN_IN + 2 * MIN_WINDOW_POINTS:
                segment_rows.append({"channel": ch, "segment": int(sid), "n_points": n,
                                      "n_valid_draws": 0, "skip_reason": "세그먼트 길이 부족 (pre/post 윈도우 확보 불가)"})
                continue

            innovations, innovation_vars, _ = kf.run(values)
            z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
            rl_posteriors = bocpd.run(z)
            final_rl = rl_posteriors[-1]

            # run-length r at time n-1 -> onset_idx = (n-1) - r
            candidate_onsets = (n - 1) - np.arange(len(final_rl))
            valid_mask = (candidate_onsets >= BURN_IN + MIN_WINDOW_POINTS) & (candidate_onsets <= n - MIN_WINDOW_POINTS)

            if not valid_mask.any() or final_rl[valid_mask].sum() <= 0:
                segment_rows.append({"channel": ch, "segment": int(sid), "n_points": n,
                                      "n_valid_draws": 0, "skip_reason": "onset 미검출 (BOCPD 리셋 패턴 없음)"})
                continue

            probs = final_rl[valid_mask]
            probs = probs / probs.sum()
            onsets_pool = candidate_onsets[valid_mask]

            sampled_onsets = rng.choice(onsets_pool, size=N_MC_DRAWS, p=probs, replace=True)

            seg_draw_rows = []
            for draw_i, onset_idx in enumerate(sampled_onsets):
                feats = welch_test_features(values, int(onset_idx))
                seg_draw_rows.append({"channel": ch, "segment": int(sid), "draw": draw_i,
                                       "onset_idx_sampled": int(onset_idx), **feats})
            draw_rows.extend(seg_draw_rows)

            ddf = pd.DataFrame(seg_draw_rows)
            n_valid = len(ddf)
            if n_valid == 0:
                segment_rows.append({"channel": ch, "segment": int(sid), "n_points": n,
                                      "n_valid_draws": 0, "skip_reason": "모든 draw에서 pre/post 윈도우 부족"})
                continue

            row = {"channel": ch, "segment": int(sid), "n_points": n, "n_valid_draws": n_valid,
                   "skip_reason": "", "onset_sample_std": float(np.std(sampled_onsets))}
            for feat in FEATURES:
                p_col, d_col = f"{feat}_p", f"{feat}_d"
                p_vals = ddf[p_col].dropna().values
                d_vals = ddf[d_col].dropna().values
                row[f"{feat}_frac_significant"] = float(np.mean(p_vals < 0.05)) if len(p_vals) else np.nan
                row[f"{feat}_d_median"] = float(np.median(d_vals)) if len(d_vals) else np.nan
            segment_rows.append(row)

    draw_df = pd.DataFrame(draw_rows)
    segment_summary_df = pd.DataFrame(segment_rows)
    return draw_df, segment_summary_df


def summarize_by_channel(segment_summary_df: pd.DataFrame) -> pd.DataFrame:
    valid = segment_summary_df[segment_summary_df["n_valid_draws"] > 0]
    rows = []
    for ch in FINAL_CHANNELS:
        sub = valid[valid["channel"] == ch]
        n_total = int((segment_summary_df["channel"] == ch).sum())
        row = {"channel": ch, "n_segments_evaluated": len(sub), "n_segments_total": n_total}
        for feat in FEATURES:
            row[f"{feat}_frac_significant_mean"] = float(sub[f"{feat}_frac_significant"].mean()) if len(sub) else np.nan
            row[f"{feat}_d_median_mean"] = float(sub[f"{feat}_d_median"].mean()) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 4. ①②보정 — 부호 상쇄 분리 + 신뢰도 등급(reliability tier)
# =============================================================================

def correct_segment(sub: pd.DataFrame, feat: str) -> dict:
    """한 세그먼트의 draw들(sub)에서 feat(level/diff/diff2)의
    부호-무관 크기, 부호 있는 값, 세그먼트 내부 방향 일관성을 계산한다."""
    d_col, p_col = f"{feat}_d", f"{feat}_p"
    d_vals = sub[d_col].dropna().values
    p_vals = sub[p_col].dropna().values

    out: dict = {}
    if len(d_vals) > 0:
        abs_d = np.abs(d_vals)
        out[f"{feat}_d_signed_median"] = float(np.median(d_vals))
        out[f"{feat}_d_abs_median"] = float(np.median(abs_d))
        out[f"{feat}_d_abs_ci_low"] = float(np.percentile(abs_d, 2.5))
        out[f"{feat}_d_abs_ci_high"] = float(np.percentile(abs_d, 97.5))

        n_pos = int(np.sum(d_vals > 0))
        n_neg = int(np.sum(d_vals < 0))
        n_nonzero = n_pos + n_neg
        out[f"{feat}_direction_consistency"] = (
            max(n_pos, n_neg) / n_nonzero if n_nonzero > 0 else np.nan
        )
    else:
        out[f"{feat}_d_signed_median"] = np.nan
        out[f"{feat}_d_abs_median"] = np.nan
        out[f"{feat}_d_abs_ci_low"] = np.nan
        out[f"{feat}_d_abs_ci_high"] = np.nan
        out[f"{feat}_direction_consistency"] = np.nan

    out[f"{feat}_frac_significant"] = float(np.mean(p_vals < 0.05)) if len(p_vals) > 0 else np.nan
    return out


def tier(n: int) -> str:
    if n == 0:
        return "excluded"
    if n >= MIN_DRAWS_HIGH:
        return "high"
    if n >= MIN_DRAWS_MEDIUM:
        return "medium"
    return "low"


def correct_mc_results(draws: pd.DataFrame, seg_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """① 부호 상쇄와 ② 신뢰도(draw 수)를 분리해 채널 요약을 다시 만든다."""
    corrected_rows = []
    for (ch, sid), sub in draws.groupby(["channel", "segment"]):
        row = {"channel": ch, "segment": sid, "n_valid_draws": len(sub)}
        for feat in FEATURES:
            row.update(correct_segment(sub, feat))
        corrected_rows.append(row)
    corrected_df = pd.DataFrame(corrected_rows)

    skipped = seg_summary.loc[
        seg_summary["n_valid_draws"] == 0, ["channel", "segment", "n_points", "skip_reason"]
    ].copy()
    skipped["n_valid_draws"] = 0

    corrected_df["reliability_tier"] = corrected_df["n_valid_draws"].apply(tier)
    skipped["reliability_tier"] = "excluded"

    full_df = pd.concat([corrected_df, skipped], ignore_index=True, sort=False)

    reliable = full_df[full_df["reliability_tier"].isin(["high", "medium"])].copy()
    channel_rows = []
    for ch in sorted(full_df["channel"].dropna().unique()):
        sub = reliable[reliable["channel"] == ch]
        n_total = int((full_df["channel"] == ch).sum())
        weights = sub["n_valid_draws"].values.astype(float)

        row = {
            "channel": ch,
            "n_segments_total": n_total,
            "n_segments_reliable(high+medium)": len(sub),
            "n_segments_excluded(low+none)": n_total - len(sub),
        }

        for feat in FEATURES:
            fs = sub[f"{feat}_frac_significant"].values
            dabs = sub[f"{feat}_d_abs_median"].values
            dsigned = sub[f"{feat}_d_signed_median"].values
            dirc = sub[f"{feat}_direction_consistency"].values

            def wmean(vals: np.ndarray) -> float:
                mask = ~np.isnan(vals)
                if not mask.any():
                    return np.nan
                return float(np.average(vals[mask], weights=weights[mask]))

            row[f"{feat}_frac_significant_weighted"] = wmean(fs)
            row[f"{feat}_d_abs_median_weighted"] = wmean(dabs)
            row[f"{feat}_d_signed_median_weighted"] = wmean(dsigned)
            row[f"{feat}_direction_consistency_weighted"] = wmean(dirc)

            valid_sign = dsigned[~np.isnan(dsigned)]
            n_pos_seg = int(np.sum(valid_sign > 0))
            n_neg_seg = int(np.sum(valid_sign < 0))
            n_seg_valid = n_pos_seg + n_neg_seg
            row[f"{feat}_frac_segments_positive"] = n_pos_seg / n_seg_valid if n_seg_valid else np.nan
            row[f"{feat}_frac_segments_negative"] = n_neg_seg / n_seg_valid if n_seg_valid else np.nan

        channel_rows.append(row)

    channel_corrected_df = pd.DataFrame(channel_rows)

    tier_report = (
        full_df.groupby(["channel", "reliability_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["high", "medium", "low", "excluded"], fill_value=0)
    )

    return full_df, channel_corrected_df, tier_report


# =============================================================================
# 5. 실행 (run.py에서 호출하는 진입점)
# =============================================================================

def main() -> None:
    section("0. 데이터 로드 및 채널 파라미터 재적합 (1단계 잠금 설정 그대로)")

    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)
    print(f"segments(5개 확정채널만): {seg.shape}")
    print(f"N_MC_DRAWS = {N_MC_DRAWS}, MIN_WINDOW_POINTS = {MIN_WINDOW_POINTS}, BURN_IN = {BURN_IN}")

    profiles = {}
    for ch in FINAL_CHANNELS:
        nominal_vals = [
            s.sort_values("timestamp")["value"].values
            for _, s in seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment")
        ]
        profiles[ch] = profile_channel(ch, nominal_vals)

    r_estimates = {ch: (profiles[ch].r_robust, max(profiles[ch].n_nominal_points, 2)) for ch in FINAL_CHANNELS}
    q_estimates = {ch: (profiles[ch].q_robust, max(profiles[ch].n_nominal_points, 2)) for ch in FINAL_CHANNELS}
    r_shrunk = hierarchical_shrink_variance(r_estimates)
    q_shrunk = hierarchical_shrink_variance(q_estimates)
    quant_floor = {
        ch: (profiles[ch].quantization_step ** 2 / 12.0) if np.isfinite(profiles[ch].quantization_step) else 0.0
        for ch in FINAL_CHANNELS
    }

    section("1. onset 불확실성 몬테카를로 전파 실행")
    draw_df, segment_summary_df = run_monte_carlo_event_study(
        seg, r_shrunk, q_shrunk, quant_floor, FINAL_CHANNELS
    )
    draw_df.to_csv(OUT_DIR / "stage2_mc_draws_raw.csv", index=False)
    segment_summary_df.to_csv(OUT_DIR / "stage2_mc_segment_summary.csv", index=False)

    section("2. 채널 단위 요약 (단순평균, 참고용)")
    channel_summary_df = summarize_by_channel(segment_summary_df)
    channel_summary_df.to_csv(OUT_DIR / "stage2_mc_channel_summary.csv", index=False)
    print(channel_summary_df.to_string(index=False))

    section("3. ①(부호 상쇄) / ②(신뢰도) 보정")
    full_df, channel_corrected_df, tier_report = correct_mc_results(draw_df, segment_summary_df)
    full_df.to_csv(OUT_DIR / "stage2_mc_segment_summary_corrected.csv", index=False)
    channel_corrected_df.to_csv(OUT_DIR / "stage2_mc_channel_summary_corrected.csv", index=False)
    tier_report.to_csv(OUT_DIR / "stage2_mc_reliability_report.csv")
    print(tier_report.to_string())

    onset_cols = ["channel", "segment", "draw", "onset_idx_sampled"]
    draw_df[onset_cols].to_parquet(OUT_DIR / "onset_posteriors.parquet", index=False)

    n_total = len(segment_summary_df)
    n_skipped = int((segment_summary_df["n_valid_draws"] == 0).sum())
    section("완료")
    print(f"평가 대상 이상 세그먼트: {n_total}개, onset 미검출로 스킵: {n_skipped}개 "
          f"({n_skipped / max(n_total, 1):.1%})")
    print(
        "해석 시 주의사항:\n"
        "1) frac_significant는 'N_MC_DRAWS개 onset 후보 중 p<0.05로 나온 비율'이며,\n"
        "   onset 위치 불확실성 하에서도 결론이 얼마나 안정적인지를 나타낸다.\n"
        "2) 이 결과는 아직 인과관계가 아니라 pre/post 차이의 통계적 유의성이다 —\n"
        "   temporal_precedence_test.py와 quasi_experimental_placebo.py가 이어져야\n"
        "   causal-signature 주장의 근거가 완성된다.\n"
        "3) diff/diff2 검정은 '제곱값'으로 분산 증가를 근사한 것이며, 공식 분산검정\n"
        "   (Levene/Brown-Forsythe) 교차검증은 향후 리비전 권장 사항이다."
    )


if __name__ == "__main__":
    main()
