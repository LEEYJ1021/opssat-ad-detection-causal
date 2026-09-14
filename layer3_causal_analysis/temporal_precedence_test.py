"""
layer3_causal_analysis/temporal_precedence_test.py
=============================================================================
Pooled Wilcoxon signed-rank temporal-precedence test (README §Layer 3 (b), Fig. 5b).

If level shift were the true driver of the anomaly and diff/diff2 were merely
downstream consequences, level should cross its anomaly threshold *first*.
This script tests that directly.

방법:
  1) onset_mc_propagation.py가 만든 200-draw onset posterior의 '중앙값'을
     각 세그먼트의 대표(canonical) onset으로 삼는다 (reliability_tier가
     high/medium인 세그먼트만).
  2) level은 원신호 값, diff/diff2는 1차/2차 차분 값에 대해 '베이스라인
     (pre-onset 구간) 대비 |z| > Z_THRESHOLD(기본 3.0)를 처음 넘는 시점'을
     onset 대비 경과초로 계산한다 — level은 부호가 세그먼트마다 뒤섞여 있어
     (onset_mc_propagation.py의 ①보정에서 확인됨) 절댓값 기준 최초 이탈
     시점으로 세 변수를 동일한 축 위에 놓는다.
  3) level/diff/diff2 세 변수의 첫 이탈시점을 세그먼트 단위로 짝지어
     Wilcoxon 부호순위검정으로 비교한다 (풀링 + 채널별).

출력:
  - stage2_precedence_crossing_times.csv   (세그먼트별 첫 이탈시점, 초)
  - stage2_precedence_test_pooled.csv      (5채널 풀링 Wilcoxon 결과)
  - stage2_precedence_test_by_channel.csv  (채널별 Wilcoxon 결과)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _shared import (
    BASE_DIR, SEGMENTS_PATH, OUT_DIR, FINAL_CHANNELS, BURN_IN,
    MIN_WINDOW_POINTS, section,
)

DRAWS_PATH = OUT_DIR / "stage2_mc_draws_raw.csv"
SEG_CORRECTED_PATH = OUT_DIR / "stage2_mc_segment_summary_corrected.csv"

Z_THRESHOLD = 3.0          # 베이스라인 대비 |z|가 이 값을 넘으면 '이탈'로 판정
PAIRS = [("level", "diff"), ("level", "diff2"), ("diff", "diff2")]

def analyze_segment_precedence(values: np.ndarray, sampling: float, onset_idx: int) -> dict:
    """onset_idx를 기준으로 level/diff/diff2 각각 '베이스라인 대비 |z|>Z_THRESHOLD를
    처음 넘는 시점'을 onset 대비 경과초(elapsed seconds)로 계산한다.
    아직 넘지 않았거나(=세그먼트 끝까지 이탈 없음) 베이스라인 추정이 불가능하면
    NaN을 반환한다 (NaN은 '측정 불가'가 아니라 '이 변수는 이탈을 보이지 않았다'는
    뜻일 수도 있다는 점에 유의)."""
    n = len(values)
    result = {"level_cross_sec": np.nan, "diff_cross_sec": np.nan, "diff2_cross_sec": np.nan}

    # --- level: raw index t 그대로 사용 ---
    pre_level = values[BURN_IN:onset_idx]
    post_level = values[onset_idx:]
    if len(pre_level) >= MIN_WINDOW_POINTS and len(post_level) >= MIN_WINDOW_POINTS:
        mean_pre, std_pre = np.mean(pre_level), np.std(pre_level, ddof=1)
        if std_pre > 0 and np.isfinite(std_pre):
            z = np.abs(post_level - mean_pre) / std_pre
            hit = np.where(z > Z_THRESHOLD)[0]
            if len(hit) > 0:
                raw_idx = onset_idx + hit[0]
                result["level_cross_sec"] = float((raw_idx - onset_idx) * sampling)

    # --- diff: diff_all의 인덱스 k는 raw 인덱스 k+1에 대응 (1칸 밀림 보정) ---
    diff_all = np.diff(values)
    diff_pre_end = max(onset_idx - 1, 0)
    diff_pre = diff_all[max(BURN_IN - 1, 0):diff_pre_end]
    diff_post = diff_all[diff_pre_end:]
    if len(diff_pre) >= MIN_WINDOW_POINTS and len(diff_post) >= MIN_WINDOW_POINTS:
        mean_pre, std_pre = np.mean(diff_pre), np.std(diff_pre, ddof=1)
        if std_pre > 0 and np.isfinite(std_pre):
            z = np.abs(diff_post - mean_pre) / std_pre
            hit = np.where(z > Z_THRESHOLD)[0]
            if len(hit) > 0:
                k = diff_pre_end + hit[0]
                raw_idx = k + 1
                result["diff_cross_sec"] = float((raw_idx - onset_idx) * sampling)

    # --- diff2: diff2_all의 인덱스 m은 raw 인덱스 m+2에 대응 (2칸 밀림 보정) ---
    diff2_all = np.diff(diff_all)
    diff2_pre_end = max(onset_idx - 2, 0)
    diff2_pre = diff2_all[max(BURN_IN - 2, 0):diff2_pre_end]
    diff2_post = diff2_all[diff2_pre_end:]
    if len(diff2_pre) >= MIN_WINDOW_POINTS and len(diff2_post) >= MIN_WINDOW_POINTS:
        mean_pre, std_pre = np.mean(diff2_pre), np.std(diff2_pre, ddof=1)
        if std_pre > 0 and np.isfinite(std_pre):
            z = np.abs(diff2_post - mean_pre) / std_pre
            hit = np.where(z > Z_THRESHOLD)[0]
            if len(hit) > 0:
                m = diff2_pre_end + hit[0]
                raw_idx = m + 2
                result["diff2_cross_sec"] = float((raw_idx - onset_idx) * sampling)

    return result


def main() -> None:
    # =============================================================================
    # 2. 데이터 로드 + 대표 onset(canonical onset) 산출
    # =============================================================================
    section("0. 데이터 로드 및 대표 onset 산출 (§2-2 몬테카를로 draw 중앙값)")

    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

    draws = pd.read_csv(DRAWS_PATH)
    seg_corrected = pd.read_csv(SEG_CORRECTED_PATH)

    reliable_ids = seg_corrected.loc[
        seg_corrected["reliability_tier"].isin(["high", "medium"]), ["channel", "segment"]
    ]
    print(f"신뢰도 high+medium 세그먼트: {len(reliable_ids)}개 (§2-2 보정 결과 그대로 재사용)")

    canonical_onset = (
        draws.merge(reliable_ids, on=["channel", "segment"], how="inner")
        .groupby(["channel", "segment"])["onset_idx_sampled"]
        .median()
        .round()
        .astype(int)
        .rename("canonical_onset_idx")
        .reset_index()
    )
    print(f"대표 onset 계산 완료: {len(canonical_onset)}개 세그먼트")


    # =============================================================================
    # 3. 세그먼트별 첫 이탈 시점 계산
    # =============================================================================
    section("1. 세그먼트별 level/diff/diff² 첫 임계값 교차 시점(초) 계산")

    precedence_rows = []
    for _, row in canonical_onset.iterrows():
        ch, sid, onset_idx = row["channel"], int(row["segment"]), int(row["canonical_onset_idx"])
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid)].sort_values("timestamp")
        values = s["value"].values
        sampling = float(s["sampling"].iloc[0])
        n = len(values)

        if onset_idx < BURN_IN + MIN_WINDOW_POINTS or onset_idx > n - MIN_WINDOW_POINTS:
            precedence_rows.append({
                "channel": ch, "segment": sid, "n_points": n, "onset_idx": onset_idx,
                "skip_reason": "onset이 세그먼트 경계에 가까워 pre/post 확보 불가",
                "level_cross_sec": np.nan, "diff_cross_sec": np.nan, "diff2_cross_sec": np.nan,
            })
            continue

        cross = analyze_segment_precedence(values, sampling, onset_idx)
        precedence_rows.append({
            "channel": ch, "segment": sid, "n_points": n, "onset_idx": onset_idx,
            "skip_reason": "", **cross,
        })

    precedence_df = pd.DataFrame(precedence_rows)
    precedence_out = OUT_DIR / "stage2_precedence_crossing_times.csv"
    precedence_df.to_csv(precedence_out, index=False)
    print(f"[저장] {precedence_out}  ({len(precedence_df)} rows)")

    n_no_skip = int((precedence_df["skip_reason"] == "").sum())
    print(f"경계 문제로 스킵된 세그먼트: {len(precedence_df) - n_no_skip}개, 분석 대상: {n_no_skip}개")
    for col in ["level_cross_sec", "diff_cross_sec", "diff2_cross_sec"]:
        n_found = int(precedence_df[col].notna().sum())
        print(f"  {col}: 이탈시점 발견된 세그먼트 {n_found}/{n_no_skip}개")


    # =============================================================================
    # 4. Wilcoxon 부호순위검정 — level vs diff vs diff² 선행성 비교
    # =============================================================================
    section("2. Wilcoxon 부호순위검정: level / diff / diff² 짝비교")

    PAIRS = [("level", "diff"), ("level", "diff2"), ("diff", "diff2")]


    def run_pairwise_test(df: pd.DataFrame, a: str, b: str) -> dict:
        """짝지은(paired) 세그먼트 단위로 a의 첫 이탈시점과 b의 첫 이탈시점을 비교.
        diff_time = t_a - t_b. 음수면 a가 더 빨리(먼저) 이탈했다는 뜻."""
        ta = df[f"{a}_cross_sec"].values
        tb = df[f"{b}_cross_sec"].values
        mask = ~np.isnan(ta) & ~np.isnan(tb)
        n_pairs = int(mask.sum())

        base = {
            "pair": f"{a} vs {b}", "n_pairs": n_pairs, "median_diff_sec": np.nan,
            "wilcoxon_stat": np.nan, "p_value": np.nan,
            f"frac_{a}_precedes_{b}": np.nan, "note": "",
        }
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
        base["median_diff_sec"] = float(np.median(diff_time))
        base["wilcoxon_stat"] = float(stat)
        base["p_value"] = float(p)
        base[f"frac_{a}_precedes_{b}"] = float(np.mean(diff_time < 0))
        return base


    valid_df = precedence_df[precedence_df["skip_reason"] == ""]

    print("--- 전체(5채널 풀링) ---")
    pooled_results = [run_pairwise_test(valid_df, a, b) for a, b in PAIRS]
    pooled_df = pd.DataFrame(pooled_results)
    print(pooled_df.to_string(index=False))

    print("\n--- 채널별 ---")
    channel_results = []
    for ch in FINAL_CHANNELS:
        sub = valid_df[valid_df["channel"] == ch]
        for a, b in PAIRS:
            r = run_pairwise_test(sub, a, b)
            r["channel"] = ch
            channel_results.append(r)
    channel_pair_df = pd.DataFrame(channel_results)
    print(channel_pair_df.to_string(index=False))

    # 다중비교 보정 (Bonferroni) — 풀링 3개 + 채널별 3개x5채널 = 18개 검정
    n_tests_total = int(pooled_df["p_value"].notna().sum()) + int(channel_pair_df["p_value"].notna().sum())
    if n_tests_total > 0:
        pooled_df["p_bonferroni"] = (pooled_df["p_value"] * n_tests_total).clip(upper=1.0)
        channel_pair_df["p_bonferroni"] = (channel_pair_df["p_value"] * n_tests_total).clip(upper=1.0)
        print(f"\n[다중비교 보정] 유효 검정 총 {n_tests_total}개 기준 Bonferroni 보정 p값을 "
              f"p_bonferroni 컬럼에 추가했습니다.")

    pooled_out = OUT_DIR / "stage2_precedence_test_pooled.csv"
    channel_out = OUT_DIR / "stage2_precedence_test_by_channel.csv"
    pooled_df.to_csv(pooled_out, index=False)
    channel_pair_df.to_csv(channel_out, index=False)
    print(f"\n[저장] {pooled_out}")
    print(f"[저장] {channel_out}")


    # =============================================================================


if __name__ == "__main__":
    main()
