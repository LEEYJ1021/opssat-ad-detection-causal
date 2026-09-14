"""
layer3_causal_analysis/labeling_protocol_audit.py
=============================================================================
External audit of the OPS-SAT-AD labeling protocol (B-5) + train/test/full
triple re-verification of every §3 causal-signature result (B-6, B-7)
(README §Layer 3 (e), Fig. 3b; "Threats to validity" table).

이 스크립트는 두 개의 독립적인 하위작업(B-5, B-6/B-7)을 하나로 묶는다.

B-5. 라벨링 프로토콜 재확인 (외부 문서 조사, 코드 아님)
-----------------------------------------------------------------------------
문제: temporal_precedence_test.py(§2-3)에서 이상 세그먼트들의 onset 위치가
세그먼트 길이 대비 평균 56.9% 지점(뒷부분 쪽)에 몰려 있다는 걸 발견했다.
이게 (1) 실제 물리적 현상인지 (2) ESA가 세그먼트를 자를 때 "이상 발생 전
여유분을 넉넉히 포함"하는 관행 때문에 생긴 인공물인지 구분이 안 된 상태였다.

조사 결과(Ruszczak et al., Scientific Data 2025; OXI 툴 SoftwareX 논문;
arXiv/PMC 전문 확인): 세그먼트는 완전히 수동(manual)으로 잘렸다. OPS-SAT
운영 엔지니어들이 이상탐지에 "흥미롭다"고 주관적으로 판단한 텔레메트리
구간을 추천했고, OXI라는 웹 기반 시각화·주석 도구를 통해 도메인 전문가들이
협업하여 세그먼트를 수동으로 추출·주석했다. 세그먼트 경계 설정 기준(예:
"onset 전 몇 초를 포함한다" 같은 정량적 규칙)은 어느 공식 문서에도 명시되어
있지 않다.

결론(코드로 확인 불가, 조사로 확인): 세그먼트 경계가 "엔지니어의 주관적
판단"으로 정해졌다는 것 자체가, onset 위치비율 쏠림이 일관된 정량적 규칙의
산물이 아니라 케이스마다 제각각인 사람의 판단이 우연히 평균적으로 뒷부분에
쏠린 것이라는 뜻이다. 절단이 규칙 기반이 아닌 전문가 주관적 판단에 의한
것이라는 점에서, diff/diff² 선행성 결과를 체계적으로 왜곡하는 규칙 기반
인공물일 가능성은 제한적이다 — 다만 완전히 배제할 수는 없다.

B-6/B-7. train-only / test-only §3 준실험 재현 + 3중 비교 (코드 재실행)
-----------------------------------------------------------------------------
§3(quasi_experimental_placebo.py)은 원래 train/test 구분 없이 전체 데이터로
수행되었다. "diff/diff²가 이상 고유의 시그니처다"라는 핵심 결론이 우연히
전체 데이터라서 잘 나온 것이 아님을 확인하기 위해, train 세그먼트만/test
세그먼트만으로 동일한 절차(관측 효과크기 계산 -> 위약 풀 생성 -> Mann-Whitney
U 검정)를 독립적으로 재현하고, 전체데이터/train-only/test-only 3중 비교표를
만든다(Layer 2의 train/test triple verification과 동일한 검증 강도).

출력:
  - labeling_protocol_audit_findings.md      (B-5 조사 결과, 논문 서술 권고안)
  - stage3_qexp_mannwhitney_train_only.csv   (train-only Mann-Whitney U)
  - stage3_qexp_mannwhitney_test_only.csv    (test-only Mann-Whitney U)
  - stage3_train_vs_full_comparison.csv      (train vs full 일치표)
  - stage3_triple_verification_summary.csv   (전체/train/test 3중 비교, Fig. 3b 소스)
  - triple_verification_matrix.csv           (README results/layer3 명명과 맞춘 사본)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from _shared import SEGMENTS_PATH, OUT_DIR, FINAL_CHANNELS, section
from quasi_experimental_placebo import (
    build_canonical_onset, compute_observed_effects, build_placebo_pool,
    run_mannwhitney, FEATURES,
)

# =============================================================================
# B-5. 라벨링 프로토콜 외부 조사 — 결과를 마크다운 파일로 기록 (코드 재실행 아님)
# =============================================================================

B5_FINDINGS_MD = """\
# B-5. OPS-SAT-AD 라벨링 프로토콜 외부 조사 결과

## 문제 제기
temporal_precedence_test.py(§2-3)에서 이상 세그먼트들의 onset 위치가 세그먼트
길이 대비 평균 56.9% 지점(뒷부분 쪽)에 몰려 있다는 것을 발견했다. 이것이

1. 실제 물리적 현상(이상이 원래 그 시점에 발생하는 경향)인지
2. ESA가 세그먼트를 자를 때 "이상 발생 전 여유분을 넉넉히 포함"하는 관행
   때문에 생긴 인공물인지

구분이 되지 않았다. 이는 코드로 답할 수 없는 질문이라, OPS-SAT-AD 공식 문서
(Zenodo, GitHub README, 논문 부록)를 대상으로 외부 자료 조사를 수행했다.

## 조사 대상
- 공식 논문: Ruszczak et al., *Scientific Data* (2025)
- OXI 툴 소프트웨어 논문 (SoftwareX)
- 위 논문의 arXiv 버전, PMC 전문

## 확인된 사실
세그먼트는 완전히 **수동(manual)**으로 잘렸다. OPS-SAT 운영 엔지니어들이
이상탐지에 "흥미롭다"고 주관적으로 판단한 텔레메트리 구간을 추천했고, **OXI**
라는 웹 기반 시각화·주석 도구를 통해 도메인 전문가들이 협업하여 정상/이상
구간을 나타내는 텔레메트리 세그먼트를 수동으로 추출·주석했다. 최초 이상 후보
목록은 ESA 우주선 운영 엔지니어 3명이 제공했고, 이후 추가로 정제되었다.

세그먼트 경계 설정 기준(예: "onset 전 몇 초를 포함한다" 같은 정량적 규칙)은
**어느 문서에도 명시되어 있지 않다**. 논문 본문, OXI 소프트웨어 논문
(SoftwareX), PMC 전문, arXiv 버전 전부 확인했지만 "왜 세그먼트가 이 길이로
잘렸는지", "onset 전 여유분을 얼마나 남기는지"에 대한 정량적 규칙은 나오지
않고, "전문가가 육안으로 보고 수동으로 판단했다"는 서술만 반복된다.

## 결론
이건 사실상 **"확인 불가"**가 최종 답이다. 다만 이것이 의미하는 바는 오히려
명확하다: 세그먼트 경계가 "엔지니어의 주관적 판단"으로 정해졌다는 것 자체가,
onset 위치비율(mean=0.569)의 쏠림이 일관된 정량적 규칙(예: "항상 onset 30초
전부터 자른다")의 산물이 아니라, 케이스마다 제각각인 사람의 판단이 우연히
평균적으로 뒷부분에 쏠린 것이라는 뜻이다. 이는 오히려 "규칙적인 인공물"이라는
우려를 다소 완화시킨다 — 만약 정량적 규칙이 있었다면 그 규칙 자체가
diff/diff² 선행성 결과를 왜곡했을 가능성이 있지만, 순수 주관적 판단이라면
그런 체계적 편향의 가능성은 낮다.

## 논문 서술 권고
> "세그먼트 경계는 ESA 전문가의 수동 주석(OXI 툴)으로 결정되었으며, 공개
> 문서에 정량적 절단 규칙은 명시되어 있지 않다. 따라서 onset 위치의 쏠림
> 경향이 라벨링 관행의 체계적 인공물인지 완전히 배제할 수는 없으나, 절단이
> 규칙 기반이 아닌 전문가 주관적 판단에 의한 것이라는 점에서 체계적 편향의
> 가능성은 제한적이다."

이 결론은 코드로 검증할 수 없으므로, B-6/B-7(train-only / test-only 재현)이
"결과가 우연이 아님"을 뒷받침하는 정량적 보강 증거 역할을 한다.
"""


def write_b5_findings() -> None:
    section("B-5. 라벨링 프로토콜 재확인 — 외부 조사 결과 기록")
    out_path = OUT_DIR / "labeling_protocol_audit_findings.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(B5_FINDINGS_MD)
    print(f"[저장] {out_path}")
    print("결론(요약): 세그먼트 경계는 전문가 수동 주석(OXI 툴)으로 결정되었으며,")
    print("공개 문서에 정량적 절단 규칙은 없음 -> 체계적 라벨링 편향 가능성은 제한적.")


# =============================================================================
# B-6/B-7. train-only / test-only §3 재현 + 3중 비교
# =============================================================================

def run_split_qexp(seg_full: pd.DataFrame, split_value: int, split_name: str) -> pd.DataFrame:
    """seg_full(5채널로 필터된 전체 세그먼트)에서 train==split_value인 세그먼트만
    남기고 quasi_experimental_placebo.py와 동일한 절차(canonical onset ->
    관측 효과크기 -> 위약 풀 -> Mann-Whitney U)를 독립적으로 재현한다."""
    if "train" not in seg_full.columns:
        raise ValueError("segments.csv에 'train' 컬럼이 없습니다.")

    seg = seg_full[seg_full["train"] == split_value].reset_index(drop=True)
    print(f"train=={split_value}({split_name}) 세그먼트만 필터링: "
          f"전체 {len(seg_full)}행 -> {split_name} {len(seg)}행 ({len(seg)/len(seg_full):.1%})")

    split_seg_ids = seg.loc[seg["anomaly"] == 1, ["channel", "segment"]].drop_duplicates()
    canonical_onset = build_canonical_onset(seg, seg_filter=split_seg_ids)
    print(f"{split_name} 기준 신뢰도 high+medium 이상 세그먼트: {len(canonical_onset)}개")

    observed_df, onset_ratio_pool = compute_observed_effects(seg, canonical_onset)
    print(observed_df.groupby("channel").size().reindex(FINAL_CHANNELS)
          .rename(f"n_{split_name}_segments").to_string())

    n_by_ch = observed_df.groupby("channel").size().reindex(FINAL_CHANNELS)
    thin = n_by_ch[n_by_ch < 5]
    if len(thin):
        print(f"[주의] 표본 5개 미만 채널: {thin.to_dict()} — 아래 MW 검정에서 "
              f"'표본부족'으로 자동 표시됨")

    placebo_df = build_placebo_pool(seg, onset_ratio_pool)
    mw_df = run_mannwhitney(observed_df, placebo_df)
    out_path = OUT_DIR / f"stage3_qexp_mannwhitney_{split_name}_only.csv"
    mw_df.to_csv(out_path, index=False)
    print(f"\n[저장] {out_path}")
    return mw_df


def compare_train_vs_full(mw_train_df: pd.DataFrame) -> pd.DataFrame | None:
    full_path = OUT_DIR / "stage3_qexp_mannwhitney.csv"
    if not full_path.exists():
        print(f"\n[안내] {full_path} 없음 — quasi_experimental_placebo.py를 먼저 실행하세요.")
        return None

    full_mw = pd.read_csv(full_path)
    cmp = mw_train_df.merge(
        full_mw[["channel", "feature", "p_value_bonferroni", "median_anomaly", "median_placebo"]],
        on=["channel", "feature"], suffixes=("_train", "_full")
    )
    cmp["same_significance_direction"] = (
        (cmp["p_value_bonferroni_train"] < 0.05) == (cmp["p_value_bonferroni_full"] < 0.05)
    )
    section("train-only vs 전체데이터 §3 결론 일치 여부")
    print(cmp[["channel", "feature", "p_value_bonferroni_train", "p_value_bonferroni_full",
               "same_significance_direction"]].to_string(index=False))
    n_match = int(cmp["same_significance_direction"].sum())
    print(f"\n일치하는 (채널x피처) 조합: {n_match} / {len(cmp)}")
    if n_match == len(cmp):
        print("-> train만으로도 §3의 핵심 결론이 그대로 재현됨. in-sample 튜닝 "
              "우려는 이 재현 절차 기준에서 뒷받침되지 않음.")
    else:
        mismatch = cmp[~cmp["same_significance_direction"]]
        print(f"-> 불일치 조합 발견: {mismatch[['channel','feature']].values.tolist()}")
    cmp.to_csv(OUT_DIR / "stage3_train_vs_full_comparison.csv", index=False)
    return cmp


def triple_verification(mw_test_df: pd.DataFrame) -> pd.DataFrame | None:
    """전체데이터 / train-only / test-only 3중 비교표 — Layer 2 triple
    verification과 동일한 강도를 Layer 3(인과분석)에 적용."""
    full_path = OUT_DIR / "stage3_qexp_mannwhitney.csv"
    train_path = OUT_DIR / "stage3_qexp_mannwhitney_train_only.csv"

    if not (full_path.exists() and train_path.exists()):
        print(f"\n[안내] 비교 대상 파일이 없어 3중 비교를 생략합니다.")
        return None

    full_mw = pd.read_csv(full_path)[["channel", "feature", "p_value_bonferroni", "median_anomaly", "median_placebo"]]
    train_mw = pd.read_csv(train_path)[["channel", "feature", "p_value_bonferroni", "median_anomaly", "median_placebo"]]
    test_mw = mw_test_df[["channel", "feature", "p_value_bonferroni", "median_anomaly", "median_placebo"]]

    tri = full_mw.merge(train_mw, on=["channel", "feature"], suffixes=("_full", "_train"))
    tri = tri.merge(test_mw, on=["channel", "feature"])
    tri = tri.rename(columns={
        "p_value_bonferroni": "p_value_bonferroni_test",
        "median_anomaly": "median_anomaly_test",
        "median_placebo": "median_placebo_test",
    })

    def sig_flag(p):
        if pd.isna(p):
            return np.nan
        return bool(p < 0.05)

    tri["sig_full"] = tri["p_value_bonferroni_full"].apply(sig_flag)
    tri["sig_train"] = tri["p_value_bonferroni_train"].apply(sig_flag)
    tri["sig_test"] = tri["p_value_bonferroni_test"].apply(sig_flag)

    def agree_flag(row):
        vals = [row["sig_full"], row["sig_train"], row["sig_test"]]
        if any(pd.isna(v) for v in vals):
            return np.nan
        return bool(vals[0] == vals[1] == vals[2])

    tri["all_three_agree"] = tri.apply(agree_flag, axis=1)

    display_cols = ["channel", "feature", "sig_full", "sig_train", "sig_test",
                     "p_value_bonferroni_full", "p_value_bonferroni_train", "p_value_bonferroni_test",
                     "all_three_agree"]
    section("전체데이터 vs train-only vs test-only 3중 비교 (최종 종합표)")
    print(tri[display_cols].to_string(index=False))

    n_valid = tri["all_three_agree"].notna().sum()
    n_agree = tri["all_three_agree"].sum()
    n_na = tri["all_three_agree"].isna().sum()
    print(f"\n3중 일치(판정 가능한 조합 중): {int(n_agree)} / {int(n_valid)}  "
          f"(판정불가 {int(n_na)}개는 별도 표시, 집계에서 제외)")

    disagree = tri[tri["all_three_agree"] == False]  # noqa: E712
    if len(disagree):
        print(f"\n[불일치 조합]:")
        print(disagree[display_cols].to_string(index=False))
    else:
        print("\n-> 판정 가능한 모든 조합에서 전체/train/test 세 슬라이스가 동일한 유의성 결론으로 일치.")

    na_rows = tri[tri["all_three_agree"].isna()]
    if len(na_rows):
        print(f"\n[판정불가 조합] (표본부족으로 세 슬라이스 중 하나 이상 NaN):")
        print(na_rows[display_cols].to_string(index=False))

    tri.to_csv(OUT_DIR / "stage3_triple_verification_summary.csv", index=False)
    tri.to_csv(OUT_DIR / "triple_verification_matrix.csv", index=False)  # README 명명과 맞춘 사본
    print(f"\n[저장] stage3_triple_verification_summary.csv / triple_verification_matrix.csv")
    return tri


def main() -> None:
    write_b5_findings()

    section("B-6. train 세그먼트만 필터링 후 §3와 동일 절차 재실행")
    seg_full = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg_full = seg_full[seg_full["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

    mw_train_df = run_split_qexp(seg_full, split_value=1, split_name="train")
    compare_train_vs_full(mw_train_df)

    section("B-7. test 세그먼트만 필터링 후 §3와 동일 절차 재실행")
    mw_test_df = run_split_qexp(seg_full, split_value=0, split_name="test")

    triple_verification(mw_test_df)

    section("완료 — 해석 가이드")
    print(
        "1) test는 원래 표본이 작으므로(전체 177개 중 test는 51개 안팎), 채널별로\n"
        "   n_anomaly<5인 조합이 나올 수 있다. 이 경우 sig_test=NaN(판정불가)로\n"
        "   표시되며, '유의하지 않다'가 아니라 '검정력 부족으로 판단 불가'임에 유의.\n"
        "2) all_three_agree가 대부분 True로 나오면, §3의 핵심 결론(level=confounder,\n"
        "   diff/diff2=이상 고유 시그니처)이 전체/train/test 세 슬라이스 모두에서\n"
        "   독립적으로 재현된다는 뜻이며, 이는 Layer 2에서 탐지 성능에 대해 수행했던\n"
        "   것과 동일한 강도의 검증을 인과분석 결과에도 적용한 것이다.\n"
        "3) B-5(라벨링 프로토콜) 결과와 B-6/B-7(train/test 재검증) 결과를 함께 보면,\n"
        "   '체계적 라벨링 편향의 가능성은 제한적'이라는 정성적 결론과 '핵심 결론이\n"
        "   표본 분할에 따라 흔들리지 않는다'는 정량적 결론이 상호 보강된다."
    )


if __name__ == "__main__":
    main()
