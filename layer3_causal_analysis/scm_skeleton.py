"""
layer3_causal_analysis/scm_skeleton.py
=============================================================================
Structural causal model (SCM) skeleton confirmation — Path A vs. Path B
(README §Layer 3, Fig. 4a; "Relationship to formal causal inference" scope
statement).

배경
-----------------------------------------------------------------------------
§2-2(onset_mc_propagation.py, 내부 pre/post), §2-3(temporal_precedence_test.py,
선행성 CMH), §2-4(meta_analysis_random_effects.py + loo_sensitivity.py,
이질성), §3-준실험(quasi_experimental_placebo.py, 위약 풀 대조)까지 네 갈래의
독립적 검증을 모두 거친 뒤에야 SCM 뼈대를 확정한다. 원래 계획했던
"level → diff → diff² 순서의 5채널 공통 SCM"은 §3 준실험 결과로 완전히
폐기하고, 아래 구조로 교체한다.

이 스크립트는 새로운 통계 검정을 수행하지 않는다 — 재계산 없이 앞선 4개
스크립트의 산출물을 읽어 하나의 "최종 판정" 문서/표로 종합하는 것이 목적이다
(numbers must trace to onset_mc_propagation.py / temporal_precedence_test.py /
meta_analysis_random_effects.py+loo_sensitivity.py / quasi_experimental_placebo.py
— see the citations inline below).

최종 SCM 뼈대 (경로 A, 신호처리 구조 기반):

    Onset (BOCPD 확률적 변화점)
          |
          v
    Diff/Diff² 분산 급증 (transient, onset 직후 국한)
          |  moderator: float_noise_suspect(872/873/874) 강함
          |             quantized(888/894) 약함
          v  (약한/불확실한 파생 경로 — 인과 사슬의 핵심 아님)
    Level 변화 (persistent, 그러나 confounded)

Level은 인과 사슬의 "결과 변수"로 취급하지 않는다 — 정상 세그먼트에서도
동일한 정도의 drift가 나타나므로(quasi_experimental_placebo.py, level MW
p_bonf=1.000 전 채널), level 변화는 "이상이 만든 결과"가 아니라 "채널이
원래 갖고 있는 완만한 추세(confounder)와 혼재된 관찰"로 다룬다.

경로 A(신호처리 구조 기반)를 채택하고 경로 B(TLE/자세 quaternion 등 물리
기반 SCM)를 보류하는 이유는 (1) 보조 데이터(TLE, 자세 quaternion) 존재
여부가 미확인 상태이고 (2) 경로 A의 결론이 물리적으로 그럴듯한지 사전
검토가 안 된 상태이기 때문이다 — 경로 B는 확장 가능성으로 남겨둔다.

출력:
  - stage3_scm_evidence_summary.csv        (단계별 근거 요약: 어느 스크립트의
                                             어느 산출물이 SCM의 어느 edge를 뒷받침하는지)
  - stage3_scm_heterogeneity_synthesis.csv (Feature x Channel 이질성 원인 및
                                             최종 판정 — Table 1/2 통합)
  - stage3_scm_final_proposition.txt       (논문 서술용 최종 SCM 명제, 한글/영문)
"""

from __future__ import annotations

import pandas as pd

from _shared import OUT_DIR, section

# =============================================================================
# [표 A] 단계별 근거 요약 — 어느 edge를, 어느 스크립트/산출물이 뒷받침하는가
# =============================================================================

EVIDENCE_ROWS = [
    {
        "evidence_id": "(1) 시간적 선행성",
        "source_script": "temporal_precedence_test.py",
        "source_artifact": "stage2_precedence_test_pooled.csv (CMH: "
                            "supplementary_diagnostics_A_D.py stage2_diagD_group_proportion_tests_cmh.csv)",
        "finding": "diff/diff2가 level보다 먼저 임계값을 넘음 "
                   "(pooled Wilcoxon p<0.001, 전체 CMH p<0.001, 채널별 기저율 통제 후에도 유지)",
        "scm_implication": "순서상 diff/diff2가 선행 -> onset -> diff/diff2 edge 지지",
    },
    {
        "evidence_id": "(2) 효과 지속성 프로파일",
        "source_script": "supplementary_diagnostics_A_D.py [A]",
        "source_artifact": "stage2_diagA_diff_temporal_profile.csv",
        "finding": "diff/diff2: 대다수가 transient(짧고 뾰족한 스파이크 후 감쇠); "
                   "level: persistent와 transient가 혼재(약 50:50)",
        "scm_implication": "diff/diff2 = onset 직후 국한된 반응, level = 느린 추세라는 질적 차이 "
                            "-> 두 변수를 같은 인과 층위에 둘 수 없음",
    },
    {
        "evidence_id": "(3) 채널 간 이질성",
        "source_script": "meta_analysis_random_effects.py + loo_sensitivity.py",
        "source_artifact": "heterogeneity_summary.csv, stage2_meta_loo_influential_flags.csv",
        "finding": "diff/diff2의 pooled frac_significant I2=92~94%(매우 높음); "
                   "level의 큰 효과크기(872)는 onset 불확실성·분모 왜소가 아닌 실제 신호(보완 G/H)",
        "scm_implication": "'5채널 공통 크기'로 뭉뚱그릴 수 없음 -> 채널 유형(moderator)을 "
                            "diff/diff2 edge에 명시적으로 부착해야 함",
    },
    {
        "evidence_id": "(4) 준실험적 정상군 대조 [결정적 근거]",
        "source_script": "quasi_experimental_placebo.py",
        "source_artifact": "stage3_qexp_mannwhitney.csv / placebo_comparison.csv",
        "finding": "level: Mann-Whitney U 전 채널 비유의(p_bonf=1.000), median_anomaly가 "
                   "median_placebo보다 오히려 작은 채널 다수; "
                   "diff/diff2: 전 채널 강한 유의(p_bonf 최대 4e-6, 대부분 <1e-15)",
        "scm_implication": "level = confounder로 재분류(이상 특유의 레벨 이동이라는 근거 소멸); "
                            "diff/diff2 = 정상 세그먼트에서 재현되지 않는 이상 고유 시그니처로 채택",
    },
]

# =============================================================================
# [표 B] Feature x Channel 이질성 원인 및 최종 판정 (Table 1 in dev-log)
# =============================================================================

HETEROGENEITY_TABLE1 = [
    {"feature": "level", "influential_channel": "CADC0872 (크기)",
     "cause": "실제 신호(onset 불확실성·분모 왜소 아님, 보완 G/H로 확인)",
     "qexp_verdict": "비유의(전 채널) -> confounder로 재분류"},
    {"feature": "level", "influential_channel": "CADC0888 (유의성)",
     "cause": "888만 유의비율 0.55, 나머지 0.96~0.98로 이질적",
     "qexp_verdict": "비유의(전 채널) -> confounder로 재분류"},
    {"feature": "diff2", "influential_channel": "CADC0874 (크기)",
     "cause": "transient_ratio와 완벽한 역상관(r=-1.0) -> 검정설계상 과소추정 가능성(방법론적 아티팩트)",
     "qexp_verdict": "강한 유의(전 채널) -> 인과 시그니처로 채택"},
    {"feature": "diff/diff2", "influential_channel": "CADC0894 (유의성)",
     "cause": "894만 frac_sig 0.81~0.86으로 압도적, 나머지 4채널은 0.001~0.18의 자연스러운 위계 존재",
     "qexp_verdict": "강한 유의(전 채널, 단 894 내부에서는 §3 준실험 기준 나머지와 유사한 패턴으로 수렴)"},
]

# =============================================================================
# [표 C] 채널별 최종 종합 판정 (Table 2 in dev-log)
# =============================================================================

CHANNEL_FINAL_TABLE2 = [
    {"channel": "CADC0872", "channel_type": "float_noise_suspect",
     "level_specificity": "크기 이상치(과대) -> 실제 신호로 확인",
     "diff_diff2_specificity": "diff/diff2 유의비율 최상위권(0.78~0.80)",
     "final_interpretation": "diff/diff2 시그니처 강함, level은 confounder"},
    {"channel": "CADC0873", "channel_type": "float_noise_suspect",
     "level_specificity": "평이함",
     "diff_diff2_specificity": "diff/diff2 유의비율 최상위권(0.89)",
     "final_interpretation": "diff/diff2 시그니처 매우 강함"},
    {"channel": "CADC0874", "channel_type": "float_noise_suspect",
     "level_specificity": "평이함",
     "diff_diff2_specificity": "diff2 효과크기 과소추정 가능성(검정설계 아티팩트, [F-2])",
     "final_interpretation": "diff2 실제 시그니처는 표 상 수치보다 강할 가능성"},
    {"channel": "CADC0888", "channel_type": "quantized",
     "level_specificity": "유의성 이상치(과소) -> level 자체가 노이즈성 지표",
     "diff_diff2_specificity": "diff/diff2 유의비율 상대적으로 약함(0.12~0.42)",
     "final_interpretation": "5채널 중 diff/diff2 시그니처가 가장 약함"},
    {"channel": "CADC0894", "channel_type": "quantized",
     "level_specificity": "평이함",
     "diff_diff2_specificity": "frac_significant 유의성 이상치(§2 내부 검정에서 압도적으로 높음)",
     "final_interpretation": "§2 내부 검정에서는 특이했으나 §3 준실험에서는 나머지와 유사한 패턴(정상 대비 유의)"},
]

FINAL_PROPOSITION_KO = (
    "OPSSAT-AD의 이상은 신호 레벨 자체의 지속적 이동이 아니라, onset 직후 국한된 "
    "변화율·가속도(diff/diff²)의 일시적 분산 급증으로 특징지어지며, 이 시그니처는 "
    "같은 채널의 정상 세그먼트에서는 재현되지 않는 이상 고유의 패턴이다. 신호 레벨의 "
    "변화는 채널이 원래 갖는 완만한 드리프트와 혼재되어 있어 인과적 결과변수가 아니라 "
    "혼란변수로 취급해야 하며, 이 시그니처의 강도는 채널 유형(float_noise_suspect > "
    "quantized)에 따라 이질적이다."
)

FINAL_PROPOSITION_EN = (
    "The OPS-SAT-AD anomaly signature is not a persistent shift in signal level, but "
    "a transient, onset-localized surge in first- and second-difference variance "
    "(diff/diff2), which is not reproduced in same-channel normal-operation segments. "
    "Level shift co-occurs with the channel's ordinary drift and must be treated as a "
    "confounder rather than a causal outcome variable; the strength of the diff/diff2 "
    "signature is heterogeneous, moderated by channel noise regime "
    "(float_noise_suspect > quantized)."
)

CORE_CONCLUSION = (
    "level 차원의 이질성(872 크기, 888 유의성)은 §3 준실험에서 level 자체가 인과적으로 "
    "무의미해짐에 따라 자연히 해소되는 반면, diff/diff2 차원의 이질성(874 크기, 894 "
    "유의성)은 인과 시그니처 '내부의' 강도 차이로 남아 채널별 조절변수(moderator)로 "
    "SCM에 반영해야 한다."
)

METHODOLOGICAL_NOTES = [
    "874의 diff2 효과크기는 [F-2]에서 방법론적 과소추정 가능성이 제기됨(전체창 평균 "
    "검정이 빠른 decay를 가장 심하게 희석) -> 정식 리비전에서는 조기창(early-window, "
    "supplementary_diagnostics_A_D.py [B]) 재검정치를 diff2 대표값으로 병기 권장.",
    "894는 §2(내부, onset_mc_propagation.py) 기준으로는 이질적이었으나 §3(준실험, "
    "quasi_experimental_placebo.py) 기준으로는 다른 4채널과 유사한 패턴으로 수렴 -> "
    "'이질성'의 정의가 비교 기준(내부 vs 대조군)에 따라 달라질 수 있다는 점을 논문 "
    "한계(limitation)로 명시할 것.",
    "이 요약표는 재계산 없이 기존 결과(§2-4, LOO, F-2, §3)를 정리한 것이므로, 원본 "
    "수치가 갱신되면 반드시 함께 갱신할 것.",
    "경로 B(TLE/자세 quaternion과 연결한 물리 기반 SCM) 착수 전, 보조 데이터(TLE, 자세 "
    "quaternion) 존재 여부 확인이 반드시 선행되어야 함 (labeling_protocol_audit.py의 "
    "B-5 외부조사와 동일 선상의 미해결 항목).",
]


def main() -> None:
    section("0. 경로 A/B 결정 근거 — 단계별 증거 요약")
    evidence_df = pd.DataFrame(EVIDENCE_ROWS)
    evidence_df.to_csv(OUT_DIR / "stage3_scm_evidence_summary.csv", index=False)
    print(evidence_df.to_string(index=False))

    section("1. [표 1] Feature x Channel 이질성 원인 및 최종 판정")
    het1_df = pd.DataFrame(HETEROGENEITY_TABLE1)
    print(het1_df.to_string(index=False))

    section("2. [표 2] 채널별 최종 종합 판정")
    het2_df = pd.DataFrame(CHANNEL_FINAL_TABLE2)
    print(het2_df.to_string(index=False))

    combined = pd.concat([
        het1_df.assign(table="table1_feature_x_channel"),
        het2_df.rename(columns={
            "channel": "influential_channel", "channel_type": "feature",
            "level_specificity": "cause", "diff_diff2_specificity": "qexp_verdict",
        }).assign(table="table2_channel_final"),
    ], ignore_index=True, sort=False)
    combined.to_csv(OUT_DIR / "stage3_scm_heterogeneity_synthesis.csv", index=False)

    section("3. 핵심 결론 (표 1·2를 관통하는 하나의 문장)")
    print(CORE_CONCLUSION)

    section("4. 최종 채택 SCM 명제 (논문 서술용)")
    print("[한글]\n" + FINAL_PROPOSITION_KO)
    print("\n[English]\n" + FINAL_PROPOSITION_EN)

    with open(OUT_DIR / "stage3_scm_final_proposition.txt", "w", encoding="utf-8") as f:
        f.write("SCM SKELETON — PATH A (signal-processing structure)\n")
        f.write("=" * 78 + "\n\n")
        f.write("Onset (BOCPD) -> Diff/Diff2 variance surge (transient, moderator: \n")
        f.write("channel noise regime) -> [weak/confounded] Level shift (persistent, \n")
        f.write("but not causal-core; see quasi_experimental_placebo.py)\n\n")
        f.write("-" * 78 + "\n[한글 명제]\n" + FINAL_PROPOSITION_KO + "\n\n")
        f.write("-" * 78 + "\n[English proposition]\n" + FINAL_PROPOSITION_EN + "\n\n")
        f.write("-" * 78 + "\n[핵심 결론]\n" + CORE_CONCLUSION + "\n\n")
        f.write("-" * 78 + "\n[남는 방법론적 주석 — 향후 리비전 시 참고]\n")
        for note in METHODOLOGICAL_NOTES:
            f.write(f"  - {note}\n")
    print(f"\n[저장] stage3_scm_final_proposition.txt")

    section("완료 — 경로 A 채택, 경로 B 보류")
    print(
        "경로 A(신호처리 구조 기반)만으로도 'diff/diff2 분산 급증'이라는 이상 고유\n"
        "시그니처와 그 채널별 조절효과까지 통계적으로 방어 가능한 수준으로 확정되었다.\n"
        "경로 B(TLE/자세 quaternion과 연결한 물리 기반 SCM)는 보조 데이터 존재 여부가\n"
        "미확인 상태이고, 경로 A 결론의 물리적 타당성 사전 검토도 안 된 상태이므로\n"
        "'확장 가능성'으로 남겨두고 현재 스코프는 경로 A로 확정한다."
    )


if __name__ == "__main__":
    main()
