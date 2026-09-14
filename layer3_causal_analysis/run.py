"""
layer3_causal_analysis/run.py
=============================================================================
Entry point for the full Layer 3 causal-signature pipeline (README §Layer 3),
invoked as `python -m layer3_causal_analysis.run`.

Execution order is dependency-driven, not just the file-listing order in the
repository tree:

  1. onset_mc_propagation.py          §2-2 / README (a) — Monte Carlo onset
                                       posterior propagation. Produces
                                       stage2_mc_draws_raw.csv and
                                       stage2_mc_segment_summary_corrected.csv,
                                       which every later script reads.
  2. temporal_precedence_test.py      §2-3 / README (b) — pooled Wilcoxon
                                       level-vs-diff-vs-diff2 precedence test.
                                       Produces stage2_precedence_crossing_times.csv,
                                       consumed by step 3.
  3. supplementary_diagnostics_A_D.py README (e), part 1 — persistence
                                       classification, early-window re-test,
                                       normal-segment control, CMH test. Needs
                                       (i) the exact fitted per-channel Kalman
                                       filters (rebuilt below, identically to
                                       how onset_mc_propagation.py builds them)
                                       and (ii) step 2's crossing-time output.
  4. quasi_experimental_placebo.py    §3 / README (c) — same-channel placebo
                                       pool comparison; the single most
                                       decisive test in the paper (Fig. 3a).
  5. labeling_protocol_audit.py       README (e), part 2 — B-5 external
                                       labeling-protocol audit + B-6/B-7
                                       train-only/test-only reproduction.
                                       Needs step 4's full-data Mann-Whitney
                                       CSV already on disk (it reads it back
                                       to build the train/full/test-only
                                       triple-comparison table).
  6. meta_analysis_random_effects.py  §2-4 / README (d) — DerSimonian-Laird
                                       random-effects pooling + I² across the
                                       5 channels.
  7. loo_sensitivity.py               README (d), continued — leave-one-
                                       channel-out sensitivity re-analysis of
                                       step 6's pooled estimates.
  8. scm_skeleton.py                  Fig. 4a — final synthesis. Performs no
                                       new statistical test; only reads
                                       artifacts already written by steps
                                       2-7 and assembles the Path A SCM
                                       skeleton + evidence tables.

Every step writes its own CSV/txt artifacts to OUT_DIR (see each script's
docstring for the exact filenames); this file does not pass data between
steps in memory except for the one deliberate exception in step 3 below.
"""

from __future__ import annotations

import pandas as pd

from _shared import (
    SEGMENTS_PATH, OUT_DIR, FINAL_CHANNELS, section,
    profile_channel, hierarchical_shrink_variance, LocalLinearTrendKF,
)

import onset_mc_propagation
import temporal_precedence_test
import supplementary_diagnostics_A_D
import quasi_experimental_placebo
import labeling_protocol_audit
import meta_analysis_random_effects
import loo_sensitivity
import scm_skeleton


def _build_locked_channel_kfs() -> dict[str, LocalLinearTrendKF]:
    """Rebuild the per-channel Kalman filters exactly as
    onset_mc_propagation.main() does internally (profile -> hierarchical
    shrinkage -> locked KF params). supplementary_diagnostics_A_D.main()
    takes an explicit `kf_by_channel` argument rather than re-deriving noise
    estimates itself (see its `get_kf()` / `_kf_cache`), so run.py is the
    natural place to construct it once and inject it — this keeps the two
    scripts decoupled while guaranteeing they use the identical filter."""
    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

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
        ch: (profiles[ch].quantization_step ** 2 / 12.0) if pd.notna(profiles[ch].quantization_step) else 0.0
        for ch in FINAL_CHANNELS
    }
    return {
        ch: LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch], quantization_floor=quant_floor[ch])
        for ch in FINAL_CHANNELS
    }


def main() -> None:
    section("LAYER 3 — 인과 시그니처 분석 파이프라인 시작 (README §Layer 3 전체)")

    section("STEP 1/8 — onset_mc_propagation.py  (§2-2, onset 불확실성 Monte Carlo 전파)")
    onset_mc_propagation.main()

    section("STEP 2/8 — temporal_precedence_test.py  (§2-3, level/diff/diff² 선행성 Wilcoxon 검정)")
    temporal_precedence_test.main()

    section("STEP 3/8 — supplementary_diagnostics_A_D.py  (보완진단 A~D)")
    kf_by_channel = _build_locked_channel_kfs()
    supplementary_diagnostics_A_D.main(kf_by_channel=kf_by_channel)

    section("STEP 4/8 — quasi_experimental_placebo.py  (§3, 위약 풀 준실험 비교, Fig. 3a)")
    quasi_experimental_placebo.main()

    section("STEP 5/8 — labeling_protocol_audit.py  (B-5 라벨링 프로토콜 감사 + B-6/B-7 train/test 재검증)")
    labeling_protocol_audit.main()

    section("STEP 6/8 — meta_analysis_random_effects.py  (§2-4, DerSimonian-Laird 랜덤효과 메타분석)")
    meta_analysis_random_effects.main()

    section("STEP 7/8 — loo_sensitivity.py  (leave-one-channel-out 민감도 분석)")
    loo_sensitivity.main()

    section("STEP 8/8 — scm_skeleton.py  (최종 SCM 뼈대 종합, Path A 확정, Fig. 4a)")
    scm_skeleton.main()

    section("LAYER 3 파이프라인 완료")
    print(f"모든 산출물은 다음 경로에 저장되었습니다: {OUT_DIR}")


if __name__ == "__main__":
    main()
