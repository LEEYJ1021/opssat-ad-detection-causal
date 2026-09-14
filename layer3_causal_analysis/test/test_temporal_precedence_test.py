"""temporal_precedence_test.py 핵심 로직 단위테스트 — 첫 이탈시점 계산
(analyze_segment_precedence)과 짝비교 Wilcoxon 검정(run_pairwise_test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from temporal_precedence_test import analyze_segment_precedence
from _shared import BURN_IN, MIN_WINDOW_POINTS

# 참고: run_pairwise_test는 temporal_precedence_test.py의 main() 내부에
# 지역함수로 정의돼 있어 모듈 레벨로 import할 수 없다. 로직이 동일한
# supplementary_diagnostics_A_D.run_pairwise_test(모듈 레벨 함수)로 그 짝비교
# Wilcoxon 로직을 test_supplementary_diagnostics_A_D.py에서 검증한다.


class TestAnalyzeSegmentPrecedence:
    def test_diff_crosses_while_level_stays_within_threshold(self):
        # onset 이후 값이 +a/-a로 빠르게 진동하는 신호를 구성한다. 진동
        # 진폭 a를 pre-level 표준편차의 2.5배로 잡으면 level의 |z|는 항상
        # 2.5(<3, 미이탈)에 머물지만, 연속한 두 진동점 사이의 diff는 2a라서
        # 그 z는 sqrt(2)*2.5 ≈ 3.54(>3)로 diff만 이탈한다 — "level은 그대로인데
        # diff/diff2만 먼저 튄다"는 §2-3 핵심 주장을 가장 단순한 형태로 재현.
        onset_idx = 60
        n_post = 80
        rng = np.random.default_rng(0)
        pre_std = 0.05
        pre = rng.normal(0, pre_std, onset_idx)
        a = 2.5 * pre_std
        post = np.array([a if i % 2 == 0 else -a for i in range(n_post)])
        values = np.concatenate([pre, post])

        result = analyze_segment_precedence(values, sampling=1.0, onset_idx=onset_idx)
        assert np.isnan(result["level_cross_sec"])       # level은 이탈하지 않음
        assert not np.isnan(result["diff_cross_sec"])     # diff는 즉시 이탈

    def test_no_crossing_gives_nan(self):
        # 완전히 평탄하고 변화 없는 신호 -> 아무 것도 임계값을 못 넘어야 함
        n = 100
        onset_idx = 50
        values = np.zeros(n) + np.random.default_rng(1).normal(0, 1e-6, n)
        result = analyze_segment_precedence(values, sampling=1.0, onset_idx=onset_idx)
        assert np.isnan(result["level_cross_sec"])
        assert np.isnan(result["diff_cross_sec"])
        assert np.isnan(result["diff2_cross_sec"])

    def test_crossing_time_scales_with_sampling_rate(self):
        n = 200
        onset_idx = 80
        rng = np.random.default_rng(2)
        pre = rng.normal(0, 0.1, onset_idx)
        post = rng.normal(20, 0.1, n - onset_idx)  # 즉각적인 level 이동
        values = np.concatenate([pre, post])

        r1 = analyze_segment_precedence(values, sampling=1.0, onset_idx=onset_idx)
        r2 = analyze_segment_precedence(values, sampling=2.5, onset_idx=onset_idx)
        assert not np.isnan(r1["level_cross_sec"])
        assert r2["level_cross_sec"] == pytest.approx(r1["level_cross_sec"] * 2.5)

