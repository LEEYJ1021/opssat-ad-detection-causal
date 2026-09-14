"""run.py 단위테스트.

run.py는 (1) 1단계 잠금 설정 그대로 채널별 Kalman filter를 재구성하는
_build_locked_channel_kfs()와 (2) 8개 스크립트를 정해진 순서로 호출하는
orchestration(main())으로 이루어진다.

(1)은 합성 segments.csv로 실제 계산 경로를 태워 검증하고, (2)는 실제 8개
파이프라인 전체를 돌리는 대신(무겁고 실데이터 필요) 각 서브모듈의 main()을
mock으로 치환해 "정확히 한 번씩, 문서화된 순서대로 호출되는가"만 검증한다
— 이것이 오케스트레이션 코드에 대한 올바른 단위테스트 범위다."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import run as run_module
from _shared import LocalLinearTrendKF, FINAL_CHANNELS


class TestBuildLockedChannelKfs:
    def test_returns_one_kf_per_final_channel(self, monkeypatch, synthetic_segments_csv):
        monkeypatch.setattr(run_module, "SEGMENTS_PATH", synthetic_segments_csv)
        kfs = run_module._build_locked_channel_kfs()
        assert set(kfs.keys()) == set(FINAL_CHANNELS)
        for ch, kf in kfs.items():
            assert isinstance(kf, LocalLinearTrendKF)
            assert kf.q > 0
            assert kf.r_eff > 0

    def test_kf_runs_without_error_on_a_segment(self, monkeypatch, synthetic_segments_csv,
                                                  synthetic_segments_df, final_channels):
        monkeypatch.setattr(run_module, "SEGMENTS_PATH", synthetic_segments_csv)
        kfs = run_module._build_locked_channel_kfs()
        ch = final_channels[0]
        one_seg = synthetic_segments_df[
            (synthetic_segments_df["channel"] == ch) & (synthetic_segments_df["anomaly"] == 1)
        ]
        first_sid = one_seg["segment"].iloc[0]
        values = one_seg[one_seg["segment"] == first_sid].sort_values("timestamp")["value"].values
        innovations, innovation_vars, _ = kfs[ch].run(values)
        assert innovations.shape == values.shape
        assert np.all(innovation_vars > 0)


class TestMainOrchestration:
    def test_calls_all_eight_stages_in_documented_order(self, monkeypatch, tmp_path):
        call_order = []

        def make_mock(name):
            m = MagicMock()
            m.side_effect = lambda *a, **kw: call_order.append(name)
            return m

        for mod_name in ["onset_mc_propagation", "temporal_precedence_test",
                         "supplementary_diagnostics_A_D", "quasi_experimental_placebo",
                         "labeling_protocol_audit", "meta_analysis_random_effects",
                         "loo_sensitivity", "scm_skeleton"]:
            monkeypatch.setattr(getattr(run_module, mod_name), "main", make_mock(mod_name))

        # _build_locked_channel_kfs()는 실제 segments.csv를 읽으므로 가짜 결과로 대체
        monkeypatch.setattr(run_module, "_build_locked_channel_kfs", lambda: {"FAKE": object()})
        monkeypatch.setattr(run_module, "OUT_DIR", tmp_path)

        run_module.main()

        assert call_order == [
            "onset_mc_propagation",
            "temporal_precedence_test",
            "supplementary_diagnostics_A_D",
            "quasi_experimental_placebo",
            "labeling_protocol_audit",
            "meta_analysis_random_effects",
            "loo_sensitivity",
            "scm_skeleton",
        ]

    def test_supplementary_diagnostics_receives_locked_kfs(self, monkeypatch, tmp_path):
        received = {}

        def fake_supp_main(kf_by_channel=None):
            received["kf_by_channel"] = kf_by_channel

        for mod_name in ["onset_mc_propagation", "temporal_precedence_test",
                         "quasi_experimental_placebo", "labeling_protocol_audit",
                         "meta_analysis_random_effects", "loo_sensitivity", "scm_skeleton"]:
            monkeypatch.setattr(getattr(run_module, mod_name), "main", MagicMock())
        monkeypatch.setattr(run_module.supplementary_diagnostics_A_D, "main", fake_supp_main)

        fake_kfs = {"CADC0872": object()}
        monkeypatch.setattr(run_module, "_build_locked_channel_kfs", lambda: fake_kfs)
        monkeypatch.setattr(run_module, "OUT_DIR", tmp_path)

        run_module.main()
        assert received["kf_by_channel"] is fake_kfs
