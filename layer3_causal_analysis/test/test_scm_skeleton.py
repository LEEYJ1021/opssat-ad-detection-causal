"""scm_skeleton.py 단위테스트 — 이 스크립트는 새 통계 검정을 하지 않고 앞선
단계의 결과를 종합하는 정적 요약이므로, (1) 근거표들의 구조가 일관적인지,
(2) main()이 예외 없이 실행되며 예상된 산출물 3종을 생성하는지를 검증한다."""

from __future__ import annotations

import pandas as pd
import pytest

import scm_skeleton as scm


class TestEvidenceTablesStructure:
    def test_evidence_rows_have_consistent_keys(self):
        keys = {frozenset(row.keys()) for row in scm.EVIDENCE_ROWS}
        assert len(keys) == 1  # 모든 행이 동일한 컬럼 집합을 가져야 함
        assert len(scm.EVIDENCE_ROWS) == 4

    def test_heterogeneity_table1_has_five_channels_worth_of_rows(self):
        channels_mentioned = {row["influential_channel"] for row in scm.HETEROGENEITY_TABLE1}
        assert len(channels_mentioned) >= 3  # 872/874/888/894 등 최소 여러 채널 등장

    def test_channel_final_table2_covers_all_final_channels(self):
        from _shared import FINAL_CHANNELS
        channels_in_table = {row["channel"] for row in scm.CHANNEL_FINAL_TABLE2}
        assert channels_in_table == set(FINAL_CHANNELS)

    def test_propositions_are_nonempty_strings(self):
        assert isinstance(scm.FINAL_PROPOSITION_KO, str) and len(scm.FINAL_PROPOSITION_KO) > 20
        assert isinstance(scm.FINAL_PROPOSITION_EN, str) and len(scm.FINAL_PROPOSITION_EN) > 20


class TestMainProducesExpectedArtifacts:
    def test_main_runs_and_writes_three_files(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(scm, "OUT_DIR", tmp_path)
        scm.main()
        capsys.readouterr()  # 콘솔 출력 소음 억제(assert 대상 아님)

        evidence_path = tmp_path / "stage3_scm_evidence_summary.csv"
        het_path = tmp_path / "stage3_scm_heterogeneity_synthesis.csv"
        prop_path = tmp_path / "stage3_scm_final_proposition.txt"
        assert evidence_path.exists()
        assert het_path.exists()
        assert prop_path.exists()

        evidence_df = pd.read_csv(evidence_path)
        assert len(evidence_df) == len(scm.EVIDENCE_ROWS)

        het_df = pd.read_csv(het_path)
        assert set(het_df["table"].unique()) == {"table1_feature_x_channel", "table2_channel_final"}

        prop_text = prop_path.read_text(encoding="utf-8")
        assert "PATH A" in prop_text or "경로 A" in prop_text or "Onset (BOCPD)" in prop_text
