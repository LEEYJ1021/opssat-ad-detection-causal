"""labeling_protocol_audit.py 핵심 로직 단위테스트.

B-5(write_b5_findings)는 정적 텍스트를 파일로 쓰는지만 확인하고, B-6/B-7의
핵심 로직인 compare_train_vs_full / triple_verification은 quasi_experimental_
placebo.py가 만드는 Mann-Whitney 결과 CSV(stage3_qexp_mannwhitney*.csv)를
읽어 비교하므로, OUT_DIR을 tmp_path로 monkeypatch하고 그 안에 합성 CSV를
미리 심어둔 뒤 로직만 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import labeling_protocol_audit as lpa


def _mw_df(channel, feature, p_bonf, med_anom, med_placebo):
    return pd.DataFrame([{
        "channel": channel, "feature": feature, "n_anomaly": 10, "n_placebo": 50,
        "u_stat": 1.0, "p_value": p_bonf, "median_anomaly": med_anom,
        "median_placebo": med_placebo, "note": "", "p_value_bonferroni": p_bonf,
    }])


class TestWriteB5Findings:
    def test_writes_markdown_with_key_sections(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        lpa.write_b5_findings()
        out_path = tmp_path / "labeling_protocol_audit_findings.md"
        assert out_path.exists()
        text = out_path.read_text(encoding="utf-8")
        assert "OXI" in text
        assert "B-5" in text


class TestCompareTrainVsFull:
    def test_missing_full_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        mw_train_df = _mw_df("CADC0872", "diff_logvar", 0.01, 5.0, 1.0)
        assert lpa.compare_train_vs_full(mw_train_df) is None

    def test_same_significance_direction_flagged_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        full_df = _mw_df("CADC0872", "diff_logvar", 0.001, 6.0, 1.0)
        full_df.to_csv(tmp_path / "stage3_qexp_mannwhitney.csv", index=False)

        mw_train_df = _mw_df("CADC0872", "diff_logvar", 0.002, 5.5, 0.9)
        cmp = lpa.compare_train_vs_full(mw_train_df)
        assert cmp is not None
        assert cmp.iloc[0]["same_significance_direction"] == True  # noqa: E712
        assert (tmp_path / "stage3_train_vs_full_comparison.csv").exists()

    def test_disagreement_flagged_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        full_df = _mw_df("CADC0872", "level_d_abs", 0.001, 6.0, 1.0)  # 유의
        full_df.to_csv(tmp_path / "stage3_qexp_mannwhitney.csv", index=False)

        mw_train_df = _mw_df("CADC0872", "level_d_abs", 0.9, 1.0, 1.0)  # 비유의
        cmp = lpa.compare_train_vs_full(mw_train_df)
        assert cmp.iloc[0]["same_significance_direction"] == False  # noqa: E712


class TestTripleVerification:
    def test_missing_prereq_files_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        mw_test_df = _mw_df("CADC0872", "diff_logvar", 0.01, 5.0, 1.0)
        assert lpa.triple_verification(mw_test_df) is None

    def test_all_three_agree_when_all_significant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        full_df = _mw_df("CADC0872", "diff_logvar", 0.001, 6.0, 1.0)
        train_df = _mw_df("CADC0872", "diff_logvar", 0.002, 5.5, 0.9)
        full_df.to_csv(tmp_path / "stage3_qexp_mannwhitney.csv", index=False)
        train_df.to_csv(tmp_path / "stage3_qexp_mannwhitney_train_only.csv", index=False)

        mw_test_df = _mw_df("CADC0872", "diff_logvar", 0.003, 5.8, 1.1)
        tri = lpa.triple_verification(mw_test_df)
        assert tri is not None
        row = tri.iloc[0]
        assert row["sig_full"] and row["sig_train"] and row["sig_test"]
        assert row["all_three_agree"] == True  # noqa: E712
        assert (tmp_path / "stage3_triple_verification_summary.csv").exists()
        assert (tmp_path / "triple_verification_matrix.csv").exists()

    def test_nan_slice_gives_undecidable_not_disagreement(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lpa, "OUT_DIR", tmp_path)
        full_df = _mw_df("CADC0894", "level_d_abs", 0.9, 1.0, 1.0)   # 비유의
        train_df = _mw_df("CADC0894", "level_d_abs", 0.9, 1.0, 1.0)  # 비유의
        full_df.to_csv(tmp_path / "stage3_qexp_mannwhitney.csv", index=False)
        train_df.to_csv(tmp_path / "stage3_qexp_mannwhitney_train_only.csv", index=False)

        # test 슬라이스는 표본부족으로 p_value가 NaN인 경우
        mw_test_df = _mw_df("CADC0894", "level_d_abs", np.nan, np.nan, np.nan)
        tri = lpa.triple_verification(mw_test_df)
        row = tri.iloc[0]
        assert pd.isna(row["sig_test"])
        assert pd.isna(row["all_three_agree"])  # 판정불가이지 '불일치'가 아님
