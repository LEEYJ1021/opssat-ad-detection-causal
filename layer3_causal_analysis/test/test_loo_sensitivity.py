"""loo_sensitivity.py 핵심 로직 단위테스트 — leave-one-channel-out 효과크기/
유의비율 재계산과 영향력 채널 판정(flag_influential_channels).

run_loo_effect_size / run_loo_frac_significant는 OUT_DIR에 CSV를 쓰므로,
loo_sensitivity 모듈의 OUT_DIR을 tmp_path로 monkeypatch해 실제 결과 폴더를
건드리지 않게 한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import loo_sensitivity as loo


@pytest.fixture
def synthetic_reliable_df(final_channels):
    """FEATURES(level/diff/diff2) x 5채널 각 6세그먼트를 갖는 합성
    reliable(=high/medium tier) DataFrame. 한 채널(final_channels[0])만 값이
    확연히 커서 leave-one-out 시 pooled 값이 크게 흔들리도록 구성한다."""
    rng = np.random.default_rng(0)
    rows = []
    influential_ch = final_channels[0]
    for ch in final_channels:
        base = 8.0 if ch == influential_ch else 1.0
        for seg in range(6):
            rows.append({
                "channel": ch, "segment": seg, "reliability_tier": "high",
                "level_d_abs_median": base + rng.normal(0, 0.1),
                "diff_d_abs_median": base + rng.normal(0, 0.1),
                "diff2_d_abs_median": base + rng.normal(0, 0.1),
                "level_frac_significant": min(0.95 if ch == influential_ch else 0.3
                                               + rng.normal(0, 0.02), 0.99),
                "diff_frac_significant": min(0.95 if ch == influential_ch else 0.3
                                              + rng.normal(0, 0.02), 0.99),
                "diff2_frac_significant": min(0.95 if ch == influential_ch else 0.3
                                               + rng.normal(0, 0.02), 0.99),
            })
    return pd.DataFrame(rows)


class TestRunLooEffectSize(object):
    def test_excluding_influential_channel_shifts_pooled_more(self, tmp_path, monkeypatch,
                                                                synthetic_reliable_df, final_channels):
        monkeypatch.setattr(loo, "OUT_DIR", tmp_path)
        result = loo.run_loo_effect_size(synthetic_reliable_df)
        influential_ch = final_channels[0]

        full_rows = result[(result["feature"] == "level") &
                            (result["excluded_channel"] == "(none, 전체포함)")]
        excl_influential = result[(result["feature"] == "level") &
                                   (result["excluded_channel"] == influential_ch)]
        excl_other = result[(result["feature"] == "level") &
                             (result["excluded_channel"] == final_channels[1])]

        assert len(full_rows) == 1 and len(excl_influential) == 1 and len(excl_other) == 1
        shift_influential = abs(excl_influential.iloc[0]["pooled_shift_from_full"])
        shift_other = abs(excl_other.iloc[0]["pooled_shift_from_full"])
        # 영향력 있는 채널을 뺐을 때가 다른 채널을 뺐을 때보다 pooled가 더 크게 움직여야 함
        assert shift_influential > shift_other
        assert (tmp_path / "stage2_meta_loo_effect_size.csv").exists()


class TestRunLooFracSignificant(object):
    def test_output_written_and_columns_present(self, tmp_path, monkeypatch, synthetic_reliable_df):
        monkeypatch.setattr(loo, "OUT_DIR", tmp_path)
        result = loo.run_loo_frac_significant(synthetic_reliable_df)
        assert (tmp_path / "stage2_meta_loo_frac_significant.csv").exists()
        assert {"pooled_proportion", "ci_low_proportion", "ci_high_proportion"}.issubset(result.columns)
        assert result["pooled_proportion"].between(0, 1).all()


class TestFlagInfluentialChannels:
    def test_full_row_never_flagged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loo, "OUT_DIR", tmp_path)
        loo_effect_df = pd.DataFrame([
            {"feature": "level", "excluded_channel": "(none, 전체포함)", "pooled": 1.0,
             "ci_low": 0.5, "ci_high": 1.5, "I2_drop_from_full": 0.0,
             "pooled_shift_from_full": 0.0},
            {"feature": "level", "excluded_channel": "CADC0872", "pooled": 5.0,
             "ci_low": 0.5, "ci_high": 1.5, "I2_drop_from_full": 50.0,
             "pooled_shift_from_full": 4.0},
        ])
        loo_frac_df = pd.DataFrame(columns=loo_effect_df.columns.tolist() +
                                    ["pooled_proportion", "ci_low_proportion", "ci_high_proportion"])
        loo_frac_df = pd.DataFrame([
            {"feature": "diff", "excluded_channel": "(none, 전체포함)", "pooled_proportion": 0.5,
             "ci_low_proportion": 0.3, "ci_high_proportion": 0.7, "I2_drop_from_full": 0.0,
             "pooled_shift_from_full": 0.0},
        ])

        # FEATURES 상수를 이 테스트 데이터에 맞춰 monkeypatch
        monkeypatch.setattr(loo, "FEATURES", ["level"])
        influential_df = loo.flag_influential_channels(
            loo_effect_df[loo_effect_df["feature"] == "level"],
            loo_frac_df[loo_frac_df["feature"] == "diff"].assign(feature="level"),
        )
        # CADC0872를 뺐을 때: I2가 50%p 이상 급락(threshold=30) + pooled(5.0)가
        # 전체 CI(0.5~1.5) 밖 -> influential=True로 판정돼야 함
        row = influential_df[influential_df["excluded_channel"] == "CADC0872"].iloc[0]
        assert row["influential"] == True  # noqa: E712
        assert (tmp_path / "stage2_meta_loo_influential_flags.csv").exists()

    def test_small_shift_not_flagged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loo, "OUT_DIR", tmp_path)
        monkeypatch.setattr(loo, "FEATURES", ["level"])
        loo_effect_df = pd.DataFrame([
            {"feature": "level", "excluded_channel": "(none, 전체포함)", "pooled": 1.0,
             "ci_low": 0.5, "ci_high": 1.5, "I2_drop_from_full": 0.0,
             "pooled_shift_from_full": 0.0},
            {"feature": "level", "excluded_channel": "CADC0873", "pooled": 1.05,
             "ci_low": 0.5, "ci_high": 1.5, "I2_drop_from_full": 2.0,
             "pooled_shift_from_full": 0.05},
        ])
        loo_frac_df = pd.DataFrame([
            {"feature": "level", "excluded_channel": "(none, 전체포함)", "pooled_proportion": 0.5,
             "ci_low_proportion": 0.3, "ci_high_proportion": 0.7, "I2_drop_from_full": 0.0,
             "pooled_shift_from_full": 0.0},
            {"feature": "level", "excluded_channel": "CADC0873", "pooled_proportion": 0.52,
             "ci_low_proportion": 0.3, "ci_high_proportion": 0.7, "I2_drop_from_full": 1.0,
             "pooled_shift_from_full": 0.02},
        ])
        influential_df = loo.flag_influential_channels(loo_effect_df, loo_frac_df)
        row = influential_df[influential_df["excluded_channel"] == "CADC0873"]
        assert not row["influential"].any()
