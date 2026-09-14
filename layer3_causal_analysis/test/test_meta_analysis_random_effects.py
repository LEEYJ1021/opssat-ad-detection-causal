"""meta_analysis_random_effects.py 핵심 로직 단위테스트 — DerSimonian-Laird
랜덤효과 메타분석, arcsine 변환, I² 해석, 채널별 효과크기/분산 산출."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta_analysis_random_effects import (
    dl_meta_analysis, interpret_I2, arcsine_transform, arcsine_backtransform,
    channel_effect_and_variance,
)


# =============================================================================
# dl_meta_analysis
# =============================================================================

class TestDlMetaAnalysis:
    def test_too_few_studies_returns_nan_with_note(self):
        out = dl_meta_analysis(np.array([1.0]), np.array([0.1]), ["A"])
        assert out["k"] == 1
        assert np.isnan(out["pooled"])
        assert "study 수 부족" in out["note"]

    def test_homogeneous_studies_give_low_i2(self):
        # 5개 채널이 전부 같은 참값(effect=2.0) 주변에서 작은 표본오차만 갖는 경우
        yi = np.array([2.0, 2.05, 1.95, 2.02, 1.98])
        vi = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        out = dl_meta_analysis(yi, vi, ["A", "B", "C", "D", "E"])
        assert out["k"] == 5
        assert out["pooled"] == pytest.approx(2.0, abs=0.1)
        assert out["I2"] < 50.0   # 이질성 낮음

    def test_heterogeneous_studies_give_high_i2(self):
        # 채널마다 참값 자체가 크게 다른 경우(0.1, 5.0, 10.0, 0.2, 8.0) -> I2가 커야 함
        yi = np.array([0.1, 5.0, 10.0, 0.2, 8.0])
        vi = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        out = dl_meta_analysis(yi, vi, ["A", "B", "C", "D", "E"])
        assert out["I2"] > 75.0

    def test_filters_out_invalid_variances(self):
        yi = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        vi = np.array([0.1, 0.1, 0.1, -1.0, 0.0])  # nan/음수/0 -> 무효
        out = dl_meta_analysis(yi, vi, ["A", "B", "C", "D", "E"])
        assert out["k"] == 2  # A, B만 유효

    def test_ci_contains_pooled_estimate(self):
        yi = np.array([1.0, 1.5, 0.8, 1.2, 1.1])
        vi = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        out = dl_meta_analysis(yi, vi, ["A", "B", "C", "D", "E"])
        assert out["ci_low"] < out["pooled"] < out["ci_high"]


class TestInterpretI2:
    @pytest.mark.parametrize("i2,expected_substr", [
        (10.0, "낮음"), (40.0, "중간"), (60.0, "높음"), (90.0, "매우 높음"),
    ])
    def test_bucketing(self, i2, expected_substr):
        assert expected_substr in interpret_I2(i2)

    def test_nan_is_undecidable(self):
        assert interpret_I2(np.nan) == "판정불가"


# =============================================================================
# arcsine_transform / arcsine_backtransform — 왕복 변환 확인
# =============================================================================

class TestArcsineRoundTrip:
    @pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.8, 0.99])
    def test_roundtrip_recovers_original_proportion(self, p):
        transformed = arcsine_transform(np.array([p]))[0]
        recovered = arcsine_backtransform(transformed)
        assert recovered == pytest.approx(p, abs=1e-6)

    def test_extreme_proportions_are_clipped_not_nan(self):
        transformed = arcsine_transform(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(transformed))

    def test_backtransform_nan_input_is_nan(self):
        assert np.isnan(arcsine_backtransform(np.nan))


# =============================================================================
# channel_effect_and_variance
# =============================================================================

class TestChannelEffectAndVariance:
    def test_computes_mean_and_se_squared(self):
        df = pd.DataFrame({
            "channel": ["A", "A", "A", "B", "B"],
            "value": [1.0, 2.0, 3.0, 5.0, 7.0],
        })
        yi, vi, labels = channel_effect_and_variance(df, "value", ["A", "B"])
        assert labels == ["A", "B"]
        assert yi[0] == pytest.approx(2.0)  # mean(1,2,3)
        expected_se_a = np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3)
        assert vi[0] == pytest.approx(expected_se_a ** 2)

    def test_single_observation_channel_gives_nan(self):
        df = pd.DataFrame({"channel": ["A"], "value": [1.0]})
        yi, vi, labels = channel_effect_and_variance(df, "value", ["A", "B"])
        assert np.isnan(yi[0])
        assert np.isnan(yi[1])  # B는 데이터 자체가 없음

    def test_missing_channel_returns_nan_not_error(self):
        df = pd.DataFrame({"channel": ["A", "A"], "value": [1.0, 2.0]})
        yi, vi, labels = channel_effect_and_variance(df, "value", ["A", "Z"])
        assert np.isfinite(yi[0])
        assert np.isnan(yi[1])
