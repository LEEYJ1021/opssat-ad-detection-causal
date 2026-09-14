"""
layer2_anomaly_detection 스모크/회귀 테스트.

실 데이터(OPS-SAT-AD segments.csv) 없이도 돌아가는 두 종류의 검증을 한다:
    1. 합성 데이터로 파이프라인 각 단계가 오류 없이 끝까지 실행되는지 (스모크)
    2. channel_scoping.py의 MCC/Youden 계산 로직이 실제 OPS-SAT-AD 실행 결과
       (CHANNEL_SCOPE_REFERENCE)를 정확히 재현하는지 (회귀)

실행: cd layer2_anomaly_detection && python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared import (  # noqa: E402
    fit_channel_profiles, load_segments, make_synthetic_segments,
    LocalLinearTrendKF, hierarchical_shrink_variance, mcc_from_counts,
)
from bocpd_forgetting import BOCPD, detect_all_channels  # noqa: E402
from channel_scoping import (  # noqa: E402
    CHANNEL_SCOPE_REFERENCE, FINAL_SCOPE, build_channel_scope, channel_mcc_youden,
)
from bootstrap_ci import bootstrap_channel_mcc  # noqa: E402
from ablation_mixture_forgetting import run_ablation, COMBOS  # noqa: E402
from triple_verification import lock_channel_scope  # noqa: E402


# -----------------------------------------------------------------------------
# 1. 회귀 테스트 — 실제 OPS-SAT-AD 실행 결과와 정확히 일치해야 함
# -----------------------------------------------------------------------------
def test_channel_scoping_reproduces_reference_scope():
    scope = build_channel_scope(CHANNEL_SCOPE_REFERENCE)
    included = scope[scope["include_in_stage2"]]["channel"].tolist()
    assert included == FINAL_SCOPE


def test_channel_scoping_mcc_values_match_reference():
    scope = build_channel_scope(CHANNEL_SCOPE_REFERENCE).set_index("channel")
    # CADC0874가 5채널 중 가장 높은 MCC (0.7398)를 가져야 함
    assert scope.loc["CADC0874", "mcc"] == pytest.approx(0.739818, abs=1e-5)
    # CADC0892는 우연수준 이하로 제외 (MCC<=0)
    assert scope.loc["CADC0892", "mcc"] < 0
    assert scope.loc["CADC0892", "scope_reason"] == "우연수준 이하 (MCC<=0)"
    # CADC0884는 anomaly 표본이 없어 구조적 제외
    assert np.isnan(scope.loc["CADC0884", "mcc"])


def test_final_locked_scope_has_five_channels_with_890_note():
    scope = build_channel_scope(CHANNEL_SCOPE_REFERENCE)
    locked = lock_channel_scope(scope)
    included = locked[locked["include_in_stage2"]]["channel"].tolist()
    assert included == FINAL_SCOPE
    assert "강화" in locked.set_index("channel").loc["CADC0890", "scope_reason_final"]


def test_mcc_from_counts_matches_manual_formula():
    tp, fn, fp, tn = 40, 10, 5, 45
    expected = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert mcc_from_counts(tp, fn, fp, tn) == pytest.approx(expected)


def test_mcc_from_counts_zero_denominator_returns_zero():
    assert mcc_from_counts(0, 0, 0, 0) == 0.0


# -----------------------------------------------------------------------------
# 2. 스모크 테스트 — 합성 데이터로 파이프라인이 오류 없이 끝까지 도는지
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_segments():
    return make_synthetic_segments(n_segments_per_channel=20, seg_len=40)


def test_load_segments_falls_back_to_synthetic(tmp_path):
    seg, channels, is_synthetic = load_segments(tmp_path / "does_not_exist.csv")
    assert is_synthetic is True
    assert len(channels) > 0
    assert "train" in seg.columns


def test_bocpd_forgetting_detects_shift_after_long_quiet_run():
    """forgetting=True가 긴 정상 구간 뒤에 온 실제 변화점 탐지 민감도를 유지/개선하는지 확인
    (§1-6 ablation 결론 — forgetting을 끄면 float_noise_suspect 채널 recall이 급락함,
    874: 0.725 -> 0.319 —의 최소 재현). 긴 조용한 구간(kappa_max를 훨씬 초과) 뒤에
    레벨 이동을 주입하면, forgetting=True 쪽이 최소한 forgetting=False보다 더
    늦게 탐지하지는 않아야 한다."""
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1, size=250)
    shifted = rng.normal(6, 1, size=30)  # 뚜렷한 레벨 이동
    z = np.concatenate([quiet, shifted])

    bocpd_forget = BOCPD(use_forgetting=True)
    bocpd_noforget = BOCPD(use_forgetting=False)
    r_forget = bocpd_forget.run(z)
    r_noforget = bocpd_noforget.run(z)

    assert r_forget["onset_idx"] is not None
    if r_noforget["onset_idx"] is not None:
        assert r_forget["onset_idx"] <= r_noforget["onset_idx"] + 5


def test_fit_channel_profiles_runs_on_synthetic(synthetic_segments):
    channels = list(synthetic_segments["channel"].unique())
    profiles, r_shrunk, q_shrunk, quant_floor = fit_channel_profiles(synthetic_segments, channels)
    assert set(profiles.keys()) == set(channels)
    for ch in channels:
        assert r_shrunk[ch] > 0
        assert q_shrunk[ch] > 0


def test_detect_all_channels_runs_on_synthetic(synthetic_segments):
    channels = list(synthetic_segments["channel"].unique())
    _, r_shrunk, q_shrunk, quant_floor = fit_channel_profiles(synthetic_segments, channels)
    out = detect_all_channels(synthetic_segments, channels, r_shrunk, q_shrunk, quant_floor)
    assert "calibration" in out and "onset_results" in out and "onset_posteriors" in out
    assert len(out["calibration"]) == len(channels)
    assert set(out["onset_results"]["channel"].unique()) <= set(channels)


def test_bootstrap_channel_mcc_runs_on_synthetic(synthetic_segments):
    channels = list(synthetic_segments["channel"].unique())
    _, r_shrunk, q_shrunk, quant_floor = fit_channel_profiles(synthetic_segments, channels)
    out = detect_all_channels(synthetic_segments, channels, r_shrunk, q_shrunk, quant_floor)
    boot = bootstrap_channel_mcc(out["onset_results"], channels, n_boot=50)
    assert len(boot) == len(channels)
    assert set(["mcc_median", "mcc_ci_low", "mcc_ci_high"]).issubset(boot.columns)


def test_run_ablation_covers_all_four_combos():
    # ablation은 채널당 n_anom/n_norm>=5(MIN_SAMPLES_PER_CLASS) 채널만 스코어링 대상으로
    # 삼으므로, 모듈 fixture(20 세그먼트/채널)보다 표본이 넉넉한 합성 데이터를 별도로 사용한다.
    seg = make_synthetic_segments(n_segments_per_channel=40, seg_len=40)
    channels = list(seg["channel"].unique())
    result = run_ablation(seg, channels)
    assert len(result["ablation_grid"]) == len(COMBOS) == 4
    assert result["best_combo"] in COMBOS
