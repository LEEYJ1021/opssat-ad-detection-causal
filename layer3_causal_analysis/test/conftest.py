"""
layer3_causal_analysis/test/conftest.py
=============================================================================
pytest 공용 설정 + 합성(synthetic) 데이터 픽스처.

이 프로젝트의 모듈(_shared.py, onset_mc_propagation.py, ...)은 패키지가
아니라 flat import(`from _shared import ...`)로 서로를 참조하므로, 테스트가
같은 디렉터리를 sys.path에 넣어줘야 import가 된다. 또한 _shared.py는 import
시점에 `OUT_DIR.mkdir(parents=True, exist_ok=True)`를 실행하므로, 테스트에서
실제 results/layer3 디렉터리를 만들지 않도록 각 테스트는 필요할 때
monkeypatch로 OUT_DIR류 경로 상수를 tmp_path로 갈아끼운다(README 참고).

여기 있는 픽스처는 전부 "합성 데이터"이며, 실제 OPS-SAT-AD 관측값이 아니다.
확정된 상수(BURN_IN=5, MIN_WINDOW_POINTS=5, FINAL_CHANNELS 5개 등)는
_shared.py의 값을 그대로 재사용해 실제 파이프라인과 조건을 맞춘다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 이 디렉터리의 부모(=layer3_causal_analysis/)를 sys.path에 추가해
# `from _shared import ...` 같은 flat import가 동작하게 한다.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

RNG_SEED = 12345


# =============================================================================
# 합성 신호 생성 헬퍼
# =============================================================================

def make_step_series(n: int, onset_idx: int, pre_mean: float = 0.0,
                      post_mean: float = 5.0, noise_std: float = 0.3,
                      seed: int = RNG_SEED) -> np.ndarray:
    """level이 onset_idx에서 pre_mean -> post_mean으로 계단 이동하는 합성 신호.
    diff/diff2에는 (이론상) 큰 변화를 만들지 않는다 — level 전용 테스트에 사용."""
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    values[:onset_idx] = pre_mean + rng.normal(0, noise_std, onset_idx)
    values[onset_idx:] = post_mean + rng.normal(0, noise_std, n - onset_idx)
    return values


def make_variance_surge_series(n: int, onset_idx: int, pre_std: float = 0.1,
                                post_std: float = 3.0, seed: int = RNG_SEED) -> np.ndarray:
    """level 평균은 그대로 두고 onset 이후 '변화량(diff)'의 분산만 급증시키는
    합성 신호 — diff/diff2 전용 시그니처(§3 핵심 주장)를 흉내낸다. cumsum으로
    diff의 표준편차를 직접 통제한다."""
    rng = np.random.default_rng(seed)
    pre_steps = rng.normal(0, pre_std, onset_idx)
    post_steps = rng.normal(0, post_std, n - onset_idx)
    steps = np.concatenate([[0.0], pre_steps, post_steps])[:n]
    # 누적합을 취하되 시작점 근처 burn-in 구간은 스텝을 0으로 둬서 완전히
    # 평평하게(std=0) 만들지 않도록 아주 작은 노이즈를 더한다.
    values = np.cumsum(steps) + rng.normal(0, 1e-6, n)
    return values


def make_flat_series(n: int, mean: float = 0.0, std: float = 0.2, seed: int = RNG_SEED) -> np.ndarray:
    """아무 변화도 없는 순수 정상(normal) 세그먼트 합성 신호."""
    rng = np.random.default_rng(seed)
    return mean + rng.normal(0, std, n)


# =============================================================================
# segments.csv 스키마를 흉내낸 합성 DataFrame 픽스처
# =============================================================================

@pytest.fixture
def final_channels():
    from _shared import FINAL_CHANNELS
    return list(FINAL_CHANNELS)


@pytest.fixture
def synthetic_segments_df(final_channels) -> pd.DataFrame:
    """5개 확정 채널 x (이상 세그먼트 여러 개 + 정상 세그먼트 여러 개)를 담은
    합성 segments.csv 형태의 DataFrame. 컬럼: channel, segment, timestamp,
    value, sampling, anomaly, train.

    각 이상 세그먼트는 diff/diff2 급증(§3 핵심 시그니처)을 갖도록 만들고,
    각 정상 세그먼트는 평탄한 신호로 만든다 — 실제 데이터의 통계 성질을
    재현하려는 목적이 아니라, 파이프라인 각 단계가 도달 가능한(NaN으로
    전부 스킵되지 않는) 최소 규모 데이터를 제공하는 것이 목적이다."""
    sampling = 1.0
    n_points = 200
    onset_idx = 90
    rows = []
    seg_id = 0
    for ch_i, ch in enumerate(final_channels):
        # 이상 세그먼트 6개
        for k in range(6):
            values = make_variance_surge_series(
                n_points, onset_idx, seed=RNG_SEED + ch_i * 100 + k
            )
            ts = pd.date_range("2024-01-01", periods=n_points, freq="s")
            train_flag = 1 if k < 4 else 0  # 4 train / 2 test
            for t, v in zip(ts, values):
                rows.append({"channel": ch, "segment": seg_id, "timestamp": t,
                             "value": v, "sampling": sampling, "anomaly": 1,
                             "train": train_flag})
            seg_id += 1
        # 정상 세그먼트 10개
        for k in range(10):
            values = make_flat_series(n_points, seed=RNG_SEED + 1000 + ch_i * 100 + k)
            ts = pd.date_range("2024-01-01", periods=n_points, freq="s")
            train_flag = 1 if k < 7 else 0
            for t, v in zip(ts, values):
                rows.append({"channel": ch, "segment": seg_id, "timestamp": t,
                             "value": v, "sampling": sampling, "anomaly": 0,
                             "train": train_flag})
            seg_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_segments_csv(tmp_path, synthetic_segments_df) -> Path:
    """synthetic_segments_df를 실제 CSV 파일로 저장한 경로. SEGMENTS_PATH를
    monkeypatch할 때 사용."""
    path = tmp_path / "segments.csv"
    synthetic_segments_df.to_csv(path, index=False)
    return path
