"""
representations/raw_sequence_features.py
===========================================
Builds the raw-sequence (no human-engineered summary statistic) dataset
shared by the 6 sequence models (CNN1D, TCN, BiLSTM, BiGRU, TinyTransformer,
LightMamba): for every pivot (canonical onset or placebo), a 3-channel
window of length 2*SEQ_WINDOW_HALF stacking [level, diff, diff2] as raw,
per-window z-normalized time series.

This is the representation used to test whether the tabular result is an
artifact of the engineered-feature design itself (README §"Two independent
data representations"): if models that never see level_d_abs / diff_logvar /
diff2_logvar still rank diff/diff2 over level, that concern is ruled out.

Logic is unchanged from the original monolithic script -- only reorganized.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from model_cross_validation.representations.tabular_summary_features import (
    FEATURES, RANDOM_STATE, FINAL_CHANNELS, BURN_IN, MIN_WINDOW_POINTS,
    MAX_NORMAL_PER_CHANNEL, K_PLACEBO_DRAWS_PER_NORMAL,
)

SEQ_WINDOW_HALF = 15  # half-width (ticks) of the pivot-centered window
SEQ_MIN_SEGMENT_LEN = 2 * SEQ_WINDOW_HALF + BURN_IN


def znorm(x: np.ndarray) -> np.ndarray:
    """Per-window z-normalization: each of the 3 channels is normalized using
    its own within-window mean/SD, so the model compares *relative* shape
    (level vs. diff vs. diff2) rather than absolute channel-to-channel scale
    -- directly matching the question this analysis asks."""
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / (sd + 1e-8)


def build_3channel_window(values: np.ndarray, pivot_idx: int, half: int = SEQ_WINDOW_HALF):
    """Return a (3, 2*half) float32 array stacking z-normalized
    [level, diff, diff2] windows centered on pivot_idx, or None if the
    segment is too short to extract a full window (including the extra
    2-sample margin diff2 needs)."""
    lo, hi = pivot_idx - half, pivot_idx + half
    if lo - 2 < 0 or hi > len(values):
        return None

    level_w = values[lo:hi]
    diff_all = np.diff(values)
    diff_w = diff_all[lo - 1: hi - 1]
    diff2_all = np.diff(diff_all)
    diff2_w = diff2_all[lo - 2: hi - 2]

    if len(level_w) != 2 * half or len(diff_w) != 2 * half or len(diff2_w) != 2 * half:
        return None

    return np.stack([znorm(level_w), znorm(diff_w), znorm(diff2_w)], axis=0).astype(np.float32)


@dataclass
class SequenceDatasetBundle:
    """rows: list of {"channel", "segment", "label", "window"} dicts.
    groups: np.ndarray of "{channel}_{segment}" group ids, same order as rows.
    """
    rows: list
    groups: np.ndarray


def build_sequence_dataset(
    seg: pd.DataFrame,
    canonical_onset: pd.DataFrame,
    onset_ratio_pool_by_channel: dict,
    final_channels: Optional[list] = None,
    random_state: int = RANDOM_STATE,
    half: int = SEQ_WINDOW_HALF,
) -> SequenceDatasetBundle:
    """Construct the raw-sequence label set using the *same* pivots
    (canonical onsets + placebo draws) as the tabular representation, so the
    two representations are directly comparable model-for-model.

    onset_ratio_pool_by_channel should be the dict returned alongside the
    tabular dataset (representations.tabular_summary_features.build_tabular_dataset),
    ensuring both representations draw placebo pivots from the identical
    onset-ratio distribution.
    """
    channels = final_channels or FINAL_CHANNELS
    rng = np.random.default_rng(random_state)

    rows, groups = [], []

    for _, row in canonical_onset.iterrows():
        ch, sid, onset_idx = row["channel"], int(row["segment"]), int(row["canonical_onset_idx"])
        if ch not in channels:
            continue
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 1)]
        if len(s) == 0:
            continue
        s = s.sort_values("timestamp")
        values = s["value"].values
        n = len(values)
        if onset_idx < BURN_IN + MIN_WINDOW_POINTS or onset_idx > n - MIN_WINDOW_POINTS:
            continue

        win = build_3channel_window(values, onset_idx, half)
        if win is not None:
            rows.append({"channel": ch, "segment": sid, "label": 1, "window": win})
            groups.append(f"{ch}_{sid}")

    for ch in channels:
        ratio_pool = np.array(onset_ratio_pool_by_channel.get(ch, []))
        if len(ratio_pool) == 0:
            continue

        normal_meta = (
            seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)]
            .groupby("segment").size().rename("n_points").reset_index()
        )
        if len(normal_meta) > MAX_NORMAL_PER_CHANNEL:
            normal_meta = normal_meta.sample(n=MAX_NORMAL_PER_CHANNEL, random_state=random_state)

        for _, nrow in normal_meta.iterrows():
            sid, n = int(nrow["segment"]), int(nrow["n_points"])
            s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 0)].sort_values("timestamp")
            values = s["value"].values

            for _ in range(K_PLACEBO_DRAWS_PER_NORMAL):
                ratio = rng.choice(ratio_pool)
                pivot_idx = int(round(ratio * n))
                pivot_idx = int(np.clip(
                    pivot_idx,
                    BURN_IN + MIN_WINDOW_POINTS,
                    max(n - MIN_WINDOW_POINTS, BURN_IN + MIN_WINDOW_POINTS),
                ))
                if pivot_idx < BURN_IN + MIN_WINDOW_POINTS or pivot_idx > n - MIN_WINDOW_POINTS:
                    continue

                win = build_3channel_window(values, pivot_idx, half)
                if win is not None:
                    rows.append({"channel": ch, "segment": sid, "label": 0, "window": win})
                    groups.append(f"{ch}_{sid}")

    return SequenceDatasetBundle(rows=rows, groups=np.array(groups))
