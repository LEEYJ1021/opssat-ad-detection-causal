"""
representations/tabular_summary_features.py
=============================================
Builds the tabular (human-engineered summary-statistic) dataset shared by all
10 tabular models: for every canonical-onset anomalous segment and every
placebo pivot drawn from a normal segment, computes the same three §3
quasi-experimental features:

    level  -> Cohen's |d| of the raw signal, pre- vs. post-pivot
    diff   -> log-variance-ratio of the first difference, pre- vs. post-pivot
    diff2  -> log-variance-ratio of the second difference, pre- vs. post-pivot

This module also owns the shared constants (FEATURES, RANDOM_STATE, etc.)
that the rest of the package imports, since the tabular feature definitions
are the analytical foundation every other representation and every model is
ultimately compared against.

Logic is unchanged from the original monolithic script (§1-§2 of
docs/dev-log/step_ai_model_cross_validation.md) -- only reorganized into a
reusable module with importable functions instead of top-level script code.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# Shared constants (imported by representations.raw_sequence_features,
# attribution.*, agreement_statistics, and run_all)
# =============================================================================
FEATURES = ["level", "diff", "diff2"]

RANDOM_STATE = 42

FINAL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]

BURN_IN = 5
MIN_WINDOW_POINTS = 5
K_PLACEBO_DRAWS_PER_NORMAL = 5
MAX_NORMAL_PER_CHANNEL = 300


# =============================================================================
# Core effect-size primitives (identical to the §3 quasi-experimental script;
# reused here rather than recomputed, per the README's "no recomputation"
# provenance rule)
# =============================================================================

def cohens_d_abs(a: np.ndarray, b: np.ndarray) -> float:
    """|Cohen's d| between two samples using pooled SD. NaN if underpowered
    (n < 2 in either sample) or degenerate (zero pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt((va + vb) / 2.0)
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float(abs(np.mean(a) - np.mean(b)) / pooled_sd)


def log_var_ratio(pre: np.ndarray, post: np.ndarray) -> float:
    """log(Var(post) / Var(pre)) -- the variance-surge proxy used for
    diff/diff2. NaN if either window has fewer than MIN_WINDOW_POINTS points
    or a non-finite/non-positive variance."""
    if len(pre) < MIN_WINDOW_POINTS or len(post) < MIN_WINDOW_POINTS:
        return np.nan
    vpre, vpost = np.var(pre, ddof=1), np.var(post, ddof=1)
    if vpre <= 0 or vpost <= 0 or not np.isfinite(vpre) or not np.isfinite(vpost):
        return np.nan
    return float(np.log(vpost / vpre))


def compute_effects(values: np.ndarray, pivot_idx: int) -> dict:
    """The three §3 features at a single pivot: level (|d| on raw values),
    diff and diff2 (log-variance-ratio on 1st/2nd differences). Offsets of
    1/2 samples are applied to diff/diff2 so their pre/post windows stay
    time-aligned with the level split at pivot_idx."""
    out = {"level": np.nan, "diff": np.nan, "diff2": np.nan}

    pre_level = values[BURN_IN:pivot_idx]
    post_level = values[pivot_idx:]
    out["level"] = cohens_d_abs(post_level, pre_level)

    diff_all = np.diff(values)
    split1 = max(pivot_idx - 1, 0)
    pre_diff = diff_all[max(BURN_IN - 1, 0):split1]
    post_diff = diff_all[split1:]
    out["diff"] = log_var_ratio(pre_diff, post_diff)

    diff2_all = np.diff(diff_all)
    split2 = max(pivot_idx - 2, 0)
    pre_diff2 = diff2_all[max(BURN_IN - 2, 0):split2]
    post_diff2 = diff2_all[split2:]
    out["diff2"] = log_var_ratio(pre_diff2, post_diff2)

    return out


# =============================================================================
# Dataset assembly
# =============================================================================

@dataclass
class TabularDatasetBundle:
    """Container returned by build_tabular_dataset().

    tab_df   -- DataFrame with columns [channel, segment, label, level, diff, diff2],
                NaN rows already dropped.
    groups   -- np.ndarray of "{channel}_{segment}" group ids, same length/order
                as tab_df (for GroupKFold so no segment leaks across folds).
    onset_ratio_pool_by_channel -- dict[channel] -> list of onset-position
                ratios observed in anomalous segments, used to draw placebo
                pivots on normal segments (mirrors the §3 normal-segment
                control design).
    """
    tab_df: pd.DataFrame
    groups: np.ndarray
    onset_ratio_pool_by_channel: dict = field(default_factory=dict)


def build_tabular_dataset(
    seg: pd.DataFrame,
    canonical_onset: pd.DataFrame,
    final_channels: Optional[list] = None,
    random_state: int = RANDOM_STATE,
) -> TabularDatasetBundle:
    """Construct the shared tabular label set: canonical-onset anomalous
    segments (label=1) vs. onset-ratio-resampled placebo pivots on normal
    segments (label=0), for the 5 scoped channels.

    Parameters
    ----------
    seg : DataFrame with columns [channel, segment, anomaly, timestamp, value],
          already filtered to the channels of interest.
    canonical_onset : DataFrame with columns [channel, segment, canonical_onset_idx]
          (median of the Layer-3 Monte Carlo onset draws per reliable segment).
    final_channels : channel scope; defaults to the 5-channel Layer-2 scope.
    random_state : seed controlling placebo-pivot sampling.

    Returns
    -------
    TabularDatasetBundle
    """
    channels = final_channels or FINAL_CHANNELS
    rng = np.random.default_rng(random_state)

    onset_ratio_pool_by_channel = {ch: [] for ch in channels}
    tab_rows, tab_group = [], []

    # 1) anomalous segments at their canonical onset -> label 1
    for _, row in canonical_onset.iterrows():
        ch, sid, onset_idx = row["channel"], int(row["segment"]), int(row["canonical_onset_idx"])
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 1)]
        if len(s) == 0:
            continue
        s = s.sort_values("timestamp")
        values = s["value"].values
        n = len(values)
        if onset_idx < BURN_IN + MIN_WINDOW_POINTS or onset_idx > n - MIN_WINDOW_POINTS:
            continue

        eff = compute_effects(values, onset_idx)
        tab_rows.append({"channel": ch, "segment": sid, "label": 1, **eff})
        tab_group.append(f"{ch}_{sid}")
        onset_ratio_pool_by_channel.setdefault(ch, []).append(onset_idx / n)

    # 2) normal segments -> K placebo pivots each, drawn from the anomalous
    #    onset-ratio distribution of the same channel -> label 0
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

                eff = compute_effects(values, pivot_idx)
                tab_rows.append({"channel": ch, "segment": sid, "label": 0, **eff})
                tab_group.append(f"{ch}_{sid}")

    raw_df = pd.DataFrame(tab_rows)
    valid_mask = raw_df[FEATURES].notna().all(axis=1).values if len(raw_df) else np.array([], dtype=bool)
    tab_df = raw_df.loc[valid_mask].reset_index(drop=True)
    groups = np.array(tab_group)[valid_mask]

    return TabularDatasetBundle(
        tab_df=tab_df, groups=groups, onset_ratio_pool_by_channel=onset_ratio_pool_by_channel
    )


def save_tabular_dataset(bundle: TabularDatasetBundle, out_path: Path) -> Path:
    """Persist tab_df to modelclass_dataset_tabular.csv (see results/README.md)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.tab_df.to_csv(out_path, index=False)
    return out_path
