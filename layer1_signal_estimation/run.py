"""
run.py
======
Entry point for Layer 1 (signal estimation).

    python -m layer1_signal_estimation.run
    python -m layer1_signal_estimation.run --estimator mixture   # ablation
    python -m layer1_signal_estimation.run --synthetic            # no data/raw/ needed

Pipeline (see README, "Layer 1 — Signal Estimation" and this package's
``__init__.py`` docstring):

    1. Load `data/raw/dataset.csv` + `data/raw/segments.csv` (or generate
       synthetic stand-in data with ``--synthetic``, so the pipeline is
       runnable and testable without first executing
       `data/download_opssat_ad.sh`).
    2. channel_classification.py  -> per-channel regime tag
    3. noise_estimation.py        -> per-channel (r, q), locked to the
                                       "simple" var(diff)/2 estimator
                                       (see that module's docstring)
    4. hierarchical_shrinkage.py  -> pool (r, q) across channels
    5. kalman_filter.py           -> denoised state estimate per segment

Outputs (consumed by Layer 2):
    results/layer1/channel_classification.json
    results/layer1/noise_estimates.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import ALL_CHANNELS, DEFAULT_DATASET_CSV, DEFAULT_RESULTS_DIR, DEFAULT_SEGMENTS_CSV, RANDOM_STATE
from .channel_classification import classify_channel, profiles_to_records
from .hierarchical_shrinkage import hierarchical_shrink_variance
from .kalman_filter import run_kalman_for_segment
from .noise_estimation import LOCKED_ESTIMATOR, EstimatorName, estimate_channel_noise


def _load_real_data(dataset_csv: Path, segments_csv: Path) -> pd.DataFrame:
    """Load and join OPS-SAT-AD `dataset.csv` (per-segment metadata,
    including the `anomaly` label) with `segments.csv` (raw per-sample
    values), on the shared `segment` key.

    Returns a long-format DataFrame with columns
    ``[segment, channel, anomaly, timestamp, value]``.
    """
    if not dataset_csv.exists() or not segments_csv.exists():
        raise FileNotFoundError(
            f"Expected OPS-SAT-AD data at {dataset_csv} and {segments_csv}.\n"
            "Run `bash data/download_opssat_ad.sh` first, or pass --synthetic "
            "to run this pipeline on generated stand-in data."
        )
    labels = pd.read_csv(dataset_csv, usecols=["segment", "anomaly"])
    seg = pd.read_csv(segments_csv, usecols=["segment", "channel", "timestamp", "value"])
    seg = seg.merge(labels, on="segment", how="left", validate="many_to_one")
    return seg


def _generate_synthetic_data(
    n_segments_per_channel: int = 40, seg_len: int = 60, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Generate synthetic stand-in telemetry with the same long-format
    schema as `_load_real_data`, so the Layer-1 pipeline can be exercised
    end-to-end (and its tests run in CI) without the real, non-redistributed
    OPS-SAT-AD dataset.

    This deliberately reproduces the two dominant OPS-SAT-AD noise
    regimes -- a quantized channel and a float-noise-suspect channel --
    each with a small fraction of anomalous segments carrying an injected
    variance surge, matching this repository's headline finding (variance
    surge, not level shift, is the anomaly signature) so downstream
    smoke tests exercise a realistic case.
    """
    rng = np.random.default_rng(seed)
    specs = {
        "SYN_QUANT": dict(step=0.0026, base=1.0, anomaly_jump=0.05),
        "SYN_FLOAT": dict(step=5e-11, base=1e-7, anomaly_jump=2e-8),
    }
    rows: List[dict] = []
    seg_id = 1
    for channel, spec in specs.items():
        n_anom = max(1, n_segments_per_channel // 5)
        anomaly_flags = np.array([1] * n_anom + [0] * (n_segments_per_channel - n_anom))
        rng.shuffle(anomaly_flags)
        for is_anom in anomaly_flags:
            base = spec["base"]
            step = spec["step"]
            values = base + np.cumsum(rng.normal(0.0, step, size=seg_len))
            if is_anom:
                onset = seg_len // 2
                values[onset:] += np.cumsum(
                    rng.normal(0.0, spec["anomaly_jump"], size=seg_len - onset)
                )
            if channel == "SYN_QUANT":
                # Round to the ADC step grid so a large fraction of first
                # differences come out exactly zero, reproducing the
                # quantized-channel regime (see channel_classification.py).
                values = np.round(values / step) * step
            for t, v in enumerate(values):
                rows.append(
                    {
                        "segment": seg_id,
                        "channel": channel,
                        "anomaly": int(is_anom),
                        "timestamp": t,
                        "value": float(v),
                    }
                )
            seg_id += 1
    return pd.DataFrame(rows)


def _nominal_values_by_channel(seg_df: pd.DataFrame) -> Dict[str, List[np.ndarray]]:
    """Group nominal (anomaly == 0) segment value arrays by channel, in
    segment order."""
    out: Dict[str, List[np.ndarray]] = {}
    nominal = seg_df[seg_df["anomaly"] == 0]
    for channel, chan_df in nominal.groupby("channel", sort=False):
        segments = [
            g.sort_values("timestamp")["value"].to_numpy(dtype=float)
            for _, g in chan_df.groupby("segment", sort=False)
        ]
        out[channel] = segments
    return out


def run_layer1(
    dataset_csv: Path = DEFAULT_DATASET_CSV,
    segments_csv: Path = DEFAULT_SEGMENTS_CSV,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    estimator: EstimatorName = LOCKED_ESTIMATOR,
    use_synthetic: bool = False,
    apply_shrinkage: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Run the full Layer-1 pipeline and write its two result artifacts.

    Returns
    -------
    (noise_df, classification_json) : the two artifacts, already written
    to ``results_dir``, also returned in-memory for convenience (e.g. for
    Layer 2 to consume directly within the same process, or for tests).
    """
    seg_df = _generate_synthetic_data() if use_synthetic else _load_real_data(dataset_csv, segments_csv)

    nominal_by_channel = _nominal_values_by_channel(seg_df)
    if not nominal_by_channel:
        raise ValueError("No nominal (anomaly == 0) segments found -- cannot estimate noise.")

    # 1. Channel classification.
    profiles = {
        channel: classify_channel(channel, segments)
        for channel, segments in nominal_by_channel.items()
    }

    # 2. Per-channel noise estimation (locked estimator by default).
    noise_estimates = {
        channel: estimate_channel_noise(channel, segments, estimator=estimator)
        for channel, segments in nominal_by_channel.items()
    }

    # 3. Hierarchical shrinkage of (r, q) across channels.
    if apply_shrinkage and len(noise_estimates) >= 2:
        r_input = {c: (e.r, e.n_diff) for c, e in noise_estimates.items()}
        q_input = {c: (e.q, e.n_diff) for c, e in noise_estimates.items()}
        r_shrunk = hierarchical_shrink_variance(r_input)
        q_shrunk = hierarchical_shrink_variance(q_input)
    else:
        r_shrunk = {c: e.r for c, e in noise_estimates.items()}
        q_shrunk = {c: e.q for c, e in noise_estimates.items()}

    noise_records = []
    for channel, est in noise_estimates.items():
        noise_records.append(
            {
                "channel": channel,
                "estimator": est.estimator,
                "r_raw": est.r,
                "q_raw": est.q,
                "r_shrunk": r_shrunk[channel],
                "q_shrunk": q_shrunk[channel],
                "n_diff": est.n_diff,
                "channel_type": profiles[channel].channel_type,
            }
        )
    noise_df = pd.DataFrame.from_records(noise_records).sort_values("channel").reset_index(drop=True)

    classification_records = profiles_to_records(profiles[c] for c in profiles)

    # 4. Per-segment Kalman state estimation (denoising) using the
    #    shrunk noise estimates -- this is Layer 1's actual output signal
    #    for Layer 2, computed here as a smoke-test / illustrative pass
    #    over every nominal segment (kept in-memory; not written to CSV,
    #    since Layer 2 re-runs the filter itself with the locked
    #    hyperparameters from `results/layer1/noise_estimates.csv`).
    for channel, segments in nominal_by_channel.items():
        profile = profiles[channel]
        q_step = profile.quantization_step if profile.channel_type == "quantized" else None
        r = r_shrunk[channel]
        q = q_shrunk[channel]
        for values in segments:
            if len(values) == 0:
                continue
            run_kalman_for_segment(values, r=r, q=q, quantization_step=q_step)

    # --- write artifacts ---
    results_dir.mkdir(parents=True, exist_ok=True)
    noise_csv_path = results_dir / "noise_estimates.csv"
    classification_json_path = results_dir / "channel_classification.json"

    noise_df.to_csv(noise_csv_path, index=False)
    with open(classification_json_path, "w", encoding="utf-8") as f:
        json.dump(classification_records, f, indent=2, ensure_ascii=False)

    print(f"[layer1] wrote {noise_csv_path} ({len(noise_df)} channels)")
    print(f"[layer1] wrote {classification_json_path} ({len(classification_records)} channels)")

    return noise_df, classification_records


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer 1 (signal estimation).")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--estimator", choices=["simple", "mixture"], default=LOCKED_ESTIMATOR,
        help="Noise-variance estimator. 'simple' (var(diff)/2) is the locked pipeline "
             "default; 'mixture' reproduces the README's ablation comparison.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic stand-in data instead of reading data/raw/ "
             "(useful when the real, non-redistributed OPS-SAT-AD data has not "
             "been downloaded yet).",
    )
    parser.add_argument(
        "--no-shrinkage", action="store_true",
        help="Skip hierarchical shrinkage and use each channel's raw noise estimate.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_layer1(
            dataset_csv=args.dataset_csv,
            segments_csv=args.segments_csv,
            results_dir=args.results_dir,
            estimator=args.estimator,
            use_synthetic=args.synthetic,
            apply_shrinkage=not args.no_shrinkage,
        )
    except FileNotFoundError as e:
        print(f"[layer1] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
