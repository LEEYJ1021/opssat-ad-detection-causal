# Data

This directory does not contain any of the three raw datasets used in this
repository (license terms do not permit redistribution of any of them).

## Quick start

```bash
bash data/download_opssat_ad.sh    # required — primary analysis (Layers 1–3)
bash data/download_smap_msl.sh     # required for § External Validation
bash data/download_smd.sh          # required for § External Validation
```

Each script populates `data/raw/<dataset>/` in the exact layout the
corresponding pipeline code expects (see below). None of the scripts modify
files outside `data/raw/`, and none require credentials — all three datasets
are fetched from open, unauthenticated public releases.

You do not need all three to run the primary OPS-SAT-AD analysis
(`layer1_signal_estimation`, `layer2_anomaly_detection`,
`layer3_signature_analysis`, `model_cross_validation`). SMAP/MSL and SMD are
needed only for `external_validation` (§ External Validation in the main
README).

---

## Dataset 1 — OPS-SAT-AD (primary)

**Source.** European Space Agency OPS-SAT mission telemetry
anomaly-detection benchmark (Ruszczak, Kotowski, Evans & Nalepa, *Scientific
Data* 12:710, 2025), distributed via Zenodo.

**Script.** `download_opssat_ad.sh`

**Expected layout after download:**
```
data/raw/opssat_ad/
├── dataset.csv              # segment-level metadata (channel, anomaly flag, train/test split)
└── segments/                # per-segment telemetry .csv files
```

**Size.** ~9 channels, 2,123 segments, 303,493 raw observations (small; downloads in seconds).

**Citation requirement.** Please cite the OPS-SAT-AD dataset release
directly, in addition to this repository's paper:

> Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2025). OPS-SAT-AD — An
> anomaly detection dataset for satellite telemetry. *Scientific Data*,
> 12, 710.

**License.** Governed by its Zenodo record's stated license; consult the
Zenodo page directly before any use beyond research reproduction. This
repository does not alter or relicense the dataset in any way.

---

## Dataset 2 — SMAP/MSL (external validation)

**Source.** NASA Soil Moisture Active Passive (SMAP) and Mars Science
Laboratory (MSL) spacecraft telemetry, released alongside the Telemanom
benchmark (Hundman, Constantinou, Laporte, Colwell & Soderstrom, KDD 2018).

**Script.** `download_smap_msl.sh`

**Expected layout after download:**
```
data/raw/smap_msl/
├── labeled_anomalies.csv    # chan_id, spacecraft, anomaly_sequences (ground-truth onset/offset), ...
├── train/
│   └── <chan_id>.npy        # nominal-only telemetry, multi-column (col 0 = signal, rest = command context)
└── test/
    └── <chan_id>.npy        # full telemetry including anomaly windows, same column layout
```

**Important preprocessing note (see main README, § External Validation,
Stage 0).** Each `.npy` file's columns beyond the first are one-hot-encoded
command/telemetry context, not additional sensor channels. All pipeline code
in `external_validation/loaders.py` extracts only column 0; if you write your
own code against this data, do the same, or your `diff`/`diff²` statistics
will be computed over a mix of signal and command bits.

**Size.** 82 channels (55 SMAP + 27 MSL); the packaged archive is a few
hundred MB. `download_smap_msl.sh` reports progress during download.

**Citation requirement.** Please cite the Telemanom paper directly, in
addition to this repository's paper:

> Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T.
> (2018). Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
> Dynamic Thresholding. *Proceedings of the 24th ACM SIGKDD International
> Conference on Knowledge Discovery & Data Mining* (KDD '18), 387–395.

**License.** Governed by the terms of the upstream public release (see the
source repository linked in `download_smap_msl.sh` for the current license
file). This repository does not alter or relicense the dataset.

---

## Dataset 3 — SMD / Server Machine Dataset (external validation)

**Source.** Industrial server telemetry released alongside the OmniAnomaly
benchmark (Su, Zhao, Niu, Liu, Sun & Pei, KDD 2019).

**Script.** `download_smd.sh`

**Expected layout after download:**
```
data/raw/smd/
├── train/
│   └── machine-<g>-<i>.txt              # comma-separated, nominal telemetry, 38 columns
├── test/
│   └── machine-<g>-<i>.txt              # comma-separated, full telemetry incl. anomalies
├── test_label/
│   └── machine-<g>-<i>.txt              # single-column, timestep-level 0/1 ground-truth
└── interpretation_label/
    └── machine-<g>-<i>.txt              # anomaly-window → contributing-dimension annotations
```

**Size.** 28 machines × 38 dimensions = 1,064 channels; the packaged archive
is on the order of tens of MB (plain-text, no compression of numeric data).

**Citation requirement.** Please cite the OmniAnomaly paper directly, in
addition to this repository's paper:

> Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., Pei, D. (2019). Robust Anomaly
> Detection for Multivariate Time Series through Stochastic Recurrent Neural
> Network. *Proceedings of the 25th ACM SIGKDD International Conference on
> Knowledge Discovery & Data Mining* (KDD '19), 2828–2837.

**License.** Governed by the terms of the upstream public release (see the
source repository linked in `download_smd.sh` for the current license file).
This repository does not alter or relicense the dataset.

---

## Verifying a download

All three scripts print the number of files/channels found after extraction
and exit non-zero if the expected top-level structure is missing, so a
failed or partial download will not silently pass through to the analysis
code. If a script reports a checksum or file-count mismatch, do not proceed
with the analysis — re-run the script, and if the problem persists, check
whether the upstream source has changed its release layout (all three are
external repositories outside this project's control) and open an issue.

## A note on reproducibility across all three datasets

None of the three datasets are redistributed in this repository, in
`results/`, or anywhere else — only derived, aggregated statistics (e.g.
`results/external_validation/*.csv`) are committed. Re-running
`data/download_*.sh` followed by the corresponding pipeline entry points
(see the main README's "Reproducing every number in this README" section)
regenerates every number and figure from the original public sources.
