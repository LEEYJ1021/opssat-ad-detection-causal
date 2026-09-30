# opssat-ad-onset-signatures

**Which Statistic Should Trigger the Alarm? A Dependence-Aware Evaluation of Onset Features for Spacecraft and Industrial Telemetry Monitoring (OPS-SAT-AD, SMAP/MSL, SMD)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-fully%20reproducible-brightgreen.svg)](#9-reproducing-every-number)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](#13-citation-and-license)

This repository is the code, data-provenance and figure-reproduction companion to **Paper 1** of a two-paper series on alarm design for satellite telemetry. It is self-contained and does not depend on any other repository.

| This series | Repository | Role |
|---|---|---|
| **Paper 1 (this repo)**: onset-feature evaluation | `opssat-ad-onset-signatures` | independent, no upstream dependencies |
| Paper 2: risk-aware alarm design (CVaR / conformal) | [`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization) | consumes `results/` artifacts of this repo |
| DSS prototype (not a paper) | [`opssat-ad-dss`](https://github.com/USERNAME/opssat-ad-dss) | consumes artifacts of both papers |

---

## Highlights

- Onset features for alarms are evaluated under nested labels, constructed placebos and small K.
- Derivative features exceed level in 3/3 benchmarks (ΔAUROC +0.042 to +0.462).
- A detector-matched placebo lowers the OPS-SAT-AD `diff` AUROC from 0.900 to 0.569.
- A fused alarm adds +1.2 pp coverage at 5% FAR in SMD (95% CI 0.4 to 2.0).
- Tie handling reversed a precedence contrast; a reporting checklist is provided.

## Abstract

Alarm thresholds for spacecraft and industrial telemetry are conventionally set on signal *level*, on the premise that an anomaly begins as a mean shift. Evidence for that premise is usually reported as a single p-value or AUROC computed as if labeled onsets were independent, although onsets are nested in segments, channels and machines and the "normal" reference is itself constructed by a procedure. This paper asks which feature family carries onset information and how much of an apparent advantage survives dependence-aware evaluation. We compare `level` with onset-relative change features (`diff`, `diff²`) on labeled anomaly onsets in three benchmarks: OPS-SAT-AD (5 scoped channels, 168 complete-case onsets), SMAP/MSL (82 channels, 88 windows) and SMD (28 machines, 327 events), each against same-channel placebo pivots. Every headline statistic is an onset-versus-placebo AUROC with a cluster-bootstrap interval at the unit that repeats, and each claim is passed through a fixed audit sequence: an independent pipeline replica, small-K resampling, FAR matching over six targets, a detector-matched placebo, a tie-handling audit and synthetic existence proofs. `diff`/`diff²` exceed `level` in all three datasets (Δ AUROC +0.462 [0.409, 0.508], +0.122 [0.045, 0.196], +0.042 [0.016, 0.066]); SMAP/MSL and SMD use ground-truth onsets, so the direction does not depend on an onset estimator. `level` adds no out-of-fold discrimination once derivatives are present (ΔAUROC −0.0007, −0.0068, −0.0009). At a matched 5% false-alarm rate in SMD a `level ∨ diff ∨ diff²` rule raises coverage by +1.2 pp [0.4, 2.0]. The audit also quantifies how evaluation choices inflate apparent advantages: placebo selection (`diff` 0.900 → 0.569), resampling unit (interval width ×2.7–3.1), statistic definition (external contrasts +0.122 → +0.060 and +0.042 → +0.010) and tie handling (precedence baselines 44.2/56.5/72.7% → 36.9/34.8/32.1%). The results support adding a derivative-variance channel to alarm feature sets with a modest, quantified gain, and provide a reporting protocol for feature-discrimination claims on nested telemetry labels.

**Keywords:** alarm design; anomaly onset; spacecraft telemetry; cluster bootstrap; false-alarm rate; small-K inference; placebo control; OPS-SAT.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and related work](#2-background-and-related-work)
3. [Data and scope](#3-data-and-scope)
4. [Methods](#4-methods)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Conclusion](#7-conclusion)
8. [Repository structure](#8-repository-structure)
9. [Reproducing every number](#9-reproducing-every-number)
10. [Artifact index](#10-artifact-index)
11. [Data and code availability](#11-data-and-code-availability)
12. [References](#12-references)
13. [Citation and license](#13-citation-and-license)

---

## 1. Introduction

### 1.1 Motivation

Monitoring and alarm logic for satellites and industrial systems typically treats a level shift as the default anomaly signature: a threshold is placed on a standardized signal value, and an alarm is raised when the value departs from its nominal range. In reliability and system-safety practice the feature on which that threshold sits determines which failures are detected early, how many false alarms operators absorb, and how much confidence can be placed in offline validation. A claim that one feature family "carries the onset information" therefore translates directly into an alarm design.

### 1.2 Gap

Feature-discrimination claims on telemetry benchmarks are exposed to three properties that are rarely addressed jointly.

1. **Labels repeat inside units.** Anomalous segments cluster in few channels (OPS-SAT-AD: five usable channels), machines (SMD: 28) or missions. An interval that treats segments as independent can be near-nominal for one estimand and cover another at 79% (Fig. 6b).
2. **The reference is constructed.** The "normal" pivots against which onsets are compared are position-matched resamples or detector outputs. Their construction changed the OPS-SAT-AD `diff` AUROC from 0.900 to 0.569 (Fig. 3c).
3. **Threshold-crossing criteria hide their operating point and their tie convention.** A fixed |z| > 3 rule has a placebo false-alarm rate of 27% to 46% in the external datasets, and integer-sample ties reversed the sign of one contrast (Fig. 7).

### 1.3 Approach

We evaluate `level`, `diff` and `diff²` on labeled onsets in three benchmarks that differ in platform, sensor physics, labeling protocol and onset provenance. The OPS-SAT-AD analysis uses detector-estimated onsets; SMAP/MSL and SMD use ground-truth onsets, which supplies an estimator-free arm. All statistics are onset-versus-placebo AUROCs with cluster-bootstrap intervals at the unit that repeats, complemented by two-level, wild-cluster and exact sign-flip resampling for the small-K case, a detector-matched placebo, FAR-matched coverage and a tie-aware precedence analysis.

### 1.4 Contributions

1. **A cross-dataset directional result with an estimator-free arm.** `diff`/`diff²` separate onsets from placebo more strongly than `level` in 3/3 datasets, and `level` adds no out-of-fold discrimination once they are present (Sections 5.2, 5.3).
2. **A quantified operating value for alarm design.** A fused `level ∨ diff ∨ diff²` rule with a joint-FAR-calibrated threshold adds +1.2 pp coverage at 5% FAR in SMD, positive at all six FAR targets (Section 5.7).
3. **An inflation ledger and reporting protocol.** Eight evaluation choices are quantified by how much they change an apparent advantage, with an estimand-dependent resampling result, an exact pooling-bias identity and a tie-handling audit (Sections 5.5, 5.8, 5.9, 6.2).
4. **A detector-matched placebo for OPS-SAT-AD** that separates the selection-conditioned part of the effect from the residual (Section 5.4).
5. **A reproducible package**: an independent from-scratch replica (4,996/4,996 rows, maximum relative error 1.8×10⁻¹⁵), a de-duplicated placebo pool, multi-seed grouped cross-validation and a row-level leakage positive control (Section 5.10).

### 1.5 Relevance to reliability and system safety

The results bear on three decisions in reliability practice: which features to place in an alarm rule (Section 6.3, rules R1–R3), how to validate the rule offline without overstating its gain (rules R4–R7), and how to report uncertainty when only a handful of anomaly-bearing channels exist (rule R5). Small-K nested designs are the typical situation in spacecraft anomaly databases, so the protocol addresses the common case.

---

## 2. Background and related work

**Spacecraft telemetry anomaly detection.** Benchmarks such as SMAP/MSL (Hundman et al., 2018), SMD (Su et al., 2019) and OPS-SAT-AD (Ruszczak et al., 2025) are used predominantly to compare detectors on point or window metrics. Wu and Keogh (2021) documented triviality and label-definition problems in common time-series benchmarks; we apply a triviality filter motivated by that work (Section 3.1). This paper differs in target: it evaluates *which feature carries onset information*, not which detector scores highest.

**Online change-point detection.** Bayesian online change-point detection (Adams and MacKay, 2007) supplies the onset front end for OPS-SAT-AD, applied to Kalman-filter innovations so that persistent mean shifts are absorbed into the state and short-lived transients remain.

**Clustered inference and small-K.** Cluster-bootstrap, wild-cluster (Webb, 2014; Cameron, Gelbach and Miller, 2008) and exact sign-flip procedures address dependence among repeated units. With K = 5 no exact test can reject at 0.05 (the smallest attainable two-sided p is 2/2⁵ = 0.0625), so intervals are read as direction and magnitude.

**Multiple testing and heterogeneity.** External datasets use Benjamini–Hochberg FDR (1995) with separate confirmatory and exploratory families. Between-channel heterogeneity is summarized with DerSimonian–Laird random effects and Higgins–Thompson I² conventions.

---

## 3. Data and scope

### 3.1 Datasets

**Table 1. Datasets, onset provenance and cluster units.**

| Dataset | Content | Onset source | Cluster unit (K) |
|---|---|---|---|
| **OPS-SAT-AD** (primary) | ESA OPS-SAT nanosatellite; 9 channels (3 magnetometer, 6 photodiode); 2,123 expert-labeled univariate segments; 5 channels in scope hold 386 anomalous segments | BOCPD on Kalman-filter innovations; canonical onset = median of 200 Monte Carlo draws | segment (1,163) and channel (5) |
| **SMAP/MSL** | NASA telemetry; 82 channels (55 SMAP, 27 MSL); 105 labeled windows, 88 after triviality filtering; window length median 120, mean 616, max 4,217 samples | Ground-truth onset indices (`labeled_anomalies.csv`) | channel (81) |
| **SMD** | Industrial server telemetry; 28 machines × 38 dimensions = 1,064 channels; 327 events; 11,493 window-level pairs (events replicated across dimensions) | Ground-truth timestep-level 0/1 labels | machine (28) |

OPS-SAT-AD labels are at segment level; onset position within a segment is estimated by BOCPD. Segment boundaries were set by manual expert annotation with ESA's OXI tool. An official train/test partition (`train ∈ {0,1}`, ≈75%/25%) supports train-fit/test-evaluate verification. In SMAP/MSL only the first (telemetry) column of each `.npy` file enters `diff`/`diff²`; the remaining one-hot command columns are excluded. Windows whose values exceed 20 nominal SDs are excluded from the primary test (17 of 105 windows). Datasets are not redistributed; `data/download_*.sh` fetches each public release.

### 3.2 Channel typing (OPS-SAT-AD)

Channels are typed from the distribution of first differences Δ within nominal segments: `quantized` if the fraction of exact-zero Δ is at least 0.10 (step = 25th percentile of non-zero |Δ|); `float_noise_suspect` if not quantized and the smallest non-zero |Δ| is below 1×10⁻⁶; `continuous` otherwise. Thresholds are re-fitted, not reused, on SMAP/MSL and SMD.

**Table 2. Channel typing.**

| Channel | Sensor | Type | frac_zero_diff | Quantization step |
|---|---|---|---|---|
| CADC0872 | magnetometer | float_noise_suspect | 0.053 | n/a |
| CADC0873 | magnetometer | float_noise_suspect | 0.052 | n/a |
| CADC0874 | magnetometer | float_noise_suspect | 0.044 | n/a |
| CADC0884 | photodiode | quantized | 0.183 | ≈0.0144 |
| CADC0886 | photodiode | quantized | 0.371 | ≈0.0274 |
| CADC0888 | photodiode | quantized | 0.354 | ≈0.0144 |
| CADC0890 | photodiode | continuous | 0.083 | n/a |
| CADC0892 | photodiode | quantized | 0.416 | ≈0.0054 |
| CADC0894 | photodiode | quantized | 0.583 | ≈0.0026 |

Channel type maps one-to-one onto sensor family in this dataset; both are also aliased with detector-selection retention (Section 5.6).

### 3.3 Channel scope: 9 → 5

Exclusion criteria are applied in order: structural (zero anomalous segments), underpowered (fewer than 5 anomalous or fewer than 5 nominal segments), chance-level (MCC ≤ 0). The rule is conservative by construction: CADC0890 has the highest point-estimate MCC (0.83) and is excluded for having only 3 nominal segments.

![Fig. 1: channel scope, MCC and Youden's J per channel, and the screening funnel](results/figures/fig01_channel_scope.png)

*Fig. 1. (a) MCC and Youden's J at locked BOCPD hyperparameters; dashed outlines mark excluded channels (S structural, U underpowered, C chance-level). (b) Screening funnel 9 → 6 → 5 → 5 → 5. The final scope (CADC0872, 0873, 0874, 0888, 0894) is identical in full-data, train-only and test-only re-derivations.*

**Table 3. Layer-2 channel scoping (locked hyperparameters).**

| Channel | Sensor | Anomalous / nominal | Segments | Prevalence | MCC | Youden's J | Status |
|---|---|---|---|---|---|---|---|
| CADC0872 | Magn. | 131 / 415 | 546 | 24.0% | 0.51 | 0.32 | included |
| CADC0873 | Magn. | 105 / 488 | 593 | 17.7% | 0.49 | 0.28 | included |
| CADC0874 | Magn. | 69 / 125 | 194 | 35.6% | 0.74 | 0.69 | included |
| CADC0884 | Phot. | 0 / 158 | 158 | 0.0% | n/a | n/a | excluded (S) |
| CADC0886 | Phot. | 3 / 8 | 11 | 27.3% | 0.00 | 0.00 | excluded (U) |
| CADC0888 | Phot. | 60 / 192 | 252 | 23.8% | 0.19 | 0.23 | included |
| CADC0890 | Phot. | 11 / 3 | 14 | 78.6% | 0.83 | 0.91 | excluded (U) |
| CADC0892 | Phot. | 34 / 177 | 211 | 16.1% | −0.09 | −0.02 | excluded (C) |
| CADC0894 | Phot. | 21 / 123 | 144 | 14.6% | 0.19 | 0.20 | included |

The five in-scope channels hold 386 anomalous segments (131 + 105 + 69 + 60 + 21) out of 2,123 segments in total.

![Fig. 5: segments per channel and anomaly prevalence](results/figures/fig05_dataset_and_precedence.png)

*Fig. 5. Segments per channel, split into anomalous and nominal, with anomaly prevalence in italics (2,123 segments; 386 anomalous in scope). Hatched and dotted outlines mark excluded channels. The precedence analysis appears in Fig. 7.*

### 3.4 Onset front end and scoreable subset

A local-linear-trend Kalman filter absorbs a persistent mean shift into its state within a few samples and leaves a short-lived transient in the innovations, so BOCPD on the innovations is a transient-sensitive front end. It yields a scoreable window for 178 of 386 anomalous segments (46.1%): 176 high-reliability, 1 medium and 1 low (177/178 = 99.4% high-or-medium). Nine of the 177 have a non-positive pre- or post-window variance in the `diff` windows (CADC0873: 1, CADC0888: 3, CADC0894: 5), leaving **168 complete-case events** (CADC0872: 41, 0873: 28, 0874: 50, 0888: 33, 0894: 16). Placebo pivots are position-matched resamples on normal segments while anomalous onsets are detector-selected; the detector-matched placebo of Section 5.4 removes this asymmetry.

**Table 4. Sample accounting.**

| Quantity | Value |
|---|---|
| OPS-SAT-AD segments / anomalous in scope | 2,123 / 386 |
| Scoreable → high-or-medium → complete-case | 178 → 177 → 168 |
| Placebo pool, raw / de-duplicated | 4,980 / 3,951 rows (20.7% duplicated) |
| Pooled complete-case rows (168 anomalous + 3,823 de-duplicated placebo) | 3,991 |
| 16-model tabular dataset (168 positive + 4,828 raw placebo) | 4,996 rows |
| Sequence windows, before / after de-duplication | 2,733 / 2,505 |
| SMAP/MSL windows: Table 6 / channel-level Table 15 / FAR-matched | 88 / 68 channels / 56 |
| SMD events / window-level pairs / FAR-matched anomalous windows | 327 / 11,493 / 8,421 |

Each analysis uses the subset for which its inputs are defined; the full accounting is in `results/partE_n_accounting.csv`.

---

## 4. Methods

### 4.1 Pipeline

The pipeline has three layers on OPS-SAT-AD (signal estimation, onset estimation with channel scoping, signature analysis), a labeling-protocol audit with train/test verification, a 16-model classifier-side cross-validation, an external-validation pipeline for SMAP/MSL and SMD, and a robustness re-analysis (cluster-robust AUROC, small-K resampling, FAR matching, detector-matched placebo, tie-aware precedence, leakage and duplicate audit, statistic-definition sensitivity, synthetic regime studies).

### 4.2 Layer 1: signal estimation

Observation and process noise are estimated from pooled nominal-segment differences as r = Var(Δ)/2 and q = Var(Δ²)/6. For quantized channels, step²/12 is added to the observation-noise variance. Efron–Morris-type empirical-Bayes shrinkage stabilizes channels with very few nominal points (CADC0886: n = 8; CADC0890: n = 3). A Gaussian-mixture noise alternative did not improve train macro-MCC and the simple estimator was retained. A local-linear-trend Kalman filter is fitted per channel using the shrunk (r, q); standardized innovations z_t = (y_t − ŷ_t)/√S_t are the sole input to Layer 2.

### 4.3 Layer 2: onset estimation and scoping

BOCPD (Adams and MacKay, 2007) runs on z_t with a Normal-Inverse-Gamma conjugate predictive model, hazard rate 1/250 and a forgetting extension (κ_max = 40). The 2×2 ablation (mixture × forgetting) was evaluated on train macro-MCC over the six scoring channels; train-only reselection reproduces the locked combination.

**Table 5. Layer-2 ablation (train macro-MCC, 6 scoring channels).**

| Mixture | Forgetting | Train macro-MCC |
|---|---|---|
| False | True (**locked**) | **0.308** |
| True | True | 0.275 |
| False | False | 0.238 |
| True | False | 0.199 |

The five-channel scope was re-derived by bootstrap CI, by train-fit/test-evaluate and by test-only bootstrap CI; all three agree (Fig. 1b). Problem-channel diagnosis for CADC0890/0892/0894 rejected a BOCPD forgetting deficiency (Hypothesis B); a noise-model deficiency (Hypothesis A) is supported for CADC0890/0894 under the mixture estimator; CADC0892 is explained by neither (steady-state kurtosis 63.32 drives spurious resets). This localizes detector-tuning difficulty to channel-specific noise modeling, separately from the feature-choice question.

### 4.4 Layer 3: signature analysis

Two hundred onset candidates per segment (seed 42) propagate onset uncertainty; the canonical onset is the median draw and defines the pre/post split. Effects are |*d*| (Cohen's d) for `level` and the log-variance ratio for `diff`, `diff²`. Each anomalous-segment effect is compared by a one-sided Mann–Whitney *U* test with a placebo pool of resampled pivots on normal segments (15 channel × feature tests, Bonferroni). Temporal precedence records the first post-onset |z| > 3 crossing per series and applies a pooled Wilcoxon signed-rank test. Temporal-profile classification, early-window sensitivity (10-point versus full window), DerSimonian–Laird random effects and leave-one-channel-out (LOO) decomposition complete the layer.

### 4.5 Cluster-robust AUROC

Each dataset's onset-versus-placebo separation is re-estimated as an AUROC with a cluster bootstrap (1,000 resamples) at the unit that repeats: segment (OPS-SAT-AD), channel (SMAP/MSL), machine (SMD; channel-level as a secondary check). Paired contrasts Δ(diff − level) and Δ(diff² − level) share resamples. Because `level` falls below 0.5 in OPS-SAT-AD, a direction-agnostic variant scores `level` as max(AUC, 1 − AUC). Incremental value of `level` is measured by an out-of-fold, group-held-out logistic model on `[level, diff, diff²]` versus `[diff, diff²]`.

### 4.6 Small-K resampling

For OPS-SAT-AD (K = 5 channels) the two contrasts are re-estimated under a two-level bootstrap (channel > segment), wild-cluster bootstrap with Webb six-point and Rademacher weights, and exact sign-flip enumeration (2⁵ = 32 patterns), plus leave-one-channel-out fits.

### 4.7 Detector-matched placebo

The stage-2 detector is run on the 996 normal segments (`partC_normal_segments_for_stage2.csv`); its detected resets serve as placebo pivots, so anomalous onsets and placebo pivots are produced by the same detector. Normal-segment detection rates and anomalous retention are reported per channel.

### 4.8 FAR-matched alarm evaluation

For each feature, the alarm threshold is calibrated to a target placebo false-alarm rate on the grid 0.5%, 1%, 2%, 5%, 10%, 20% (pre-specified: 1% and 5%). The fusion rule `level ∨ diff ∨ diff²` is calibrated so that the *joint* placebo FAR equals the target. Coverage is the fraction of anomalous windows alarmed; paired contrasts against `level` use cluster-bootstrap intervals (machine for SMD, channel for SMAP/MSL). Fixed |z| > 3 operating points are reported with their measured FAR.

### 4.9 Precedence and tie handling

Crossing times are integer sample indices, so simultaneous crossings (t_level = t_diff) are frequent. The primary analysis excludes ties; sensitivity analyses score ties as 0.5 and as `diff`-first (the original convention). Intervals are cluster bootstraps (SMAP/MSL: channel, K = 63; SMD: machine, K = 28; OPS-SAT-AD: canonical pairs treated as independent, K = 237). Synthetic AR(1) series calibrate what a pure level shift produces under the criterion.

### 4.10 External validation

Stage 0 applies command-column exclusion, per-dataset channel re-typing, triviality filtering and a sample-size-matched design (SMAP/MSL has a median of 1 window per channel, maximum 3, so dataset-pooled and type-stratified tests are primary). Multiple comparisons use Benjamini–Hochberg FDR with a confirmatory family (dataset-pooled, 3 tests) and an exploratory family (channel-type-stratified). Stage 2 uses ground-truth onsets with an adaptive, non-sliding post-onset search window; a 150 to 5,000-sample sweep showed that the window's upper bound does not induce censoring bias (n = 40 → 41).

### 4.11 Classifier-side cross-validation

Sixteen architecturally diverse models solve the binary task canonical-onset segments versus resampled placebo pivots (4,996 tabular rows; 2,733 sequence windows, 2,505 after de-duplication). Ten tabular models use the three engineered features (SHAP or permutation attribution): LogReg (L2, L1), GaussianNB, kNN, SVM (RBF), RandomForest, ExtraTrees, GradBoost, XGBoost, LightGBM. Six sequence models use per-window z-normalized three-channel series (Integrated Gradients): CNN1D, TCN, BiLSTM, BiGRU, TinyTransformer, LightMamba. A seventh-teenth model (MLP_shallow, AUC 0.403) was removed by the fit-quality screen. Shares are normalized within each model and compared by rank and by the 1/3 uniform reference.

### 4.12 Synthetic studies

*Coverage (K = 5, 500 replications):* single-level, two-level and wild-cluster intervals against the size-weighted and channel-mean estimands. *Pooling bias (300 replications per row; K = 8, 500 replications for the endogenous-size design):* the exact pair-decomposition identity pooled AUROC = [Σₖ n₁ₖn₀ₖ·AUCₖ + cross-cluster pair terms] / (N₁N₀), and simulated bias for size–effect correlation ρ ∈ {−0.8, −0.4, 0, 0.4, 0.8}. *K-sensitivity:* interval widths on SMD machine subsets.

### 4.13 Audit sequence

Every headline statistic passes through: (i) an independent from-scratch replica of the data pipeline, (ii) resampling at the higher level, (iii) FAR matching over an extended grid, (iv) a detector-matched placebo, (v) a tie-handling audit of the precedence criterion, (vi) a duplicate-pool and leakage audit, (vii) a statistic-definition sensitivity analysis, and (viii) a synthetic existence proof.

---

## 5. Results

### 5.1 Data consistency and pipeline integrity

**Table 6. Consistency gate (OPS-SAT-AD).**

| Check | Result |
|---|---|
| Replica vs saved 16-model dataset (4,996 rows) | exact row-for-row match; max relative error 1.8×10⁻¹⁵ |
| Replica vs saved canonical effect values (168 events) | max relative error 7.8×10⁻¹⁵ |
| Replica placebo pool (raw / de-duplicated) | 4,980 / 3,951 rows; matches reported counts |
| Replica re-test of the placebo-pool comparison | same significance pattern in all 15 channel × feature cells |
| Attrition funnel | 386 → 178 scoreable (46.1%) → 177 high-or-medium (99.4%) → 168 complete-case |
| Channel-level retention (scoreable / anomalous) | CADC0872 32.1% (42/131), CADC0873 27.6% (29/105), CADC0874 72.5% (50/69), CADC0888 60.0% (36/60), CADC0894 100.0% (21/21) |
| Pooled AUROC (complete-case n = 3,991) | level 0.438, diff 0.900, diff² 0.917 |
| Stratified-by-channel AUROC (Simpson check) | level 0.434, diff 0.908, diff² 0.937; no material distortion |
| Monte Carlo median vs canonical onset (level AUROC, per channel) | 0.463 vs 0.468, 0.276 vs 0.289, 0.468 vs 0.470, 0.424 vs 0.454, 0.558 vs 0.574 (raw-pool denominators); the Monte Carlo median is not the source of the `level` reversal |

The exact match rules out a class of end-to-end pipeline bugs (onset-index mismatch, feature-definition drift, placebo-pool corruption) as an explanation for any result below. The three magnetometer channels, which carry the strongest derivative signature, have the lowest retention (27.6% to 32.1% for CADC0872/0873; CADC0874 72.5%), so instrument type is aliased with detector selectivity.

### 5.2 Direction: derivative features separate onsets from placebo in 3/3 datasets

**Table 7. Cluster-robust AUROC and paired contrasts (95% cluster-bootstrap CI; OPS-SAT-AD against the position-matched placebo).**

| Dataset | Cluster unit (K) | AUROC `level` | AUROC `diff` | AUROC `diff²` | Δ(diff − level) | Δ(diff² − level) |
|---|---|---|---|---|---|---|
| OPS-SAT-AD | segment (1,163; 168 events) | 0.438 [0.401, 0.477] | 0.900 [0.875, 0.924] | 0.917 [0.890, 0.943] | **+0.462 [0.409, 0.508]** | **+0.479 [0.427, 0.527]** |
| SMAP/MSL | channel (81; 88 windows) | 0.554 [0.497, 0.608] | 0.677 [0.600, 0.746] | 0.642 [0.568, 0.713] | **+0.122 [0.045, 0.196]** | **+0.088 [0.008, 0.164]** |
| SMD | machine (28; 327 events) | 0.534 [0.521, 0.546] | 0.576 [0.553, 0.599] | 0.576 [0.554, 0.597] | **+0.042 [0.016, 0.066]** | **+0.042 [0.017, 0.066]** |
| SMD (reference) | channel (1,064; 327 events) | 0.534 [0.523, 0.546] | 0.576 [0.566, 0.585] | 0.576 [0.566, 0.585] | +0.042 [0.028, 0.054] | +0.042 [0.028, 0.055] |

![Fig. 6: cluster-robust contrasts across datasets and schemes; SMD channel-level breadth](results/figures/fig06_cross_dataset_placebo.png)

*Fig. 6. (a) Δ AUROC(diff − level) and Δ AUROC(diff² − level) with 95% cluster-bootstrap intervals. OPS-SAT-AD rows compare resampling schemes (segment, two-level, wild-Webb, wild-Rademacher, exact sign-flip 2⁵); triangles mark the direction-agnostic `level` variant. The K = 5 rows are read as direction and magnitude because no exact test can reject at 0.05 (minimum p = 0.0625). (b) SMD channel-level breadth: fraction of 1,038 channels significant per feature (Wilson 95% intervals, descriptive).*

The contrast is positive in every row and its size follows the dataset: large in OPS-SAT-AD, moderate in SMAP/MSL and small in SMD. Because `level` lies below 0.5 in OPS-SAT-AD, the direction-agnostic contrast removes the part of +0.462 that arises from the reversed direction.

**Table 8. Direction-agnostic variant.**

| Dataset | `level`, direction-agnostic | Δ(diff − level) | Δ(diff² − level) |
|---|---|---|---|
| OPS-SAT-AD | 0.562 [0.523, 0.599] | **+0.338** | **+0.355** |
| SMAP/MSL | 0.554 (unchanged) | +0.122 | +0.088 |
| SMD | 0.534 (unchanged) | +0.042 | +0.042 |

At channel level in SMD, 97/1,038 channels (9.3%; Wilson 95% 7.7% to 11.3%) are significant for `level`, 142/1,038 (13.7%; 11.7% to 15.9%) for `diff` and 145/1,038 (14.0%; 12.0% to 16.2%) for `diff²`: the derivative features flag about 1.5 times as many channels.

### 5.3 `level` adds no incremental discrimination

**Table 9. Incremental value of `level` (out-of-fold AUROC, `[level, diff, diff²]` minus `[diff, diff²]`).**

| Dataset | Δ AUROC | 95% CI | p (cluster bootstrap) |
|---|---|---|---|
| OPS-SAT-AD (segment) | −0.0007 | [−0.0014, −0.0001] | 0.03 |
| SMAP/MSL (channel) | −0.0068 | [−0.0381, 0.0241] | 0.70 |
| SMD (machine) | −0.0009 | [−0.0143, 0.0129] | 0.93 |

`level` contributes no positive incremental discrimination in any dataset; the OPS-SAT-AD change (|Δ| < 0.001) is negligible in size.

### 5.4 Small-K robustness and the `level` reversal

**Table 10. Δ AUROC under alternative resampling schemes (OPS-SAT-AD, position-matched placebo).**

| Scheme | Unit (K) | Δ(diff − level) [95% CI] | Δ(diff² − level) [95% CI] |
|---|---|---|---|
| Point estimate | n/a | +0.462 | +0.479 |
| Single-level bootstrap | segment (1,163) | [0.409, 0.508] | [0.427, 0.527] |
| Two-level bootstrap | channel > segment (5) | [0.282, 0.593] | [0.327, 0.600] |
| Wild cluster, Webb 6-point | channel (5) | [0.342, 0.582] | [0.379, 0.578] |
| Wild cluster, Rademacher | channel (5) | [0.334, 0.590] | [0.380, 0.578] |
| Exact sign-flip enumeration (2⁵ = 32) | channel (5) | [0.350, 0.574] | [0.384, 0.574] |

Single-level interval widths are 0.099 (`diff`) and 0.100 (`diff²`); two-level widths are 0.311 and 0.273 (2.7 to 3.1 times wider). Both contrasts keep their sign and size under every scheme. Leave-one-channel-out fits preserve the sign as well:

**Table 11. Leave-one-channel-out (OPS-SAT-AD).**

| Channel dropped | AUROC `level` | Δ(diff − level) | Δ(diff² − level) |
|---|---|---|---|
| none (full) | 0.438 | 0.462 | 0.479 |
| CADC0872 | 0.426 | 0.448 | 0.476 |
| CADC0873 | 0.486 | 0.390 | 0.417 |
| CADC0874 | 0.437 | 0.475 | 0.489 |
| CADC0888 | 0.433 | 0.495 | 0.494 |
| CADC0894 | 0.415 | 0.490 | 0.513 |

Pooled `level` moves toward 0.5 only when CADC0873 is removed.

![Fig. 3: channel-level level AUROC, pooled interval by scheme, slice replication, and detector-matched placebo](results/figures/fig03_quasi_experimental.png)

*Fig. 3. (a) Channel-level `level` AUROC (complete-case) and pooled AUROC 0.438 with 95% intervals under four resampling schemes. (b) Full / train-only / test-only significance calls (Bonferroni): 12 of 13 decidable calls agree; the single disagreement (CADC0888 `diff`, test-slice p = 0.051) is a monotone power loss. (c) Position-matched (open circle) versus detector-matched (filled, 95% CI) placebo; normal-segment detection rate / anomalous retention: 0872 18%/31%, 0873 22%/28%, 0874 30%/72%, 0888 99%/60%, 0894 88%/100%. The detector was run on the 996 normal segments used as placebo; detected resets are pivots.*

**Table 12. Pooled `level` AUROC (0.438) under the same schemes.**

| Scheme | 95% CI | p vs 0.5 | Excludes 0.5 |
|---|---|---|---|
| Single-level (segment) | [0.401, 0.477] | 0.004 | yes |
| Two-level (channel > segment) | [0.350, 0.543] | 0.244 | no |
| Wild cluster, Webb | [0.364, 0.510] | 0.151 | no |
| Exact enumeration | [0.370, 0.506] | 0.182 | no |

The pooled `level` reversal is significant only when segments are treated as the independent unit; under channel-level resampling it is indistinguishable from chance. Channel-level `level` AUROCs are 0.468, 0.289, 0.470, 0.476 and 0.593 for CADC0872, 0873, 0874, 0888 and 0894 (n = 41, 28, 50, 33, 16). The within-channel CADC0873 value (0.289; two-sided p_bonf = 0.002) rests on segment clusters inside one channel. We therefore read `level` in OPS-SAT-AD as inconclusive rather than as evidence for or against discrimination.

### 5.5 Detector-matched placebo: selection inflation in OPS-SAT-AD

**Table 13. Onset-versus-placebo AUROC by placebo type (OPS-SAT-AD, pooled).**

| Feature | Position-matched | Detector-matched (95% CI) |
|---|---|---|
| `level` | 0.438 | 0.382 [0.327, 0.441]; direction-agnostic 0.618 |
| `diff` | 0.900 | **0.569 [0.518, 0.626]** |
| `diff²` | 0.917 | **0.720 [0.675, 0.769]** |

Per channel, `diff` under the detector-matched placebo falls to 0.136 in CADC0888 and 0.314 in CADC0894 (Fig. 3c). Detector-matched placebo counts are 254 pivots. The pooled derivative AUROCs remain above 0.5 (`diff` [0.518, 0.626], `diff²` [0.675, 0.769]), so both derivative features retain discrimination against a detector-produced reference, but the magnitude is much smaller than the position-matched value. `diff²` remains above the direction-agnostic `level` (0.720 vs 0.618); `diff` does not (0.569 vs 0.618). We therefore treat OPS-SAT-AD as a case study whose magnitude is detector-conditioned, and report the selection-conditioned difference as an inflation term in Section 6.2.

### 5.6 Placebo-pool comparison, slice replication and channel heterogeneity

**Table 14. Placebo-pool comparison (OPS-SAT-AD, raw placebo pool, one-sided Mann–Whitney, Bonferroni over 15 tests).**

| Feature | Channels significant | Significant-segment fraction |
|---|---|---|
| `level` | 0/5 (p_bonf capped at 1.000) | 54% to 92% within-segment (§4.4 pre/post test) |
| `diff` | 5/5 (all p_bonf < 2×10⁻⁴) | magnetometers 0.78 to 0.89; photodiodes 0.12 to 0.42 |
| `diff²` | 5/5 (all p_bonf < 2×10⁻⁴) | magnetometers 0.78 to 0.89; photodiodes 0.12 to 0.42 |

A p_bonf of 1.000 for `level` is the Bonferroni cap (raw p × 15 ≥ 1), and is read as no detectable excess over placebo; no equivalence test was run. The de-duplicated pool (3,951 rows) changes p-values by at most a factor of about three and no significance call. Train-only reproduction (178,504 of 240,979 rows, 74.1%; 126 anomalous segments) matches full data in 15/15 calls; test-only reproduction (62,475 rows, 25.9%; 51 anomalous segments) matches in 12/13 decidable calls, with one borderline (CADC0888 `diff`, p = 0.051) and two not decidable (CADC0894, n = 4 < 5).

Within-segment significance shows that `level` is significant in 54% to 92% of segments per channel and `diff`/`diff²` in 2% to 76%. Temporal-profile classification reconciles this with the precedence test: `diff` is transient in 98% of segments, `diff²` in 94%, and `level` is a near-even mix (50%/45%); the `level` significance rate falls from 0.864 to 0.633 in the early window, a larger relative loss than for `diff`/`diff²`.

![Fig. 4: DerSimonian–Laird heterogeneity across five channels](results/figures/fig04_scm_and_heterogeneity.png)

*Fig. 4. Random-effects heterogeneity, k = 5 channels, for effect size |d| (top) and significant-segment fraction (bottom). Bands follow Higgins–Thompson conventions; with k = 5, I² and Q are descriptive.*

**Table 15. Between-channel heterogeneity (DerSimonian–Laird, k = 5).**

| Statistic | Feature | I² | Q p |
|---|---|---|---|
| Effect size \|d\| | `level` | 65.7% | 0.020 |
| Effect size \|d\| | `diff` | 97.2% | 1.0×10⁻⁴ |
| Effect size \|d\| | `diff²` | 93.8% | 2.6×10⁻¹³ |
| Significant-segment fraction | `level` | 82.8% | 1.1×10⁻⁴ |
| Significant-segment fraction | `diff` | 93.8% | 3.2×10⁻¹³ |
| Significant-segment fraction | `diff²` | 92.2% | 1.9×10⁻¹⁰ |

Leave-one-out decomposition localizes the heterogeneity: removing CADC0872 eliminates all `level`-|d| heterogeneity, removing CADC0874 eliminates all `diff²`-|d| heterogeneity, and removing CADC0894 nearly halves the pooled `diff`/`diff²` significance fraction. The magnetometer-over-photodiode pattern is aliased with channel type and with detector retention (28% to 100%), and `diff` reverses in the two photodiode channels under detector matching; we treat it as a hypothesis for future instrument-level work.

### 5.7 External replication (SMAP/MSL, SMD)

**Table 16. Placebo-pool comparison across datasets (pooled p-values are nominal one-sided tests; windows are nested in channels and machines, so Table 7 is the calibrated comparison).**

| Dataset | Scope | `level` | `diff` | `diff²` |
|---|---|---|---|---|
| OPS-SAT-AD | channel-level (n = 5) | 0/5 significant | 5/5 (all p < 2×10⁻⁴) | 5/5 (all p < 2×10⁻⁴) |
| SMAP/MSL | dataset-pooled (88 windows) | p_fdr = 0.031 | p_fdr = 4.7×10⁻⁹ | p_fdr = 1.3×10⁻⁶ |
| SMD | dataset-pooled (11,493 windows ≈ 327 events × ≤ 38 dimensions) | p_fdr = 8×10⁻³⁸ | p_fdr = 3.7×10⁻¹⁶⁷ | p_fdr = 1.4×10⁻¹⁶⁶ |

Outside OPS-SAT-AD, `level` carries a small component above placebo in the pooled test; in SMAP/MSL its channel-cluster interval includes 0.5 (0.554 [0.497, 0.608]), so the component is weak there and small but reliable in SMD (0.534 [0.521, 0.546]). Among channel types, SMAP/MSL `continuous` channels show `level` n.s. (p_fdr = 0.35) with `diff`/`diff²` significant, which is the OPS-SAT-AD pattern; `quantized` channels show `level` weakly significant. `float_noise_suspect` cannot be tested in SMAP/MSL (1 of 82 channels, 3 windows) and shows no significant effect for any feature in SMD (162 windows across 17 channels), so the instrument-type moderator is untestable externally.

**Table 17. Statistic-definition sensitivity: `level` with pooled-SD versus pre-onset-SD denominator (`level_prez`).**

| Dataset | Increase in `level` AUROC with `level_prez` | Δ(diff − level) with pooled SD | Δ(diff − level_prez) | p |
|---|---|---|---|---|
| OPS-SAT-AD | not an explanation for the reversal | +0.462 | +0.447 | ≤ 0.002 |
| SMAP/MSL (68 channels) | +0.080 | +0.122 | +0.060 | 0.22 |
| SMD | +0.035 | +0.042 | +0.010 | 0.50 |

Part of the external `diff` advantage is therefore definitional: when `level` is standardized by the pre-onset SD, as `diff` effectively is, the two features are not distinguishable externally. Together with Table 9, this frames the derivative features as *complementary and sufficient* rather than as uniformly superior.

### 5.8 Operating value at a matched false-alarm rate

![Fig. 6c: coverage versus FAR for SMD and SMAP/MSL, with fusion-minus-level contrasts](results/figures/fig06c_coverage_vs_far.png)

*Fig. 6c. (a, b) Coverage of anomalous windows against placebo FAR (log axis) for `level`, `diff`, `diff²` and the fusion rule `level ∨ diff ∨ diff²`; the shaded region is FAR-matched (calibrated), crosses are fixed |z| > 3 at their measured FAR. (c, d) Fusion-minus-`level` coverage with cluster-bootstrap 95% intervals; bold marks the pre-specified 1% and 5% targets. SMD: n = 8,421 anomalous windows, machine clusters. SMAP/MSL: n = 56 windows, channel clusters.*

**Table 18. Fusion-minus-`level` coverage (percentage points; cluster-bootstrap 95% CI).**

| FAR target | SMD (n = 8,421; machine clusters) | SMAP/MSL (n = 56; channel clusters) |
|---|---|---|
| 0.5% | +0.2 [0.0, 0.6] | −1.8 [−5.9, 0.0] |
| **1%** | **+0.8 [0.3, 1.4]** | **+0.0 [0.0, 0.0]** |
| 2% | +0.7 [−0.1, 1.5] | −1.8 [−7.4, 5.5] |
| **5%** | **+1.2 [0.4, 2.0]** | **+3.6 [0.0, 11.3]** |
| 10% | +1.5 [−0.0, 3.2] | +0.0 [−5.9, 6.8] |
| 20% | +1.8 [0.4, 3.4] | +5.4 [−2.0, 13.0] |

In SMD, coverage at 5% FAR rises from 15.9% (`level`) to 17.2% (fusion), a gain of +1.2 pp. The gain is positive at all six targets (+0.25 to +1.84 pp) with intervals excluding zero at four of six (0.5%, 1%, 5%, 20%; the 0.5% lower bound rounds to 0.0). In the 56-window SMAP/MSL sample the sign of the gain changes across targets, so that sample does not rank fusion against single-derivative rules. Fixed |z| > 3 has a placebo false-alarm rate of 46.4% (`level`) and 37.8% (`diff`) in SMAP/MSL and 43.7% and 27.2% in SMD; the crosses in Fig. 6c sit at different operating points, so their gaps partly reflect FAR, not the feature. A FAR-matched pipeline on raw telemetry is not available for OPS-SAT-AD.

### 5.9 Estimand-dependent resampling and pooling bias

![Fig. 6b: interval coverage by estimand and pooling bias under endogenous cluster size](results/figures/fig06b_coverage_and_pooling_bias.png)

*Fig. 6b. (a) Coverage of nominal 95% intervals for the size-weighted (pooled) and channel-mean (cluster) estimands in a synthetic two-level design (K = 5, 500 replications). (b) Bias of pooled and equal-weight AUROC relative to the channel-mean AUROC when cluster size correlates with effect size (ρ = −0.8) or is independent (ρ = 0); K = 8, 500 replications; mean ± 1 SD.*

**Table 19. Coverage of 95% intervals (K = 5, 500 replications, MC SE ≈ 0.010).**

| Procedure | Coverage, size-weighted target | Coverage, channel-mean target | Mean width |
|---|---|---|---|
| Single-level (segment) | 96.2% | **79.2%** | 0.066 |
| Two-level | 99.6% | 99.4% | 0.188 |
| Wild cluster | 98.2% | 97.4% | 0.163 |

The single-level interval is near nominal for the pooled estimand and under-covers the channel-mean estimand; channel-level procedures cover both but are conservative. The appropriate resampling unit is determined by the estimand.

**Table 20. Pooled versus equal-weight AUROC (300 replications per row).**

| Size–effect correlation ρ | Mean (pooled − equal-weight) | SD |
|---|---|---|
| −0.8 | −0.0488 | 0.031 |
| −0.4 | −0.0245 | 0.032 |
| 0.0 | +0.0003 | 0.030 |
| +0.4 | +0.0224 | 0.031 |
| +0.8 | +0.0469 | 0.029 |

The pair-decomposition identity holds to 0 (pooled) and 2.8×10⁻¹⁷ (weighted-mean versus covariance form) on test data and is enforced by `layer3_signature_analysis/tests/test_pooled_identity.py`; the sign of the simulated bias matches the identity in 5/5 conditions.

**Table 21. Endogenous cluster size (retention) and pooled-AUROC bias (K = 8, 500 replications).**

| Mechanism | Pooled bias (mean, SD) | Equal-weight bias (mean, SD) |
|---|---|---|
| Size correlated with effect (ρ = −0.8) | −0.0487 (0.0300) | −0.0004 (0.0208) |
| Size independent of effect | +0.0003 (0.0266) | −0.0001 (0.0222) |

The observed OPS-SAT-AD retention versus `diff` AUROC gives Spearman ρ = −0.60 (t-approximation p = 0.285; exact permutation p = 0.35; n = 5); the retention-weighted mean `diff` AUROC is 0.864 versus 0.885 equal-weight (difference 0.021). The observed correlation has the sign of the simulated mechanism; with n = 5 it is compatible with, but does not establish, that mechanism.

**Table 22. K-sensitivity of interval widths (SMD machine subsets; descriptive).**

| K (machines) | Reps | Width single | Width hierarchical | Ratio hier/single | Ratio wild/single |
|---|---|---|---|---|---|
| 5 | 20 | 0.082 | 0.148 | 1.79 | 1.22 |
| 12 | 20 | 0.067 | 0.113 | 1.68 | 1.10 |
| 20 | 20 | 0.065 | 0.102 | 1.57 | 0.90 |
| 28 | 1 | 0.064 | 0.090 | 1.41 | 0.80 |

The hierarchical/single ratio decreases with K (log–log slope −0.11, p = 0.004) and remains about 1.4 at K = 28. Subsets overlap and K = 28 is a single full-data replicate, so this is a descriptive trend.

### 5.10 Classifier-side robustness and leakage audit

![Fig. 2: feature-importance shares for six sequence models under three protocols](results/figures/figS_fig02_sequence_partial.png)

*Fig. 2. Normalized feature-importance shares (Integrated Gradients) for the six sequence models in the original arm, the re-run with test-fold early stopping, and the re-run with within-training-fold early stopping. † marks `level` share ≥ 1/3; the dotted line is the 1/3 uniform reference. Models share data and labels, so counts are descriptive; AUC is out-of-fold.*

**Table 23. Sequence-model arm by training protocol (6 models).**

| Protocol | `level` top-ranked | `diff` top | `diff²` top | `level` share < 1/3 | Out-of-fold AUC range |
|---|---|---|---|---|---|
| Original arm | 0/6 | 4 | 2 | 5/6 | 0.969 to 0.994 |
| Re-run, test-fold early stopping | 1/6 (CNN1D) | 4 | 1 | 4/6 | 0.982 to 0.995 |
| Re-run, within-training-fold early stopping | 0/6 | 4 | 2 | 6/6 | 0.960 to 0.982 |

In the original 16-model arm, 0/16 models rank `level` first (`diff` first in 5, `diff²` in 11; binomial reference (2/3)¹⁶, p = 0.0015, descriptive); tabular 0/10, sequence 0/6; 15/16 give `level` less than 1/3 (the exception is LightMamba, 0.428). Kendall's *W* = 0.609 (χ² = 19.50, df = 2, p = 5.8×10⁻⁵) for the original arm; bootstrap stability over 200 resamples favors `diff`+`diff²` over `level` in 100% of resamples for RandomForest and LogisticRegression. With three features, "`diff` + `diff²` > `level`" is expected 2:1 under a uniform-share null, so the informative quantities are the top-rank count and the size of the `level` share.

The tabular dataset was re-derived from raw time series (exact match) and re-run with a de-duplicated placebo pool, multi-seed stratified-group CV and a row-level CV as a leakage positive control (AUC moves by at most 0.017; positive control |Δ| ≤ 0.009, non-systematic). The six sequence models were re-run under the two early-stopping protocols of Fig. 2; under the within-training-fold protocol (60 epochs) no sequence model ranks `level` first, whereas under test-fold early stopping CNN1D does (0.371 versus 0.362 for `diff²`). This check controls the classifier side only; dependence on the onset estimator is addressed by the external datasets (Section 5.7).

### 5.11 Temporal precedence and tie handling

Crossing times are integer sample indices, so simultaneous crossings occur in 45% / 33% of SMAP/MSL anomalous / normal pairs and 66% / 61% of SMD pairs. The primary analysis excludes ties.

**Table 24. Precedence ordering and normal-control baseline (ties excluded; 95% cluster CI).**

| Dataset | Regime | n pairs | `diff` precedes `level` |
|---|---|---|---|
| OPS-SAT-AD | anomalous | 39 | 71.8% [56.4, 84.6] |
| OPS-SAT-AD | normal control | 198 | 36.9% [30.3, 43.4] |
| SMAP/MSL | anomalous | 22 | 40.9% [19.0, 62.5] |
| SMAP/MSL | normal control | 5,074 | 34.8% [28.7, 41.7] |
| SMD | anomalous | 1,219 | 30.6% [26.4, 36.3] |
| SMD | normal control | 2,770 | 32.1% [29.9, 34.2] |

![Fig. 7: precedence with ties excluded and anomaly-attributable increment under three tie conventions](results/figures/fig07_precedence_baseline.png)

*Fig. 7. (a) Non-tied pairs in which `diff` crosses |z| > 3 before `level`, anomalous versus normal control; small open circles show the originally reported values (ties counted as `diff`-first). (b) Anomalous-minus-normal increment in percentage points with cluster-bootstrap 95% intervals under three tie conventions.*

**Table 25. Increment (anomalous − normal), percentage points, with tie-handling sensitivity.**

| Dataset | Primary: ties excluded | Ties = 0.5 | Ties → `diff`-first (original) |
|---|---|---|---|
| OPS-SAT-AD | **+34.9 [18.6, 49.2]** (p_boot = 0.002) | not computed | +40.1 [29.4, 50.8] (nominal; original pairing rule, 70/224) |
| SMAP/MSL | **+6.1 [−14.3, 26.0]** (p_boot = 0.64) | +5.1 [−6.0, 16.3] | +11.0 [−4.0, 23.9] |
| SMD | **−1.5 [−5.5, 3.5]** (p_boot = 0.49) | +0.4 [−1.2, 2.1] | +3.0 [−0.9, 6.4] |

With ties excluded, `diff` precedes `level` in fewer than half of non-tied pairs at placebo pivots in all three datasets (32% to 37%); counting ties as `diff`-first had produced baselines of 44.2%, 56.5% and 72.7%. Only the OPS-SAT-AD increment excludes zero; in the external datasets the increment is not distinguishable from zero and, in SMD, changes sign with the tie convention. The criterion has limited diagnostic power: it uses a fixed |z| > 3 threshold whose placebo FAR is 27% to 46%, it conditions on pairs where features cross, and in synthetic AR(1) series with φ ≥ 0.8 a *pure level shift* already yields `diff`-first in 74% to 92% of pairs. The decomposition is therefore a calibration control and is not used as evidence for a variance-surge mechanism. Pair counts in Table 24 follow the canonical-pair rule; the original pairing rule gives 70/224 (`level` vs `diff`), 70/171 (`level` vs `diff²`) and 161/211 (`diff` vs `diff²`), where the `diff` versus `diff²` sign fraction (92.5% vs 35.5%) and the signed-rank test (p = 0.368) disagree, so no claim rests on that comparison. The SMD normal-control count is 7,105 pairs in `stage2_pairs.csv` (73.5% under the original counting rule), against 7,304 (72.7%) in the first version; the anomalous count (3,612) and the SMAP/MSL counts (40 / 7,596) reproduce.

---

## 6. Discussion

### 6.1 Principal findings

1. **Direction.** Derivative-variance features separate labeled onsets from same-channel placebo pivots more strongly than `level` in every dataset, and the direction is stable across resampling units, K, FAR targets and classifier architectures. Two of the three onset sources are ground-truth labels, so the direction does not depend on an onset estimator.
2. **Sufficiency.** Once `diff`/`diff²` are present, `level` adds no out-of-fold discrimination in any dataset. An alarm feature set built on derivative-variance channels loses nothing in discrimination relative to one that also includes `level`.
3. **Operating value.** The measured gain in coverage at matched FAR is modest: +1.2 pp at 5% FAR and +0.8 pp at 1% FAR in SMD, positive at all six FAR targets.
4. **Evaluation sensitivity.** The size of an apparent advantage depends on the placebo, the resampling unit, the statistic definition and the tie convention (Section 6.2).

### 6.2 The inflation ledger

**Table 26. Evaluation choices and their effect on the apparent advantage.**

| Source | Naive reading | Corrected reading | Where |
|---|---|---|---|
| Placebo selection (OPS-SAT-AD) | `diff` 0.900, `diff²` 0.917 | `diff` 0.569 [0.518, 0.626], `diff²` 0.720 [0.675, 0.769] | Table 13, Fig. 3c |
| Direction of `level` | Δ(diff − level) = +0.462 | direction-agnostic +0.338 | Table 8, Fig. 6a |
| Resampling unit | `level` 0.438 [0.401, 0.477], p = 0.004 | [0.350, 0.543], p = 0.244 (two-level) | Table 12, Fig. 3a |
| Interval width of the contrast | 0.099 (segment) | 0.311 (two-level), 2.7 to 3.1 times wider | Table 10 |
| Statistic definition of `level` | +0.122 (SMAP/MSL), +0.042 (SMD) | +0.060 (p = 0.22), +0.010 (p = 0.50) | Table 17 |
| Tie handling in first-crossing precedence | baselines 44.2 / 56.5 / 72.7% | 36.9 / 34.8 / 32.1% | Table 24, Fig. 7 |
| Pooling versus equal weighting | pooled bias −0.049 at ρ = −0.8 | ≈ 0 at ρ = 0; equal-weight bias ≈ 0 in both designs | Tables 20, 21, Fig. 6b |
| Training protocol for sequence models | 1/6 rank `level` first (test-fold early stopping) | 0/6 (within-training-fold early stopping) | Table 23, Fig. 2 |

Each row is an analysis choice that changes the reading of the same data. The ledger is the paper's transferable contribution: when a feature advantage is reported on nested telemetry labels, each row identifies a question to ask and a corrected number to expect.

### 6.3 Design rules for alarm engineers

**Table 27. Design and reporting rules that follow from the evidence (associational).**

| # | Rule | Evidence |
|---|---|---|
| R1 | Add a derivative-variance channel (`diff` and/or `diff²`) to the alarm feature set; expect a gain of about 1 pp coverage at 5% FAR | Tables 7, 18; Figs. 6a, 6c |
| R2 | Combine features with an OR rule and calibrate the joint threshold on the joint placebo FAR, not per-feature fixed thresholds | Table 18; fixed |z| > 3 has FAR 27% to 46% |
| R3 | Do not expect `level` to add discrimination once derivative features are present | Table 9 |
| R4 | When evaluating on detector-selected onsets, use a placebo produced by the same detector and expect the offline advantage to shrink | Table 13, Fig. 3c |
| R5 | Resample at the unit that repeats; report pooled and channel-level intervals and name the estimand | Tables 10, 12, 19; Figs. 3a, 6b |
| R6 | Report `level` AUROC together with its direction-agnostic value | Table 8 |
| R7 | Report ties and vary their handling when using first-crossing criteria on integer-sample data | Tables 24, 25; Fig. 7 |
| R8 | Prioritize sensors by instrument type only when family, type and detector retention are separated | Fig. 3c, Table 15 |

### 6.4 Claim ledger

**Table 28. Evidence status of each claim.**

| Claim | Status | Evidence | Boundary |
|---|---|---|---|
| `diff`/`diff²` exceed `level` in cluster-robust onset-vs-placebo AUROC, 3/3 datasets | Established (direction) | Table 7, Fig. 6a | Size: large (OPS-SAT-AD), moderate (SMAP/MSL), small (SMD) |
| Direction holds at channel level with K = 5 | Established (direction and magnitude) | Tables 10, 11 | Minimum attainable p = 0.0625 |
| `level` adds nothing given derivative features | Established | Table 9 | Out-of-fold, group-held-out logistic model |
| Fusion raises coverage at matched FAR in SMD | Established | Table 18, Fig. 6c | +1.2 pp at 5%; four of six targets exclude zero |
| Fusion gain in SMAP/MSL | Not resolved | Table 18 | 56 windows; sign changes across targets |
| `diff²` exceeds `level` under detector-matched placebo | Established for the pool | Table 13, Fig. 3c | Per-channel reversals in CADC0888/0894 |
| `diff` exceeds direction-agnostic `level` under detector-matched placebo | Not supported | Table 13 | 0.569 vs 0.618 |
| Pooled `level` reversal in OPS-SAT-AD | Inconclusive | Table 12 | Significant only at segment level |
| Magnetometer > photodiode derivative effect | Hypothesis | Fig. 3c, Table 15 | Aliased with type, family and retention |
| Precedence increment | Descriptive | Table 25, Fig. 7 | Only OPS-SAT-AD excludes zero |
| No `level`-first model in 16 (original arm) | Descriptive | Table 23, Fig. 2 | Sequence side depends on the training protocol |

### 6.5 Limitations and scope of validity

**Table 29. Scope of validity.**

| Limitation | Consequence for the claims | What extends the evidence |
|---|---|---|
| K = 5 channels in OPS-SAT-AD | OPS-SAT-AD contrasts are direction and magnitude, not formal tests | additional missions or labeled channels |
| OPS-SAT-AD onsets depend on the BOCPD front end (46.1% scoreable) | magnitude is detector-conditioned | detector-matched placebo (Section 5.5); alternative front ends |
| Detector-matched intervals are segment-level | between-channel uncertainty is not reflected in Table 13 | channel-level resampling of the detector-matched intervals |
| External `diff` advantage depends on the `level` definition | with `level_prez`: +0.060 (p = 0.22) and +0.010 (p = 0.50) | equal-footing statistic definitions in future protocols |
| No comparison with published detectors | the paper claims a feature-level result, not a detector that outperforms published ones | baseline-detector comparison on the same labels |
| No raw-telemetry FAR pipeline for OPS-SAT-AD | operating value is measured in SMD and SMAP/MSL only | FAR pipeline on OPS-SAT-AD raw telemetry |
| SMAP/MSL FAR-matched sample is 56 windows | fusion is not ranked against single-derivative rules there | larger labeled sample |
| Tabular SHAP shares and Kendall's *W* are reported for the original arm only | the re-run arm covers sequence models | method-matched SHAP re-run of the ten tabular models |
| Two pairing rules for OPS-SAT-AD precedence (70/224 original, 39/198 canonical) | conclusions rest on the canonical-pair rule | crossing-time inputs under a single pairing rule |
| No documented quantitative segment-cutting rule in OPS-SAT-AD; anomalous onsets skew toward the back half of segments (mean position ratio ≈ 0.569) | labeling provenance is open; its consequence for the direction is bounded by SMAP/MSL and SMD, which follow different protocols | provenance documentation from the dataset authors |

The study is associational. It does not identify a causal mechanism linking onsets to derivative-statistic surges, does not claim superiority over published detectors, and does not claim a general instrument-physics effect. `scm_skeleton.py` is retained for provenance only.

### 6.6 Directions for future work

Paper 2 builds risk-aware thresholds (CVaR, conformal calibration) on the features selected here; the DSS prototype consumes both. Within this line, priorities are a reduced replication of the 16-model benchmark on SMAP/MSL or SMD, channel-level resampling of the detector-matched intervals, and a FAR pipeline for OPS-SAT-AD raw telemetry.

---

## 7. Conclusion

Across three benchmarks with different platforms, sensors, labeling protocols and onset provenance, derivative-variance features carry at least as much onset information as signal level, `level` adds nothing once they are present, and a fused alarm with a joint-FAR-calibrated threshold adds a measurable but modest coverage gain at matched false-alarm rate. The size of the advantage depends on the placebo, the resampling unit, the statistic definition and the tie convention; the inflation ledger and the accompanying rules give a protocol for reporting feature-discrimination claims on nested telemetry labels. The recommendation for practice is to add a derivative-variance channel to the alarm feature set, calibrate on the joint false-alarm rate, and state the estimand, placebo and tie convention behind any reported advantage.

---

## 8. Repository structure

```
opssat-ad-onset-signatures/
│
├── README.md                              # this file — full methodology + inline figures
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── Makefile                               # `make all` reproduces every figure/table below
├── care_protocol_v8.py                    # §11.3 small-K resampling (P2), coverage (P1), K-sensitivity (P3), retention bias (P4), tier rules (P5), method table (P6); writes results_care_v8/
├── proposition1_verification.py           # §11.3 pair-decomposition identity check and bias-direction simulation (Table R9)
├── run_pending_analyses.py                # §11.4, §11.9, §11.6 analyses: detector-matched placebo (C), tie-aware Table 13 (D), sequence-model re-run (A, sequence part), extended FAR grid (F)
├── ress_priority_analyses_v5.py           # loaders, detection dataset, FAR-matched alarm evaluation (imported by run_pending_analyses.py)
├── ress_followup_v6.py                    # replica, detector-matched placebo, pair-level Table 13, sequence re-run (imported by run_pending_analyses.py)
├── make_pending_figures.py                # renders Fig. 3 (panel c) and Fig. 6c from results_pending/ and results_ress_followup_v6/
├── extract_inputs.py                      # rebuilds figure-input CSVs from README tables/captions (see §15; figures made this way are consistency checks only)
│
├── data/
│   ├── download_opssat_ad.sh              # fetches OPS-SAT-AD from Zenodo (no redistribution)
│   ├── download_smap_msl.sh               # fetches SMAP/MSL (Telemanom) release (no redistribution)
│   ├── download_smd.sh                    # fetches SMD (OmniAnomaly) release (no redistribution)
│   ├── README.md                          # dataset layout, checksum, license notes for all three datasets
│   └── raw/                               # (gitignored) populated by download scripts
│
├── layer1_signal_estimation/
│   ├── channel_classification.py          # quantized / float_noise_suspect / continuous auto-tagging
│   ├── noise_estimation.py                # var(diff)/2 vs. mixture-model comparison
│   ├── kalman_filter.py                   # local-linear-trend Kalman filter, state estimation
│   ├── hierarchical_shrinkage.py          # Bayesian shrinkage for low-sample channels
│   ├── tests/
│   └── run.py
│
├── layer2_anomaly_detection/              # onset estimation (BOCPD) + channel scoping
│   ├── bocpd_forgetting.py                # BOCPD with forgetting-factor extension
│   ├── ablation_mixture_forgetting.py     # 2×2 ablation grid (mixture × forgetting)
│   ├── channel_scoping.py                 # MCC / Youden's J channel decision table (Fig. 1a)
│   ├── triple_verification.py             # full / train-only / test-only re-fit & re-select (Fig. 1b)
│   ├── bootstrap_ci.py                    # per-channel MCC bootstrap confidence intervals
│   ├── problem_channel_diagnosis.py       # Hypothesis A/B diagnosis for 890/892/894 (§ Layer 2.5)
│   ├── tests/
│   └── run.py
│
├── layer3_signature_analysis/
│   ├── onset_mc_propagation.py            # Monte Carlo propagation of onset-time posterior (200 draws)
│   ├── temporal_precedence_test.py        # pooled Wilcoxon signed-rank, level vs diff vs diff² (Fig. 5b; original pairing rule)
│   ├── temporal_profile_classification.py # transient vs. persistent post-onset trajectory classification
│   ├── early_window_sensitivity.py        # 10-point vs. full-window Welch-test comparison
│   ├── quasi_experimental_placebo.py      # Mann–Whitney U vs. same-channel placebo pool (Fig. 3a)
│   ├── meta_analysis_random_effects.py    # DerSimonian–Laird random-effects meta-analysis + I² (Fig. 4)
│   ├── loo_sensitivity.py                 # leave-one-channel-out sensitivity analysis
│   ├── scm_skeleton.py                    # retained for provenance; the structural working model is no longer cited (see §14)
│   ├── supplementary_diagnostics_A_D.py   # cross-correlation, CMH test, normal-segment controls, lag resolution
│   ├── labeling_protocol_audit.py         # external audit of the OPS-SAT-AD labeling protocol
│   ├── train_test_triple_reverification.py# train-only / test-only reproduction of the placebo-pool test
│   ├── tests/                             # includes test_pooled_identity.py: identities (*) and (**) as regression tests
│   └── run.py
│
├── model_cross_validation/
│   ├── models/                            # 16 model wrappers (+1 excluded by the fit-quality screen)
│   │   ├── linear.py                      # LogReg (L1, L2)
│   │   ├── probabilistic.py               # GaussianNB
│   │   ├── kernel.py                      # SVM (RBF)
│   │   ├── instance_based.py              # kNN
│   │   ├── tree_ensembles.py              # RandomForest, ExtraTrees, GradBoost, XGBoost, LightGBM
│   │   ├── shallow_mlp.py                 # MLP_shallow — excluded (AUC=0.403, below chance; failed fit)
│   │   ├── conv_sequence.py               # CNN1D, TCN
│   │   ├── recurrent_sequence.py          # BiLSTM, BiGRU
│   │   ├── attention_sequence.py          # TinyTransformer
│   │   └── ssm_sequence.py                # LightMamba (pure-PyTorch selective state-space substitute)
│   ├── representations/
│   │   ├── tabular_summary_features.py    # 10 models: engineered summary-statistic features
│   │   └── raw_sequence_features.py       # 6 models: raw 3-channel windowed time series
│   ├── attribution/
│   │   ├── shap_permutation.py            # tabular-model attribution (SHAP + permutation fallback; method recorded per row)
│   │   └── integrated_gradients.py        # sequence-model attribution (Captum / manual IG fallback)
│   ├── agreement_statistics.py            # Kendall's W, binomial test, bootstrap stability (original arm)
│   ├── tests/
│   └── run_all.py
│
├── external_validation/
│   ├── stage0_preprocessing.py            # command-column exclusion, channel re-typing, triviality filter
│   ├── stage1_placebo_pooled.py           # dataset-pooled and channel-type-stratified placebo tests (FDR) (Fig. 6)
│   ├── stage2_temporal_precedence.py      # ground-truth-onset precedence + normal-segment control; writes pair-level rows
│   ├── window_cap_sensitivity.py          # 150–5,000-sample search-window sensitivity sweep
│   └── run_external_validation.py
│
├── results/
│   ├── README.md                          # index of every artifact below + provenance table + sample-size accounting (partE_n_accounting.csv)
│   ├── layer1/  (noise_estimates.csv, channel_classification.json)
│   ├── layer2/
│   │   ├── channel_scope.json                  # → consumed by opssat-ad-risk-optimization
│   │   ├── channel_scope_full_table.csv, channel_scope_final_locked.csv
│   │   ├── ablation_grid.csv, ablation_best_combo_test_eval.csv
│   │   ├── bootstrap_ci.csv, bootstrap_ci_test_only.csv
│   │   └── train_fit_test_eval.csv, full_vs_test_ci_comparison.csv
│   ├── layer3/
│   │   ├── onset_posteriors.parquet             # → consumed by opssat-ad-risk-optimization & opssat-ad-dss
│   │   ├── placebo_comparison.csv
│   │   ├── temporal_precedence.csv, temporal_precedence_normal_control.csv
│   │   ├── temporal_precedence_group_comparison.csv, temporal_precedence_lag_resolution.csv
│   │   ├── heterogeneity_summary.csv            # → consumed by opssat-ad-dss
│   │   ├── triple_verification_matrix.csv
│   │   ├── per_channel_auroc.csv                # channel-level AUROC, complete-case; → Fig. 3a
│   │   ├── cluster_robust_contrasts.csv         # Table R2 contrasts incl. direction-agnostic; → Fig. 6a
│   │   ├── partC_normal_segments_for_stage2.csv # the 996 normal segments given to the stage-2 detector
│   │   ├── partC_detector_matched_pivots.csv    # detector-matched placebo pivots and their features
│   │   ├── partC_detector_matched_auroc.csv     # Table R12 (per channel and pooled); → Fig. 3c
│   │   └── partC_detection_rates.csv            # Table R12b (normal detection rate, anomalous retention); → Fig. 3c
│   ├── model_cross_validation/
│   │   ├── feature_attribution_16models.csv     # original arm
│   │   ├── partA2_seq_shares_asrun.csv          # sequence re-run, test-fold early stopping (superseded protocol)
│   │   ├── partA2_seq_shares_fixed.csv          # sequence re-run, within-training-fold early stopping (reported)
│   │   ├── partA2_seq_perf.csv                  # out-of-fold AUC of both sequence re-runs
│   │   ├── family_summary.csv, performance_leaderboard.csv, bootstrap_stability.csv
│   │   └── agreement_statistics.json            # W for the original arm only
│   ├── external_validation/
│   │   ├── stage1_placebo_pooled_dataset.csv        # Table 12 pooled rows
│   │   ├── stage1_channel_level_significance.csv    # Table 12 SMD channel-level counts; → Fig. 6
│   │   ├── stage1_channel_type_stratified.csv       # exploratory FDR family (§ Stage 1)
│   │   ├── stage2_pairs.csv                         # pair-level crossing times (SMAP/MSL, SMD)
│   │   ├── partD_table13_cluster_bootstrap.csv      # Table 13 primary rows (ties excluded), cluster CIs; → Fig. 7
│   │   ├── taskD_tie_sensitivity_external_recomputed.csv   # Table 13 tie-handling sensitivity (external); → Fig. 7
│   │   ├── taskD_pair_count_check.csv               # pair counts vs. previously reported counts
│   │   ├── far_coverage_curve.csv                   # coverage on the FAR grid 0.5–20%; → Fig. 6c
│   │   ├── far_coverage_contrasts.csv               # paired contrasts vs level with cluster CIs; → Fig. 6c
│   │   ├── fixed_threshold_points.csv               # fixed |z| > 3 operating points with measured FAR; → Fig. 6c
│   │   └── window_cap_sensitivity.csv               # 150–5,000-sample sweep log
│   ├── figures/
│   │   ├── _figstyle.py                             # shared style, CSV loader with column checks, n-assertions, figure_manifest.json
│   │   ├── fig01_channel_scope.png
│   │   ├── figS_fig02_sequence_partial.png          # Fig. 2 (sequence models, three protocols)
│   │   ├── fig03_quasi_experimental.png             # panels a, b, c
│   │   ├── fig04_scm_and_heterogeneity.png          # file name kept; now the I² panel only
│   │   ├── fig05_dataset_and_precedence.png
│   │   ├── fig06_cross_dataset_placebo.png
│   │   ├── fig06b_coverage_and_pooling_bias.png
│   │   ├── fig06c_coverage_vs_far.png
│   │   └── fig07_precedence_baseline.png
│   └── tables/                            # camera-ready LaTeX tables (.tex) mirroring the CSVs above (incl. R3b, R7–R12)
│
├── results_care_v8/                       # written by care_protocol_v8.py (§11.3, Tables R7–R11)
│   ├── P1_coverage_raw.csv, P1_coverage_summary.csv          # Table R8
│   ├── P2_smallK_convergence.csv, P2_level_reversal_diagnosis.csv, P2_leave_one_channel_out.csv   # Tables R7, R7b, R7c
│   ├── P3_K_sensitivity_{raw,summary,trend_test}.csv + .png   # Table R11
│   ├── P4_retention_bias_{raw,summary}.csv, P4_real_retention_table.csv   # Table R10
│   ├── P5_care_tier_v2.csv, P6_method_selection_table.csv
│   └── SUMMARY.md
│
└── docs/
    ├── dev-log/                           # raw, unabridged research log (see note below)
    │   ├── step1_signal_estimation.md, step2_anomaly_detection.md
    │   ├── step3_causal_analysis.md, step3b_labeling_protocol_revalidation.md
    │   ├── step_ai_model_cross_validation.md, step_external_validation.md
    ├── METHODS_SUPPLEMENT.md
    └── REPRODUCIBILITY_CHECKLIST.md
```

The files in `docs/dev-log/` are the research logs kept during development, included verbatim (file names retain the original "causal" wording). `step_external_validation.md` also records the SMAP/MSL sliding-onset scheme that contaminated the pre-onset baseline for windows longer than the baseline (diagnosed by a jointly-valid-candidate check and a window-length stratification, then discarded), the 150 to 5,000-sample search-window sweep (n = 40 → 41; `diff`-first 67.5% → 65.9%, original counting) and the consistency-gate results of the re-analysis.

---

## 9. Reproducing every number

```bash
git clone https://github.com/USERNAME/opssat-ad-onset-signatures
cd opssat-ad-onset-signatures
pip install -r requirements.txt          # or: conda env create -f environment.yml
bash data/download_opssat_ad.sh
bash data/download_smap_msl.sh
bash data/download_smd.sh
make all                                  # regenerates every figure and table
```

Seeds, package versions and expected runtimes are in `docs/REPRODUCIBILITY_CHECKLIST.md`. Every figure script asserts the sample sizes of its input CSVs and writes `figure_manifest.json` (`results/figures/_figstyle.py`). `results/README.md` holds the sample-size accounting table (`partE_n_accounting.csv`) referenced by every table and figure caption. `docs/METHODS_SUPPLEMENT.md` is the extended methodological record.

---

## 10. Artifact index

**Tables (in this README) and the repository artifacts that produce them.**

| README table | Artifact ID | Source file(s) |
|---|---|---|
| Table 3 | Table 1 | `results/layer2/channel_scope_full_table.csv`, `channel_scope_final_locked.csv` |
| Table 5 | Table 8 | `results/layer2/ablation_grid.csv`, `ablation_best_combo_test_eval.csv` |
| Table 6 | R1 | `results/README.md` provenance table; replica in `ress_followup_v6.py` |
| Tables 7, 8 | R2 | `results/layer3/cluster_robust_contrasts.csv` |
| Table 9 | R2 companion | `results/layer3/cluster_robust_contrasts.csv` |
| Tables 10, 11, 12 | R7, R7c, R7b | `results_care_v8/P2_*.csv` |
| Table 13 | R12, R12b | `results/layer3/partC_detector_matched_auroc.csv`, `partC_detection_rates.csv` |
| Table 14 | 3 | `results/layer3/placebo_comparison.csv`, `triple_verification_matrix.csv` |
| Table 15 | 4 | `results/layer3/heterogeneity_summary.csv` |
| Table 16 | 12 | `results/external_validation/stage1_placebo_pooled_dataset.csv`, `stage1_channel_level_significance.csv`, `stage1_channel_type_stratified.csv` |
| Table 17 | R4 | `results/layer3/cluster_robust_contrasts.csv` and `level_prez` re-analysis |
| Table 18 | R3, R3b | `results/external_validation/far_coverage_curve.csv`, `far_coverage_contrasts.csv`, `fixed_threshold_points.csv` |
| Tables 19, 20, 21, 22 | R8, R9, R10, R11 | `results_care_v8/P1_*`, `proposition1_verification.py`, `P4_*`, `P3_*` |
| Tables 23 | 2 (sequence arm) | `results/model_cross_validation/partA2_seq_shares_*.csv`, `partA2_seq_perf.csv` |
| Tables 24, 25 | 13 | `results/external_validation/partD_table13_cluster_bootstrap.csv`, `taskD_tie_sensitivity_external_recomputed.csv`, `stage2_pairs.csv` |

**Figures.**

| Figure | File | Rendered from |
|---|---|---|
| Fig. 1 | `results/figures/fig01_channel_scope.png` | `channel_scope.csv` |
| Fig. 2 | `results/figures/figS_fig02_sequence_partial.png` | `feature_attribution_16models.csv`, `partA2_seq_shares_asrun.csv`, `partA2_seq_shares_fixed.csv`, `partA2_seq_perf.csv` |
| Fig. 3 | `results/figures/fig03_quasi_experimental.png` | (a, b) `per_channel_auroc.csv` and Table R7b constants; (c) `partC_detector_matched_auroc.csv`, `partC_detection_rates.csv` |
| Fig. 4 | `results/figures/fig04_scm_and_heterogeneity.png` | `heterogeneity_summary.csv` |
| Fig. 5 | `results/figures/fig05_dataset_and_precedence.png` | `channel_scope.csv` |
| Fig. 6 | `results/figures/fig06_cross_dataset_placebo.png` | `cluster_robust_contrasts.csv`, `stage1_channel_level_significance.csv` |
| Fig. 6b | `results/figures/fig06b_coverage_and_pooling_bias.png` | `P1_coverage_summary.csv`, `P4_retention_bias_summary.csv`, `P4_real_retention_table.csv` |
| Fig. 6c | `results/figures/fig06c_coverage_vs_far.png` | `far_coverage_curve.csv`, `far_coverage_contrasts.csv`, `fixed_threshold_points.csv` |
| Fig. 7 | `results/figures/fig07_precedence_baseline.png` | `partD_table13_cluster_bootstrap.csv`, `stage2_pairs.csv` |

Downstream artifacts consumed by Paper 2 and the DSS prototype: `results/layer2/channel_scope.json`, `results/layer3/onset_posteriors.parquet`, `results/layer3/heterogeneity_summary.csv`.

---

## 11. Data and code availability

Analysis code, data-provenance scripts, figure source scripts, development logs and the robustness re-analysis are in this repository under the MIT license. Datasets are not redistributed; the scripts in `data/` fetch them from their public releases (OPS-SAT-AD from its Zenodo record; SMAP/MSL from the Telemanom release; SMD from the OmniAnomaly release). Checksums and license notes are in `data/README.md`.

---

## 12. References

- Adams, R. P., and MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.
- Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427.
- DerSimonian, R., and Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188.
- Efron, B., and Morris, C. (1973). Stein's estimation rule and its competitors: an empirical Bayes approach. *Journal of the American Statistical Association*, 68(341), 117–130.
- Higgins, J. P. T., and Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*, 21(11), 1539–1558.
- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., and Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *Proceedings of KDD 2018*.
- Ruszczak, B., Kotowski, K., Evans, D., and Nalepa, J. (2025). The OPS-SAT benchmark for detecting anomalies in satellite telemetry. *Scientific Data*, 12, 710.
- Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., and Pei, D. (2019). Robust anomaly detection for multivariate time series through stochastic recurrent neural network. *Proceedings of KDD 2019*.
- Webb, M. D. (2014). Reworking wild bootstrap based inference for clustered errors. Queen's Economics Department Working Paper No. 1315.
- Wu, R., and Keogh, E. J. (2021). Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. *IEEE Transactions on Knowledge and Data Engineering*.

---

Machine-readable citation metadata is in `CITATION.cff`. License: MIT (see `LICENSE`).
