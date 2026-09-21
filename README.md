# opssat-ad-onset-signatures

**Derivative-Variance Signatures of Telemetry Anomaly Onsets: Cross-Dataset Evidence from OPS-SAT-AD, SMAP/MSL, and SMD**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-fully%20reproducible-brightgreen.svg)](#reproducing-every-number-in-this-readme)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](#citation)

> **Summary of findings.** We compare what changes at labeled anomaly onsets in three benchmarks: OPS-SAT-AD (9 ESA spacecraft telemetry channels, 5 in analytical scope), SMAP/MSL (82 NASA spacecraft channels) and SMD (1,064 industrial server-machine channels). In all three, the variance of the first and second differences of the signal (`diff`, `diff²`) separates anomaly onsets from same-channel placebo pivots more strongly than a level (mean) shift does. This **directional result is the same in all three datasets**, including the two (SMAP/MSL, SMD) whose onsets are ground-truth labels rather than model-estimated.
>
> Two stronger readings hold **only in OPS-SAT-AD**: (i) no detectable level excess over placebo (0/5 channels, Bonferroni-corrected) and (ii) a large anomalous-vs-normal contrast in onset-timing order (`diff` crosses threshold before `level` in 84.3% of anomalous but only 44.2% of normal-control pairs). Outside OPS-SAT-AD, `diff`-first ordering is also common in normal-control segments (SMAP/MSL 56.5%, SMD 72.7%), which is what one expects if differencing has an operator-level speed advantage on *any* local disturbance. Measured against that normal-control baseline, the anomaly-attributable increment in `diff`-first ordering is **+40.1 percentage points in OPS-SAT-AD, +11.0 pp in SMAP/MSL (not significant, n=40), and +3.9 pp in SMD**.
>
> We therefore claim an **associational, direction-level** result that travels across datasets, report a **quantified and strongly dataset-dependent** anomaly-specific increment on top of an operator baseline, and confine the strict-specificity reading to OPS-SAT-AD's instrument physics. We do not claim a universal causal mechanism or a detector that outperforms published methods.

This repository is the code, data-provenance and figure-reproduction companion to **Paper 1** of a two-paper series built primarily on OPS-SAT-AD, with independent generalization testing on SMAP/MSL and SMD. It is **self-contained and independently evaluable** and does not depend on any other repository.

| This series | Repository | Status |
|---|---|---|
| **Paper 1 (this repo)** — Cross-dataset signature analysis | `opssat-ad-onset-signatures` | ✅ independent, no upstream deps |
| Paper 2 — Risk-aware alarm design (CVaR / conformal) | [`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization) | depends on this repo's `results/` artifacts |
| DSS prototype (not a paper) | [`opssat-ad-dss`](https://github.com/USERNAME/opssat-ad-dss) | depends on both papers' artifacts |

---

## Table of Contents

1. [Evidence hierarchy and claim scope](#evidence-hierarchy-and-claim-scope)
2. [Why this repository exists](#why-this-repository-exists)
3. [Onset dependence of each result](#onset-dependence-of-each-result)
4. [Headline results](#headline-results)
5. [Repository structure](#repository-structure)
6. [Datasets](#datasets)
7. [Methodology (OPS-SAT-AD primary pipeline)](#methodology-ops-sat-ad-primary-pipeline)
   - [Layer 1 — Signal Estimation](#layer-1--signal-estimation)
   - [Layer 2 — Onset Estimation and Channel Scoping](#layer-2--onset-estimation-and-channel-scoping)
   - [Layer 3 — Signature Analysis](#layer-3--signature-analysis)
   - [Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification](#layer-3b--labeling-protocol-audit--traintest-triple-verification)
   - [Model Cross-Validation (16 architecturally diverse models)](#model-cross-validation-16-architecturally-diverse-models)
8. [External Validation / Generalization Study (SMAP/MSL, SMD)](#external-validation--generalization-study-smapmsl-smd)
9. [Analytical baseline: why differencing may respond faster](#analytical-baseline-why-differencing-may-respond-faster)
10. [Practical implications for alarm design](#practical-implications-for-alarm-design)
11. [Scope of the inferential claims](#scope-of-the-inferential-claims)
12. [Figures](#figures)
13. [Results tables](#results-tables)
14. [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
15. [Threats to validity and how each was addressed](#threats-to-validity-and-how-each-was-addressed)
16. [Limitations](#limitations)
17. [Planned extensions (not in this release)](#planned-extensions-not-in-this-release)
18. [Citation](#citation)
19. [License](#license)

---

## Evidence hierarchy and claim scope

The paper's claims are graded by how far the evidence travels. Readers (and reviewers) should treat the tiers as separate claims with separate support.

| Tier | Claim | Where it is supported | Status |
|---|---|---|---|
| **1 — Established (direction)** | `diff`/`diff²` variance features separate labeled anomaly onsets from same-channel placebo pivots more strongly than `level`. | OPS-SAT-AD channel-level placebo test (Table 3); SMAP/MSL and SMD dataset-pooled tests with **ground-truth onsets** (Table 12, Fig. 6) | Same direction in 3/3 datasets |
| **2 — Quantified (baseline decomposition)** | `diff`-first onset ordering has a substantial operator-level baseline (visible in normal controls); the anomaly-attributable increment over that baseline is +40.1 pp (OPS-SAT-AD), +11.0 pp (SMAP/MSL, n.s.), +3.9 pp (SMD). | Normal-segment controls in all three datasets (Table 13, Fig. 7) | Same sign in 3/3; magnitude strongly dataset-dependent |
| **3 — Conditional on OPS-SAT-AD** | No detectable `level` excess over placebo (0/5 channels); ordering contrast large enough to reverse in normal controls; stronger `diff`/`diff²` effects in `float_noise_suspect` (magnetometer) than `quantized` (photodiode) channels. | Layer 3 of the OPS-SAT-AD pipeline; train/test triple verification; 16-model attribution | Not observed in SMAP/MSL or SMD; moderator confounded with sensor family (see [Limitations](#limitations)) |
| **4 — Not claimed** | A universal causal mechanism linking onsets to derivative-statistic surges; superiority of a `diff`-based detector over published detectors; formal (do-calculus) identification. | — | Out of scope |

**Scope of detection results.** Layer 2 detection metrics (Table 1) are reported only to justify channel scoping and to lock BOCPD hyperparameters. This paper does not present a detector benchmark, and a comparison against published OPS-SAT-AD detectors is deliberately out of scope; the contribution is a signature-identification finding, not a state-of-the-art detection claim.

---

## Why this repository exists

Change-point and anomaly-detection pipelines for spacecraft and industrial telemetry commonly treat a **level shift** (a change in the mean of the raw signal) as the default anomaly signature: it is the easiest thing to visualize, the easiest thing to threshold, and the default target of most textbook change-point models. This repository documents a systematic investigation, beginning with the **OPS-SAT-AD** benchmark (ESA OPS-SAT mission, 9 telemetry channels, 2,123 expert-labeled univariate segments), of whether level shifts are in fact the operative signature, or whether **variance surges in the first and second differences (`diff`, `diff²`)** carry more anomaly-relevant information.

This is not a claim we set out to prove. The project's early working hypothesis, stated before any hypothesis testing began, was the naive `level → diff → diff²` ordering: a level shift is the primary event and derivative statistics are downstream artifacts of it. Section 3.2's pooled Wilcoxon test was run to confirm this ordering and instead found the opposite pattern: `diff`/`diff²` precede `level` in 82–84% of paired segments within OPS-SAT-AD. Every subsequent analysis was designed to stress-test whether that reversal was real or an artifact, and, subsequently, whether it holds outside OPS-SAT-AD.

Because a single-pipeline, single-dataset finding is easy to dismiss, the analysis proceeded along six lines of attack, each targeting a different class of confound. We state explicitly what each line does and does not control:

1. **Is `level`'s apparent significance just a weaker null?** → same-channel quasi-experimental placebo-pool comparison (§ Layer 3.7).
2. **Is the pattern a side effect of how segments were manually cut?** → external audit of the OPS-SAT-AD labeling protocol (§ Layer 3b). *Result: undecidable; not excluded.*
3. **Is the result an artifact of using the full dataset rather than a train/test split?** → independent train-only and test-only reproduction (§ Layer 3b).
4. **Is the attribution ranking an artifact of a single learner's inductive bias?** → 16-model cross-validation over two data representations (§ Model Cross-Validation). *This controls the **classifier** side only. It does not control how onsets were estimated, because all 16 models are trained on labels derived from the same BOCPD onsets.*
5. **Is the result an artifact of the onset estimator (BOCPD on Kalman innovations)?** → replication on SMAP/MSL and SMD, whose onsets are ground-truth labels (§ External Validation). *This is the only line that controls onset-estimation dependence.*
6. **Is the result an artifact of OPS-SAT-AD's instrument physics (quantization, sensor noise regime)?** → the same external replication. *Result: direction confirmed; strict-specificity and reversal claims not confirmed.*

Lines 1–3 and 6 converge within OPS-SAT-AD. Lines 5–6 confirm the directional claim outside OPS-SAT-AD but not the strongest readings. We treat this as an informative statement about the scope of the result rather than a failure to be minimized.

This repository packages:

- The **signal-estimation → onset-estimation → signature-analysis** pipeline for OPS-SAT-AD (Layers 1–3).
- The **labeling-protocol audit** and **train/test/bootstrap triple re-verification** (Layer 3b).
- The **16-model cross-validation** benchmark and its agreement statistics.
- An **external-validation pipeline** applying the placebo-pool and temporal-precedence logic, adapted to each dataset's label structure, to SMAP/MSL and SMD.
- **Every figure regenerated from its exact source script** (`results/figures/*.py`), so every number a reader sees traces to code.
- **Raw development logs** (`docs/dev-log/`): the unabridged analytical trail, including negative results, discarded ablations, and methodological corrections (train/test leakage checks, SHAP return-shape bugs, cuDNN backward-mode fixes, and the external-validation debugging trail). These are provided for transparency and are not a substitute for the Methods text in this README and `docs/METHODS_SUPPLEMENT.md`.

---

## Onset dependence of each result

Which results depend on a model-estimated onset, and which do not, determines how each should be weighted.

| Result | Onset source | Depends on BOCPD onset estimate? |
|---|---|---|
| OPS-SAT-AD Layer 3 (§3.1–3.7), triple verification, 16-model cross-validation | BOCPD on Kalman-filter innovations; canonical onset = median of 200 Monte Carlo draws | **Yes** |
| SMAP/MSL Stage 1–2 | Ground-truth onset indices (`labeled_anomalies.csv`) | **No** |
| SMD Stage 1–2 | Ground-truth timestep-level 0/1 labels | **No** |

**Known direction of possible bias in the OPS-SAT-AD onsets.** A local-linear-trend Kalman filter absorbs a persistent mean shift into its state within a few samples, leaving a short-lived transient in the innovations. BOCPD run on those innovations may therefore preferentially select onsets where a transient variance change is present, and only 178/386 anomalous segments (46.1%) yield a scoreable window. In addition, per the §3.7 design, placebo pivots are resampled positions on normal segments, position-matched to anomalous onsets, whereas anomalous onsets are detector-selected; to the extent that is true, the placebo comparison inherits a selection asymmetry. We do not quantify this bias in this release.

**Consequence for how the results are weighted.** The strong-form OPS-SAT-AD readings (Tier 3) are conditional on the onset estimator and on OPS-SAT-AD's instrument physics. The onset-independent evidence (SMAP/MSL, SMD) supports only the directional claim (Tier 1) and the baseline decomposition (Tier 2). This is why the directional claim, not the strict-specificity claim, is the load-bearing result.

---

## Headline results

| Claim | OPS-SAT-AD evidence | Holds in SMAP/MSL, SMD? |
|---|---|---|
| Channel scope: 9 → 5 channels, identical across 3 independent derivations | Fig. 1, Table 1 | N/A (OPS-SAT-AD-specific scoping) |
| `diff`/`diff²` separate onsets from placebo more strongly than `level` (**direction**) | Fig. 3a, Table 3 | **Yes — 3/3 datasets** (Table 12, Fig. 6) |
| No detectable `level` excess over placebo (0/5 channels; p_bonf capped at 1.000) | Fig. 3a, Table 3 | **No** — `level` is weakly but significantly above placebo in SMAP/MSL (p_fdr=0.031) and SMD (p_fdr=8×10⁻³⁸, nominal; see independence caveat) (Table 12, Fig. 6) |
| `diff` precedes `level` in onset timing; **anomalous-vs-normal contrast** | 84.3% (anomalous) vs 44.2% (normal): **reversal** (Fig. 5b, Table 10) | **Same sign, much smaller**: +11.0 pp (SMAP/MSL, n.s.), +3.9 pp (SMD); **no reversal** (Table 13, Fig. 7) |
| `float_noise_suspect` (magnetometer) shows stronger `diff`/`diff²` effects than `quantized` (photodiode) | Fig. 4a (confounded with sensor family) | Undecided in SMAP/MSL (1/82 channels of this type); not confirmed in SMD |
| Placebo-pool result replicates on full / train-only / test-only splits | Fig. 3b (12/13 decidable comparisons agree) | Not re-tested externally |
| 0/16 models rank `level` as top feature; 15/16 give `level` less than the 1/3 uniform share | Fig. 2, Tables 2, 6, 7 | Not re-tested externally |
| No documented quantitative segment-cutting rule in the OPS-SAT-AD labeling protocol | — | N/A |
| `level`'s high within-segment significance despite losing the precedence race is reconciled via transient-vs-persistent profile classification | §3.3 | Not re-tested externally |

**Reading this table.** Row 2 is the load-bearing result. Row 4 is reported as a decomposition (operator baseline plus anomaly-attributable increment) rather than as a binary causal-vs-not verdict. Rows 3 and 5 are OPS-SAT-AD-conditional and are retained because the boundary is itself informative (§ External Validation, § Limitations).

---

## Repository structure

```
opssat-ad-onset-signatures/
│
├── README.md                              # this file — full methodology + inline figures
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── Makefile                               # `make all` reproduces every figure/table below
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
│   └── run.py                             # entry point: `python -m layer1_signal_estimation.run`
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
│   ├── temporal_precedence_test.py        # pooled Wilcoxon signed-rank, level vs diff vs diff² (Fig. 5b)
│   ├── temporal_profile_classification.py # transient vs. persistent post-onset trajectory classification
│   ├── early_window_sensitivity.py        # 10-point vs. full-window Welch-test comparison
│   ├── quasi_experimental_placebo.py      # Mann–Whitney U vs. same-channel placebo pool (Fig. 3a)
│   ├── meta_analysis_random_effects.py    # DerSimonian–Laird random-effects meta-analysis + I² (Fig. 4b)
│   ├── loo_sensitivity.py                 # leave-one-channel-out sensitivity analysis
│   ├── scm_skeleton.py                    # descriptive structural working model, Path A vs. Path B (Fig. 4a)
│   ├── supplementary_diagnostics_A_D.py   # cross-correlation, CMH test, normal-segment controls, lag resolution
│   ├── labeling_protocol_audit.py         # external audit of the OPS-SAT-AD labeling protocol
│   ├── train_test_triple_reverification.py# train-only / test-only reproduction of the placebo-pool test
│   ├── tests/
│   └── run.py
│
├── model_cross_validation/
│   ├── models/                            # 16 model wrappers (+1 excluded, see § below)
│   │   ├── linear.py                      # LogReg (L1, L2)
│   │   ├── probabilistic.py               # GaussianNB
│   │   ├── kernel.py                      # SVM (RBF)
│   │   ├── instance_based.py              # kNN
│   │   ├── tree_ensembles.py              # RandomForest, ExtraTrees, GradBoost, XGBoost, LightGBM
│   │   ├── shallow_mlp.py                 # MLP_shallow — EXCLUDED (AUC=0.403, below chance; failed fit)
│   │   ├── conv_sequence.py               # CNN1D, TCN
│   │   ├── recurrent_sequence.py          # BiLSTM, BiGRU
│   │   ├── attention_sequence.py          # TinyTransformer
│   │   └── ssm_sequence.py                # LightMamba (pure-PyTorch selective state-space substitute)
│   ├── representations/
│   │   ├── tabular_summary_features.py    # 10 models: engineered summary-statistic features
│   │   └── raw_sequence_features.py       # 6 models: raw 3-channel windowed time series
│   ├── attribution/
│   │   ├── shap_permutation.py            # tabular-model attribution (SHAP + permutation fallback)
│   │   └── integrated_gradients.py        # sequence-model attribution (Captum / manual IG fallback)
│   ├── agreement_statistics.py            # Kendall's W, binomial test, bootstrap stability (Fig. 2b)
│   ├── tests/
│   └── run_all.py                         # entry point: `python -m model_cross_validation.run_all`
│
├── external_validation/
│   ├── stage0_preprocessing.py            # command-column exclusion, channel re-typing, triviality filter
│   ├── stage1_placebo_pooled.py           # dataset-pooled and channel-type-stratified placebo tests (FDR) (Fig. 6)
│   ├── stage2_temporal_precedence.py      # ground-truth-onset precedence + normal-segment control (Fig. 7)
│   ├── window_cap_sensitivity.py          # 150–5,000-sample search-window sensitivity sweep
│   └── run_external_validation.py         # entry point
│
├── results/
│   ├── README.md                          # index of every artifact below + provenance table
│   ├── layer1/
│   │   ├── noise_estimates.csv
│   │   └── channel_classification.json
│   ├── layer2/
│   │   ├── channel_scope.json                  # → consumed by opssat-ad-risk-optimization
│   │   ├── channel_scope_full_table.csv
│   │   ├── channel_scope_final_locked.csv
│   │   ├── ablation_grid.csv
│   │   ├── ablation_best_combo_test_eval.csv
│   │   ├── bootstrap_ci.csv
│   │   ├── bootstrap_ci_test_only.csv
│   │   ├── train_fit_test_eval.csv
│   │   └── full_vs_test_ci_comparison.csv
│   ├── layer3/
│   │   ├── onset_posteriors.parquet             # → consumed by opssat-ad-risk-optimization & opssat-ad-dss
│   │   ├── placebo_comparison.csv
│   │   ├── temporal_precedence.csv
│   │   ├── temporal_precedence_normal_control.csv
│   │   ├── temporal_precedence_group_comparison.csv
│   │   ├── temporal_precedence_lag_resolution.csv
│   │   ├── heterogeneity_summary.csv            # → consumed by opssat-ad-dss
│   │   └── triple_verification_matrix.csv
│   ├── model_cross_validation/
│   │   ├── feature_attribution_16models.csv
│   │   ├── family_summary.csv
│   │   ├── performance_leaderboard.csv
│   │   ├── bootstrap_stability.csv
│   │   └── agreement_statistics.json
│   ├── external_validation/
│   │   ├── stage1_placebo_pooled_dataset.csv        # Table 12 pooled rows; → Fig. 6a
│   │   ├── stage1_channel_level_significance.csv    # Table 12 SMD channel-level counts; → Fig. 6b
│   │   ├── stage1_channel_type_stratified.csv       # exploratory FDR family (§ Stage 1)
│   │   ├── stage2_temporal_precedence.csv           # Table 13 precedence rows; → Fig. 7a
│   │   ├── stage2_baseline_decomposition.csv        # Table 13 increment rows; → Fig. 7b
│   │   └── window_cap_sensitivity.csv               # 150–5,000-sample sweep log
│   ├── figures/
│   │   ├── fig01_channel_scope.png
│   │   ├── fig01_channel_scope.py
│   │   ├── fig02_model_cross_validation.png
│   │   ├── fig02_model_cross_validation.py
│   │   ├── fig03_quasi_experimental.png
│   │   ├── fig03_quasi_experimental.py
│   │   ├── fig04_scm_and_heterogeneity.png
│   │   ├── fig04_scm_and_heterogeneity.py
│   │   ├── fig05_dataset_and_precedence.png
│   │   ├── fig05_dataset_and_precedence.py
│   │   ├── fig06_cross_dataset_placebo.png
│   │   ├── fig06_cross_dataset_placebo.py
│   │   ├── fig07_precedence_baseline.png
│   │   └── fig07_precedence_baseline.py
│   └── tables/                            # camera-ready LaTeX tables (.tex) mirroring the CSVs above
│
└── docs/
    ├── dev-log/                           # raw, unabridged research log (see note below)
    │   ├── step1_signal_estimation.md
    │   ├── step2_anomaly_detection.md
    │   ├── step3_causal_analysis.md
    │   ├── step3b_labeling_protocol_revalidation.md
    │   ├── step_ai_model_cross_validation.md
    │   └── step_external_validation.md
    ├── METHODS_SUPPLEMENT.md               # manuscript-ready extended methods (mirrors README methodology)
    └── REPRODUCIBILITY_CHECKLIST.md
```

> **Note on `docs/dev-log/`.** These files are the original conversational research logs kept during development, included **verbatim** for analytical transparency (file names retain the original "causal" wording). `step_external_validation.md` documents a methodological correction made mid-analysis: an early version of the SMAP/MSL temporal-precedence test used a sliding-window onset-refinement scheme that contaminated the pre-onset baseline for windows longer than the baseline length. This was diagnosed, the affected results were discarded, and the analysis was re-run with ground-truth onsets and a baseline-safe adaptive search window (§ External Validation). It is disclosed here per the reproducibility standard applied throughout this repository.

---

## Datasets

### OPS-SAT-AD (primary)

European Space Agency's OPS-SAT experimental nanosatellite telemetry anomaly-detection benchmark (Ruszczak, Kotowski, Evans & Nalepa, *Scientific Data* 12:710, 2025).

- **9 channels**: 3 magnetometer channels (`CADC0872`, `CADC0873`, `CADC0874`) + 6 photodiode channels (`CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894`).
- **2,123 univariate segments**, each expert-labeled *nominal* or *anomalous*. Labels are at segment level; onset position within a segment is not given and is estimated via BOCPD (§ Layer 2).
- Per-channel segment counts and anomaly prevalence vary sharply (0% to 78.6%); see Fig. 1, Fig. 5a and Table 1.
- An official train/test partition (`train ∈ {0,1}`, stratified by anomaly rate, ≈75%/25%) is used throughout for train-fit/test-evaluate verification.
- Segment boundaries were determined by **manual expert annotation** using ESA's OXI visualization tool; no quantitative cutting rule is documented in the public materials (§ Layer 3b).
- Not redistributed; `data/download_opssat_ad.sh` fetches it from its Zenodo record.

### SMAP/MSL (external validation)

NASA Soil Moisture Active Passive (SMAP) and Mars Science Laboratory (MSL) spacecraft telemetry, released with the Telemanom benchmark (Hundman et al., KDD 2018).

- **82 channels** (55 SMAP + 27 MSL), each a long continuous series with **ground-truth onset/offset indices** per anomaly window (`labeled_anomalies.csv`), a structurally different label format from OPS-SAT-AD's whole-segment labels.
- Each `.npy` file has multiple columns; only the first (telemetry) column is used. The remaining columns are one-hot-encoded command context and are excluded from all `diff`/`diff²` computation (Stage 0).
- Anomaly window lengths are highly variable (median 120 samples, mean 616, max 4,217), unlike OPS-SAT-AD's short, uniformly windowed segments; this drove a methodological adaptation (Stage 2).
- A triviality filter motivated by Wu & Keogh's (2021) critique of SMAP/MSL and similar benchmarks excludes extreme, visually obvious point outliers from the primary test (Stage 0).
- Not redistributed; `data/download_smap_msl.sh` fetches the public release.

### SMD (Server Machine Dataset, external validation)

Industrial server telemetry from the OmniAnomaly benchmark (Su et al., KDD 2019).

- **28 machines × 38 dimensions = 1,064 channels**, each with **timestep-level (0/1) ground-truth anomaly labels**; 1,038 of the 1,064 channels enter the channel-level counts reported in Stage 1.
- Included as a domain-generalization check (technological systems beyond spacecraft) rather than a direct replication of the spacecraft-telemetry claim. SMD is *supporting*, not primary, generalization evidence, given the domain shift.
- Not redistributed; `data/download_smd.sh` fetches the public release.

---

## Methodology (OPS-SAT-AD primary pipeline)

### Layer 1 — Signal Estimation

**Goal**: obtain a denoised, well-calibrated state estimate for each channel before any change-point logic is applied, so that downstream onset estimation is not confounded by channel-specific noise characteristics.

**1.1 Channel typing.** Each channel is classified from the empirical distribution of first differences (Δ) computed within nominal (`anomaly = 0`) segments only. Thresholds were set from inspection of this dataset's empirical diagnostics (not derived from first principles) and were **re-fitted, not reused,** on SMAP/MSL and SMD:

- **`quantized`** if the fraction of exact-zero Δ is ≥ 0.10 (quantization step estimated as the 25th percentile of non-zero |Δ|).
- **`float_noise_suspect`** if not quantized and the smallest non-zero |Δ| is < 1 × 10⁻⁶.
- **`continuous`** otherwise.

| Channel | Type | frac_zero_diff | Quantization step |
|---|---|---|---|
| CADC0872 | float_noise_suspect | 0.053 | n/a |
| CADC0873 | float_noise_suspect | 0.052 | n/a |
| CADC0874 | float_noise_suspect | 0.044 | n/a |
| CADC0884 | quantized | 0.183 | ≈0.0144 |
| CADC0886 | quantized | 0.371 | ≈0.0274 |
| CADC0888 | quantized | 0.354 | ≈0.0144 |
| CADC0890 | continuous | 0.083 | n/a |
| CADC0892 | quantized | 0.416 | ≈0.0054 |
| CADC0894 | quantized | 0.583 | ≈0.0026 |

Channel type is fully aliased with sensor family in this dataset: all three `float_noise_suspect` channels are magnetometers and all `quantized` channels are photodiodes (the single `continuous` channel, CADC0890, is a photodiode excluded from the analytical scope).

**1.2 Noise estimation.** `r = Var(Δ)/2`, `q = Var(Δ²)/6`, from pooled nominal-segment differences. A Gaussian-mixture alternative was evaluated within the 2×2 ablation (Table 8) and did not improve train macro-MCC (0.275 vs. 0.308 with forgetting; 0.199 vs. 0.238 without), so the simple estimator was retained.

**1.3 Quantization floor.** For `quantized` channels, `step²/12` is added to the observation-noise variance.

**1.4 Hierarchical Bayesian shrinkage.** Efron–Morris-type empirical-Bayes shrinkage stabilizes channels with very few nominal points (CADC0886: n=8; CADC0890: n=3). Shrunk estimates are listed in Table 11.

**1.5 State estimation.** A local-linear-trend Kalman filter is fit per channel using the shrunk `r`, `q`. Standardized innovations `z_t = (y_t − ŷ_t)/√S_t` are the sole input to Layer 2.

---

### Layer 2 — Onset Estimation and Channel Scoping

**2.1 Change-point model.** Bayesian Online Change-Point Detection (BOCPD; Adams & MacKay 2007) on `z_t`, Normal-Inverse-Gamma conjugate predictive model, hazard rate `1/250`, with a **forgetting** extension (`κ_max = 40`) preventing over-confidence during long nominal stretches.

**2.2 Full 2×2 ablation.**

| Combination | Train macro-MCC (6 scoring channels) |
|---|---|
| mixture=False, forgetting=True (**locked**) | **0.308** |
| mixture=True, forgetting=True | 0.275 |
| mixture=False, forgetting=False | 0.238 |
| mixture=True, forgetting=False | 0.199 |

Train-only reselection independently reproduces the same locked combination.

**2.3 Channel scoping.** MCC and Youden's *J* at locked hyperparameters, with exclusion criteria applied in order: structural (zero anomalous segments), underpowered (fewer than 5 anomalous or 5 normal segments), chance-level (MCC ≤ 0). See **Table 1**. These metrics serve scoping only (see [Evidence hierarchy](#evidence-hierarchy-and-claim-scope)).

**2.4 Triple verification.** The 5-channel scope was independently re-derived via bootstrap CI, train-fit/test-eval, and test-only bootstrap CI; all three slices agree (Fig. 1b).

**2.5 Problem-channel diagnosis (CADC0890/0892/0894).** Hypothesis B (BOCPD forgetting deficiency) was rejected; Hypothesis A (noise-model deficiency) is supported for CADC0890/CADC0894 under the mixture estimator; CADC0892 is explained by neither, since extreme steady-state kurtosis (63.32) drives spurious resets unrelated to either mechanism.

---

### Layer 3 — Signature Analysis

**Goal**: determine what changes at an OPS-SAT-AD anomaly onset, and test whether the answer is an artifact of labeling, channel choice, or data-split leakage. All Layer 3 analyses operate on the 5 scoped channels and on **BOCPD-estimated onsets** (see [Onset dependence](#onset-dependence-of-each-result)).

**3.1 Onset-uncertainty propagation (Monte Carlo).** 200 onset candidates per segment (seed=42); pre/post comparisons for `level` (Cohen's *d*) and for `diff`, `diff²` (log-variance-ratio proxy on squared values). 178/386 anomalous segments (46.1%) yield a scoreable window; 99.4% fall in the high-reliability tier. The scoreable subset is a selection and results are conditional on it.

Within-segment, `level` is significant in 54–92% of segments per channel, while `diff`/`diff²` are significant in only 2–76%. Taken alone this suggests `level` is more important, the opposite of the precedence-test conclusion below. This is reconciled in §3.3.

**3.2 Temporal precedence test.** The canonical onset (median of 200 MC draws) defines the pre/post split; the first post-onset |z|>3 crossing time is recorded per series.

| Comparison | n pairs | *p* (pooled Wilcoxon) | Result |
|---|---|---|---|
| `level` vs. `diff` | 70 | 1.4×10⁻⁵ | `diff` precedes `level` in **84.3%** of pairs |
| `level` vs. `diff²` | 70 | 3.2×10⁻⁵ | `diff²` precedes `level` in **82.9%** of pairs |
| `diff` vs. `diff²` | 161 | 0.368 (n.s.) | no reliable ordering by the signed-rank test; not used for any claim (see reporting note under Table 10) |

This is inconsistent with the initial working hypothesis `level → diff → diff²`.

**3.3 Reconciling §3.1 and §3.2.**
- **(a) Temporal-profile classification**: `diff` is transient in 98% of segments, `diff²` in 94%, and `level` is a near-even mix (50%/45%).
- **(b) Early-window sensitivity**: `level`'s significance rate drops 0.864→0.633 in the early window, a larger relative loss than `diff`/`diff²`.
- **(c) Normal-segment control.** The test was repeated on normal segments with resampled pivots (mean position ratio ≈0.569). Within OPS-SAT-AD the ordering **reverses**: `level` precedes `diff` in 55.8% of normal pairs (vs. 15.7% anomalous) and precedes `diff²` in 74.3% (vs. 17.1%). Read as a baseline decomposition, the normal-control fraction estimates the operator-level baseline and the anomalous-minus-normal difference estimates the anomaly-attributable increment (+40.1 pp for `level` vs. `diff`). This reversal does not replicate outside OPS-SAT-AD (§ External Validation, Stage 2, Fig. 7).

**3.4 Formal group comparison and diff-vs-diff² tie resolution.** Two-proportion *z*/Fisher exact tests and a channel-stratified CMH test confirm the orderings after stratification (CMH p≤4.33×10⁻¹⁵). For diff vs. diff², the cross-correlation lag in anomalous segments differs from zero (p=0.00094) but anomalous-vs-normal lag distributions do not differ (p=0.906), so no reliable diff/diff² ordering is claimed in either regime.

**3.5 Random-effects meta-analysis (DerSimonian–Laird, k=5 channels).** *I²* ranges 65.7%–97.2% (Table 4), arguing against a single channel-agnostic pooled effect size. With k=5, *I²* and Q are themselves imprecise and are reported descriptively.

**3.6 Leave-one-out (LOO) decomposition.** Removing CADC0872 eliminates all `level`-|*d*| heterogeneity; removing CADC0874 eliminates all `diff²`-|*d*| heterogeneity; removing CADC0894 nearly halves the pooled `diff`/`diff²` significance fraction, without evidence of a categorically separate population.

**3.7 Quasi-experimental placebo-pool comparison.** Each anomalous segment's canonical-onset effect is compared, via one-sided Mann–Whitney *U*, against a placebo pool of resampled pivots on normal segments (15 tests, Bonferroni-corrected).

*How to read the statistics.*
- Each feature is tested with a rank test on its **own** statistic (|*d*| for `level`; log-variance ratio for `diff`/`diff²`). The tests answer "is this feature's onset effect distinguishable from placebo?" for each feature separately. The p-values are **not** effect-size comparisons across features, because the statistics are on different scales.
- `level` p_bonf is reported as 1.000 in all five channels. This is the Bonferroni cap (raw p × 15 ≥ 1, i.e. raw p ≳ 0.067), not an estimate of equality. The correct reading is **no detectable excess over placebo**; no equivalence test was run, so "indistinguishable" is a statement about failure to reject, not demonstrated equivalence.

*Result.* Within OPS-SAT-AD: `level` shows no detectable excess over placebo in **all 5 channels**; `diff`/`diff²` are significant in **all 5 channels** (Table 3). `diff`/`diff²` strength is markedly higher in the three `float_noise_suspect` magnetometer channels (0.78–0.89 significant-segment fraction) than in the two `quantized` photodiode channels (0.12–0.42); this contrast is confounded with sensor family (see [Limitations](#limitations)).

**Interpretive consequence (OPS-SAT-AD-scoped).** Within OPS-SAT-AD, the `level` signal that looked significant within-segment (§3.1) is consistent with anomaly-non-specific channel drift, while the `diff`/`diff²` variance surge is not reproduced in the placebo pool. This reading is conditional on the BOCPD onset estimator and on OPS-SAT-AD's instrument physics; § External Validation reports the weaker, cross-dataset-consistent version.

**Structural working model (Fig. 4a, "Path A"; descriptive, OPS-SAT-AD-scoped)**:

```
Onset (BOCPD-estimated change-point)
      │
      ▼
Diff/Diff² variance surge (transient, onset-localized; stronger in float_noise_suspect
                            (872/873/874) than quantized (888/894) — confounded with sensor family)
      │
      ▼  (weak / dataset-conditional — see § External Validation)
Level shift (persistent in ~50% of segments; no detectable excess over placebo in
             OPS-SAT-AD, but detectable in SMAP/MSL and SMD)
```

The arrows describe temporal and structural ordering used to organize the analysis. They are asserted from signal-processing structure and checked against data for consistency; they are not identified causal edges (§ Scope of the inferential claims).

---

### Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification

**3b.1 Labeling-protocol audit.** The §3.3(c) normal-segment control found anomalous-segment onsets skewed toward the back half of their segment (mean position ratio ≈0.569). No documented quantitative segment-cutting rule was found in the primary publication, its SoftwareX companion, or either preprint; segments were cut manually via ESA's OXI tool. The finding is **undecidable**: because boundary placement was subjective, the skew more plausibly reflects idiosyncratic manual judgment than a systematic labeling artifact, but this cannot be excluded.

**3b.2 Train-only and test-only reproduction of §3.7.**

| Slice | Coverage | Result |
|---|---|---|
| Train-only | 178,504/240,979 rows (74.1%); 126 anomalous segments | 15/15 (channel×feature) significance calls match full-data exactly |
| Test-only | 62,475/240,979 rows (25.9%); 51 anomalous segments | 12/13 decidable calls match; 1 borderline (CADC0888 `diff`, p=0.051, monotone power loss); 2 undecidable (CADC0894, n=4<5) |

<!-- VERIFY before submission: 126 (train) + 51 (test) = 177 anomalous segments, versus 178 scoreable anomalous segments in the full-data analysis (§3.1). Confirm whether one segment lacks a train/test flag or is dropped by a per-slice filter, and state the reason here. -->

**12 of 13 decidable comparisons agree exactly** across full/train/test slices (Fig. 3b). The single disagreement's p-value rises monotonically as the sample shrinks rather than reversing direction, consistent with a power artifact.

---

### Model Cross-Validation (16 architecturally diverse models)

**Goal**: test whether the feature-importance ranking (`diff`/`diff²` above `level`) depends on a particular learner's inductive bias.

**What this controls and what it does not.** All models are trained on the same binary problem, whose labels come from the BOCPD-estimated canonical onsets (§3.7 design). The check therefore controls **classifier-side** bias only. It does not control onset-estimation bias (see [Onset dependence](#onset-dependence-of-each-result)), and the models are not statistically independent because they share data and labels.

**Task and labels.** Canonical-onset anomalous segments vs. resampled placebo pivots: 4,996 tabular rows (168 positive / 4,828 negative) and 2,733 sequence-window rows. Placebo pivots are multiple per normal segment, so out-of-fold AUCs are used **only to order models**, not as performance claims; the quantity of interest is the attribution share.

**Two data representations.**
- *Tabular* (10 models): the three engineered features of §3.7. Attribution: SHAP / permutation importance.
- *Raw sequence* (6 models): per-window z-normalized 3-channel time series with no human-engineered summary statistic. Attribution: Integrated Gradients.

Attribution shares from different methods (SHAP/permutation vs. Integrated Gradients) are normalized within each model and are not strictly comparable in magnitude across the two representations.

**Model zoo — 16 models, 9 coarse families (13 finer-grained groupings):**

| Coarse family | Models (finer grouping) | Representation |
|---|---|---|
| Linear | LogReg (L2); LogReg (L1, sparse) | tabular |
| Probabilistic | GaussianNB | tabular |
| Instance-based | kNN | tabular |
| Kernel | SVM (RBF) | tabular |
| Tree ensemble | RandomForest (bagging); ExtraTrees (bagging, extra); GradBoost, XGBoost, LightGBM (boosting) | tabular |
| Convolutional | CNN1D; TCN (dilated causal) | sequence |
| Recurrent | BiLSTM, BiGRU | sequence |
| Attention | TinyTransformer | sequence |
| State-space (SSM) | LightMamba (pure-PyTorch substitute) | sequence |

A 17th model (MLP_shallow) was excluded after AUC=0.403 (below chance, failed optimization). Headline statistics are over **16 models**.

**Result.**
- **Zero of 16 models rank `level` as the top feature** (binomial p=.0015, i.e. (2/3)¹⁶ under a uniform-ranking null; nominal, because models are not independent). The result holds separately within the tabular group (0/10) and the sequence group (0/6).
- **15 of 16 models give `level` less than the 1/3 uniform share**; the exception is LightMamba (0.428). Equivalently, the combined `diff`+`diff²` share exceeds its 2/3 uniform expectation in 15/16 models.
- Kendall's *W* = 0.609 (χ²=19.50, df=2, p<.001) measures concordance of the feature ranking across models; it is read descriptively for the same non-independence reason.
- Bootstrap stability (200 resamples): 100% of resamples favor `diff`+`diff²` over `level` for RandomForest and LogisticRegression.
- The State-space family has the narrowest margin (`level`=0.428 vs. `diff`=0.441), attributed to the lightweight non-CUDA substitute block used here.

Because the design has only three features, "`diff`+`diff²` > `level`" is expected at 2:1 under a uniform-share null; the informative quantities are the top-rank count and the size of the `level` share, which are the statistics reported above.

**Scope note.** This check was run only on OPS-SAT-AD. A reduced replication on SMAP/MSL or SMD is listed under [Planned extensions](#planned-extensions-not-in-this-release).

---

## External Validation / Generalization Study (SMAP/MSL, SMD)

### Motivation and positioning

The external datasets provide the only **onset-independent** evidence in this paper: their onsets are ground-truth labels, not BOCPD estimates, and the placebo pivots are compared against human-labeled onsets rather than detector-selected ones. We re-ran the two most decisive OPS-SAT-AD tests, the placebo-pool comparison (§3.7) and the temporal-precedence test with normal-segment control (§3.2–3.3c), on SMAP/MSL and SMD. This is a confirmatory replication with the directional hypothesis (`diff`/`diff²` > `level`) fixed in advance of the external runs, not a re-run of the full OPS-SAT-AD pipeline; the 16-model cross-validation and the labeling-protocol audit were not repeated externally.

### Stage 0 — Adaptations required by the label formats

Unlike OPS-SAT-AD's whole-segment labels, SMAP/MSL and SMD provide **ground-truth onset locations** (window start/end indices, or timestep-level 0/1 labels). This removes the need for the Monte Carlo onset-uncertainty propagation of §3.1 and eliminates one source of onset-location uncertainty. Five dataset-specific adaptations are documented in `docs/dev-log/step_external_validation.md`:

- **SMAP/MSL command-column exclusion.** `diff`/`diff²` are computed only on the first (telemetry) column.
- **Channel re-typing.** The quantized / float_noise_suspect / continuous thresholds were re-fitted on each dataset's own nominal-segment difference distribution.
- **Multiple-comparisons correction.** Per-channel Bonferroni (used for OPS-SAT-AD's 15 tests) is inappropriately conservative for 82 and 1,064 channels. We use Benjamini–Hochberg FDR, with **confirmatory** (dataset-pooled, 3 tests) and **exploratory** (channel-type-stratified, up to 9 tests) families corrected separately, since the latter is nested within and not independent of the former.
- **Triviality filtering.** Following Wu & Keogh's (2021) critique, anomaly windows whose values exceed 20 nominal standard deviations are excluded from the primary test (`is_trivial_anomaly`).
- **Underpowered per-channel testing.** SMAP/MSL has a median of 1 anomaly window per channel (max 3), so per-channel Mann–Whitney tests are structurally underpowered (0/66 channels reached the n≥5 minimum in an initial attempt). Dataset-pooled and channel-type-stratified tests are therefore the primary external result, with per-channel counts reported for SMD.

### Stage 1 — Placebo-pool comparison (external replication of §3.7; Fig. 6)

**Table 12 — Placebo-pool comparison across datasets**

| Dataset | Scope | `level` | `diff` | `diff²` |
|---|---|---|---|---|
| **OPS-SAT-AD** | channel-level (n=5) | 0/5 significant (p_bonf capped at 1.000) | 5/5 significant (all p<2×10⁻⁴) | 5/5 significant (all p<2×10⁻⁴) |
| **SMAP/MSL** | dataset-pooled (n_anom=88 windows) | significant (p_fdr=0.031) | significant (p_fdr=4.7×10⁻⁹) | significant (p_fdr=1.3×10⁻⁶) |
| **SMD** | dataset-pooled (n_anom=11,493 windows) | significant (p_fdr=8×10⁻³⁸) | significant (p_fdr=3.7×10⁻¹⁶⁷) | significant (p_fdr=1.4×10⁻¹⁶⁶) |

**Direction is preserved in all three datasets**: `diff`/`diff²` clear placebo by much larger margins than `level`. The p-value gap is about 7 orders of magnitude in SMAP/MSL and about 129 in SMD. **Strict specificity (`level` = null) is not preserved**: `level` is weakly but reliably above placebo in both external datasets. Fig. 6a plots these three rows on a common −log₁₀(p) axis (broken for SMD); Fig. 6b gives the SMD channel-level breadth that is the more conservative summary described next.

**Independence caveat for the pooled p-values.** The pooled tests treat anomaly windows as exchangeable units, but windows are nested within channels and machines and are not independent. The extreme SMD p-values (10⁻¹⁶⁷) are therefore nominal and should be read as descriptive of direction, not as calibrated evidence strength. The **channel-level counts** are the more conservative summary (Fig. 6b): in SMD, 97/1,038 channels (9.3%) are significant for `level`, versus 142/1,038 (13.7%) for `diff` and 145/1,038 (14.0%) for `diff²`, i.e. roughly 1.5 times as many channels for the derivative features. That gap is consistent in direction with the pooled result and much more modest in size. A cluster-robust re-analysis (bootstrap by channel/machine) is listed under [Planned extensions](#planned-extensions-not-in-this-release).

**Channel-type-stratified results** (exploratory family, separately FDR-corrected). In SMAP/MSL, the `continuous` channel type shows `level` n.s. (p_fdr=0.35) with `diff`/`diff²` significant, the OPS-SAT-AD pattern exactly, while the `quantized` type shows `level` weakly significant. The `float_noise_suspect` type could not be tested in SMAP/MSL (1/82 channels, 3 anomaly windows) and showed no significant effect for any feature in SMD (fewest anomaly windows there: 162 across 17 channels).

### Stage 2 — Temporal precedence and normal-segment control (external replication of §3.2–3.3c; Fig. 7)

**Methodological note.** An initial attempt to increase power by sliding multiple onset candidates within each SMAP/MSL anomaly window (analogous to OPS-SAT-AD's Monte Carlo onset propagation) was found, on diagnosis, to contaminate the pre-onset baseline whenever the anomaly window (median 120 samples) exceeded the baseline length (20 samples): offset-shifted "onset candidates" ended up partially inside the anomaly itself. This was caught via a paired jointly-valid-candidate check and a window-length stratification test (both in `docs/dev-log/step_external_validation.md`). The results below use the corrected method: the **ground-truth onset only**, with an **adaptive (non-sliding) post-onset search window** whose upper bound was confirmed, via a sensitivity sweep from 150 to 5,000 samples, not to introduce censoring bias (n and effect direction stable across a 33× range of window caps).

**Table 13 — Precedence ordering and normal-control baseline decomposition**

| Dataset | Regime | n | `diff` precedes `level` | *p* |
|---|---|---|---|---|
| **OPS-SAT-AD** | anomalous | 70 | 84.3% | 1.4×10⁻⁵ |
| **OPS-SAT-AD** | normal control | 224 | 44.2% (**level** precedes in 55.8%) | — |
| **SMAP/MSL** | anomalous | 40 | 67.5% | 0.223 (n.s.) |
| **SMAP/MSL** | normal control | 7,596 | 56.5% | 1.06×10⁻¹²⁴ |
| **SMD** | anomalous | 3,612 | 76.6% | 2.9×10⁻²⁵ |
| **SMD** | normal control | 7,304 | 72.7% | 1.67×10⁻⁴⁴ |

Fig. 7a plots each dataset's normal-control and anomalous fractions with 95% Wilson intervals on the same axis.

**Baseline decomposition.** If differencing has an operator-level speed advantage on any local disturbance (see [Analytical baseline](#analytical-baseline-why-differencing-may-respond-faster)), the normal-control fraction estimates that baseline and the anomalous-minus-normal difference estimates the anomaly-attributable increment. Fig. 7b plots this increment directly, with an approximate 95% interval.

| Dataset | Normal-control baseline (`diff` first) | Anomalous (`diff` first) | Increment (pp) | Reading |
|---|---|---|---|---|
| OPS-SAT-AD | 44.2% | 84.3% | **+40.1** [29.4, 50.8] | Large; baseline below 50%, so ordering reverses |
| SMAP/MSL | 56.5% | 67.5% | +11.0 [−3.6, 25.6] | Same sign; anomalous-regime result n.s. (n=40), undecided |
| SMD | 72.7% | 76.6% | +3.9 [2.2, 5.6] | Same sign; small (nominal two-proportion *z*≈4.4 from the reported fractions, treating pairs as independent, so descriptive only) |

**Interpretation.**
- **The reversal is OPS-SAT-AD-specific.** In SMAP/MSL and SMD, `diff` precedes `level` in both regimes, at broadly similar rates, so no reversal occurs. For SMD this is a **non-replication** at high statistical power; for SMAP/MSL the anomalous-regime result is **underpowered and undecided**, because only 40 of 88 non-trivial windows have `level`, `diff` and `diff²` all crossing threshold within the same window (confirmed not to be a search-window-cap artifact).
- **The anomalous-vs-normal contrast has the same sign in all three datasets**, but its magnitude spans an order of magnitude (+40.1 vs. +3.9 pp). Outside OPS-SAT-AD, most of the `diff`-first ordering is present in normal controls as well, i.e. attributable to the operator-level baseline rather than to anomaly onsets. A small positive increment is what an anomaly-specific component layered on that baseline would produce, but the data here cannot separate it from other regime differences, and we do not present it as proof of one.
- **This is the single external result most in tension with the strongest form of the OPS-SAT-AD claim.** We report it as a boundary condition (§ Limitations, § Scope of the inferential claims) rather than explain it away.

### Summary of what generalizes and what does not

| | OPS-SAT-AD | SMAP/MSL | SMD |
|---|---|---|---|
| `diff`/`diff²` more strongly placebo-separating than `level` (direction) | ✓ | ✓ | ✓ |
| `level` shows no detectable excess over placebo | ✓ (0/5) | ✗ (weakly significant) | ✗ (significant) |
| Anomalous-vs-normal `diff`-first contrast has positive sign | ✓ (+40.1 pp) | ✓ (+11.0 pp, n.s.) | ✓ (+3.9 pp) |
| Ordering reverses in normal-segment control | ✓ | ✗ (anomalous n.s.) | ✗ (both significant, same direction) |
| `float_noise_suspect` > `quantized` pattern | ✓ (confounded with sensor family) | Undecidable (n<10) | Not confirmed |

Full per-dataset artifacts, including pooled/stratified test outputs and the sensitivity-analysis log, are in `results/external_validation/`.

---

## Analytical baseline: why differencing may respond faster

This section is an analytical argument and reports no new analysis.

The first post-onset |z|>3 crossing time is a signal-to-baseline-noise criterion: a series crosses when a disturbance exceeds three pre-onset standard deviations of that series. For series with slowly varying or autocorrelated components (drift, thermal trends, operating-mode wandering), first differencing suppresses those components, so the pre-onset standard deviation of `diff` is small relative to that of `level` for a comparable local disturbance. A disturbance of fixed amplitude is then more likely to cross the 3σ threshold in `diff` first, **irrespective of whether it is an anomaly**. `diff²` inherits the same advantage for variance-type disturbances.

We treat this as a **null-model expectation, not a result**. Whether it accounts for the external-dataset ordering is what the normal-control baseline in Table 13 / Fig. 7a measures: it is large in SMD (72.7%) and SMAP/MSL (56.5%), and below 50% in OPS-SAT-AD (44.2%), where the slowly varying components apparently do not dominate level noise to the same degree. This paper does not include a simulation that maps the regimes in which the operator baseline favors `diff` (see [Planned extensions](#planned-extensions-not-in-this-release)).

---

## Practical implications for alarm design

These implications concern **feature choice** and are hypothesis-generating: this paper does not benchmark a `diff`-based detector against `level`-based or published detectors.

- **Feature choice for thresholding.** Across all three datasets, `diff`/`diff²` variance features separate labeled onsets from placebo more strongly than `level` by per-feature rank tests (§ Layer 3.7, § External Validation). Favoring derivative-based features, at minimum as a complement to level-based thresholds, is a reasonable design choice; how much detection performance improves is not measured here. Practitioners should expect the operator-level baseline (Table 13, Fig. 7) to inflate apparent `diff` advantages on autocorrelated channels.
- **Channel-type-dependent sensitivity (OPS-SAT-AD heuristic).** Within OPS-SAT-AD, `diff`/`diff²`-based alarms may warrant higher sensitivity for `float_noise_suspect` (magnetometer) channels and supplementary `level`-based corroboration for `quantized` (photodiode) channels, where the derivative signature is markedly weaker (Table 3, § Layer 3.7). This pattern is confounded with sensor family and was not confirmed externally, so it is a tuning heuristic pending validation on more instrument types.
- **False-alarm trade-offs.** The Layer 2 scoping results (Table 1) show `CADC0892` and `CADC0894` combining high nominal recall with high false-alarm rates once BOCPD is applied naively, driven by extreme steady-state kurtosis rather than by a level/derivative feature-choice issue (§2.5). Feature choice alone does not resolve detector-tuning challenges and must be paired with channel-specific noise modeling.
- **Relationship to risk-aware alarm design (Paper 2).** This paper addresses *what* signal to threshold; the companion paper ([`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization)) addresses *how* to set the threshold under an explicit risk criterion (CVaR-based, with conformal calibration), consuming this repository's `onset_posteriors.parquet` and `channel_scope.json` artifacts.

---

## Scope of the inferential claims

We use the term **"signature"** to describe a claim of the form *"feature X changes at labeled anomaly onsets more than at placebo pivots on the same channel."* We deliberately avoid "causal" in the title and headline claims.

**What supports the OPS-SAT-AD reading.** A quasi-experimental same-channel placebo design; temporal-precedence testing with a normal-segment control; channel-stratified confirmation (CMH); train/test/bootstrap triple verification; and convergent classifier-side evidence from 16 architecturally diverse models. The structural skeleton (Fig. 4a) is asserted from the sensor's signal-processing structure and checked for consistency with data.

**What we do not claim.**
- We do **not** claim formal statistical identification in the Pearl do-calculus sense, and we do not run a discovery algorithm (PC, FCI, LiNGAM) to derive the skeleton's edges.
- We do **not** claim that OPS-SAT-AD's placebo comparison is free of selection asymmetry, since anomalous onsets are detector-selected and placebo pivots are position-matched resamples (see [Onset dependence](#onset-dependence-of-each-result)).
- Outside OPS-SAT-AD we claim only that `diff`/`diff²` carry more placebo-discriminative information than `level`, a purely associational statement that is still useful for feature selection but is **not** evidence of a universal mechanism linking anomaly onsets to derivative-statistic surges.

**Effect of the external results.** The normal-segment-control reversal, the strongest OPS-SAT-AD evidence against a "differencing is always faster" explanation, does not replicate in SMAP/MSL or SMD (Fig. 7). In both, `diff` precedes `level` in anomalous and normal regimes alike, a pattern consistent with the alternative explanation the reversal was designed to rule out. The small positive anomalous-minus-normal increments (Table 13, Fig. 7b) are compatible with an anomaly-specific component on top of the baseline, but do not establish it. A claim that has been tested against external data and narrowed accordingly is more trustworthy than one asserted from a single benchmark.

---

## Figures

Figures are generated by the scripts in `results/figures/` from `results/*.csv` / `results/*.json`. Figures 1–5 are OPS-SAT-AD-scoped; Figures 6–7 report the three-dataset external-validation comparisons (Tables 12–13) directly, so the cross-dataset result no longer rests on tables alone. Figures 2, 4, 5 and 7 encode categories redundantly by greyscale/hatch/marker shape and are legible in black-and-white print; Figures 1, 3 and 6 use blue/teal or open/filled-marker encoding with direct labels.

### Figure 1 — Channel scoping: MCC/Youden's J and the 9→5 funnel

![Figure 1: Channel scoping](results/figures/fig01_channel_scope.png)

*(a) MCC and Youden's J per channel at the locked hyperparameters (`mixture=False, forgetting=True`). Filled bars: channels in the final scope; dashed outlines: excluded channels (S structural, U underpowered, C chance-level). Counts under each channel are anomalous / nominal segments. CADC0890 has the highest point-estimate MCC (0.83) but only 3 nominal segments. (b) The 9→5 channel-scoping funnel; stage sizes and removals are derived from the channel table, and the scope is identical in the full, train-only and test-only re-derivations. These metrics serve channel scoping only and are not a detector benchmark.*
**Reproduces from:** `results/layer2/channel_scope.json`, `results/layer2/bootstrap_ci.csv` — **Script:** `results/figures/fig01_channel_scope.py`

### Figure 2 — 16-model cross-validation: feature attribution and agreement (OPS-SAT-AD only)

![Figure 2: Model cross-validation](results/figures/fig02_model_cross_validation.png)

*(a) Normalized feature-importance shares (level / diff / diff²) for each of 16 models, sorted by out-of-fold AUC within representation. The dotted line marks the 1/3 uniform share per feature; † marks the one model above it (LightMamba). 15 of 16 models give `level` less than 1/3, and no model ranks `level` first. (b) Number of models ranking each feature first, and agreement statistics recomputed from the shares; the binomial and Kendall's W values are nominal because the models share data and labels. This check controls classifier-side bias only, was run on OPS-SAT-AD only, and was not re-run on SMAP/MSL or SMD.*
**Reproduces from:** `results/model_cross_validation/feature_attribution_16models.csv`, `results/model_cross_validation/agreement_statistics.json` — **Script:** `results/figures/fig02_model_cross_validation.py`

### Figure 3 — Quasi-experimental placebo-pool test and triple verification (OPS-SAT-AD)

![Figure 3: Quasi-experimental comparison](results/figures/fig03_quasi_experimental.png)

*(a) −log₁₀(p_bonferroni), OPS-SAT-AD placebo-pool test, 15 tests. The `level` bars have zero height in every channel because p_bonf is capped at 1.000 (no detectable excess over placebo; not a demonstration of equivalence). Each feature is tested on its own statistic, so bar heights are not effect-size comparisons across features; channel type is aliased with sensor family. See Table 12 / Fig. 6 for the weaker `level` result in SMAP/MSL and SMD. (b) Full / train-only / test-only significance-call agreement matrix, OPS-SAT-AD-internal; the framed row is the single disagreement, diagnosed as a power artifact.*
**Reproduces from:** `results/layer3/placebo_comparison.csv`, `results/layer3/triple_verification_matrix.csv` — **Script:** `results/figures/fig03_quasi_experimental.py`

### Figure 4 — Structural working model and meta-analytic heterogeneity (OPS-SAT-AD)

![Figure 4: SCM and heterogeneity](results/figures/fig04_scm_and_heterogeneity.png)

*(a) Descriptive structural working model for OPS-SAT-AD ("Path A"): onset is followed by a transient `diff`/`diff²` variance surge; the `level` node is shown as persistent but confounded, and the "level excluded" annotation refers to OPS-SAT-AD's placebo test only (SMAP/MSL and SMD show weak `level` signal, Table 12). The moderator box (float_noise_suspect vs. quantized) is confounded with sensor family and was not confirmed externally. (b) DerSimonian–Laird heterogeneity (*I²*) across the 5 scoped channels, for both effect size |d| and significant-segment fraction; with k=5 these are descriptive.*
**Reproduces from:** `results/layer3/heterogeneity_summary.csv` — **Script:** `results/figures/fig04_scm_and_heterogeneity.py`

### Figure 5 — Dataset overview and pooled temporal-precedence test (OPS-SAT-AD)

![Figure 5: Dataset and precedence](results/figures/fig05_dataset_and_precedence.png)

*(a) OPS-SAT-AD dataset composition: segments per channel split into anomalous and nominal, with excluded channels shown in dashed/hatched bars and anomaly prevalence in italics. (b) Share of pairs in which the faster (second-named) feature crosses |z| > 3 first, in anomalous segments and normal-segment controls, for level-vs-diff, level-vs-diff² and diff-vs-diff². The diff-vs-diff² sign fraction and the pooled signed-rank test (Table 9) disagree, so no claim rests on that comparison. The corresponding SMAP/MSL and SMD results, including the normal-control baseline decomposition, are in Table 13 and Fig. 7.*
**Reproduces from:** `data/raw/` (segment metadata), `results/layer3/temporal_precedence.csv`, `results/layer3/temporal_precedence_normal_control.csv` — **Script:** `results/figures/fig05_dataset_and_precedence.py`

### Figure 6 — Placebo-pool separation across datasets

![Figure 6: Cross-dataset placebo comparison](results/figures/fig06_cross_dataset_placebo.png)

*(a) −log10 adjusted p for `level`, `diff` and `diff²` in OPS-SAT-AD (weakest of five channels, Bonferroni), SMAP/MSL and SMD (pooled, BH-FDR); the axis is broken because SMD p-values saturate. Pooled p-values treat windows as exchangeable and are nominal (§ Stage 1 independence caveat). (b) SMD channel-level breadth: fraction of tested channels significant per feature, with 95% Wilson intervals — the more conservative summary of the same result, showing roughly 1.5× as many channels flagged by the derivative features as by `level`.*
**Reproduces from:** `results/layer3/placebo_comparison.csv` (OPS-SAT-AD), `results/external_validation/stage1_placebo_pooled_dataset.csv`, `results/external_validation/stage1_channel_level_significance.csv` — **Script:** `results/figures/fig06_cross_dataset_placebo.py`

### Figure 7 — Operator baseline and anomaly-attributable increment

![Figure 7: Precedence ordering and baseline decomposition](results/figures/fig07_precedence_baseline.png)

*(a) Share of `diff`-before-`level` pairs in normal-segment controls (open circles) and anomalous segments (filled circles) with 95% Wilson intervals, for OPS-SAT-AD, SMAP/MSL and SMD. Only OPS-SAT-AD's baseline sits below 50%, which is why only OPS-SAT-AD shows a reversal. (b) The anomalous-minus-normal difference in percentage points with an approximate 95% interval. Both panels are computed from the reported fractions and pair counts, treating pairs as independent, so they are nominal (§ Stage 2). The increment has the same sign in all three datasets but ranges from +3.9 to +40.1 pp; the SMAP/MSL interval spans zero, so that dataset's anomalous-regime result remains undecided.*
**Reproduces from:** `results/layer3/temporal_precedence_normal_control.csv` (OPS-SAT-AD), `results/external_validation/stage2_temporal_precedence.csv`, `results/external_validation/stage2_baseline_decomposition.csv` — **Script:** `results/figures/fig07_precedence_baseline.py`

---

## Results tables

*Tables 1–11 are the OPS-SAT-AD primary-analysis tables; the external-validation tables (12–13) are in the section above, each paired with a figure (Fig. 6, Fig. 7).*

### Table 1 — Channel scoping decision table (underlies Fig. 1)

| Channel | Sensor | # segments | Anomaly % | Recall | FA rate | MCC | Youden's J | Decision |
|---|---|---|---|---|---|---|---|---|
| CADC0872 | Magnetometer #1 | 546 | 24.0 | 0.321 | 0.000 | 0.514 | 0.321 | **Included** |
| CADC0873 | Magnetometer #2 | 593 | 17.7 | 0.276 | 0.000 | 0.489 | 0.276 | **Included** |
| CADC0874 | Magnetometer #3 | 194 | 35.6 | 0.725 | 0.032 | 0.740 | 0.693 | **Included** |
| CADC0884 | Photodiode #1 | 158 | 0.0 | n/a | 0.241 | n/a | n/a | Excluded (structural) |
| CADC0886 | Photodiode #2 | 11 | 27.3 | 0.000 | 0.000 | 0.000 | 0.000 | Excluded (underpowered: n=3/8) |
| CADC0888 | Photodiode #3 | 252 | 23.8 | 0.617 | 0.391 | 0.194 | 0.226 | **Included** |
| CADC0890 | Photodiode #4 | 14 | 78.6 | 0.909 | 0.000 | 0.826 | 0.909 | Excluded (underpowered: n=11/14) |
| CADC0892 | Photodiode #5 | 211 | 16.1 | 0.971 | 0.994 | −0.090 | −0.024 | Excluded (chance-level) |
| CADC0894 | Photodiode #6 | 144 | 14.6 | 1.000 | 0.797 | 0.189 | 0.203 | **Included** |

### Table 2 — Model cross-validation summary (underlies Fig. 2, OPS-SAT-AD only)

| Model | Family | Representation | AUC | level share | diff share | diff² share |
|---|---|---|---|---|---|---|
| TinyTransformer | Attention | sequence | 0.994 | 0.178 | 0.438 | 0.384 |
| TCN | Conv (dilated causal) | sequence | 0.993 | 0.200 | 0.380 | 0.419 |
| BiGRU | Recurrent | sequence | 0.989 | 0.285 | 0.388 | 0.327 |
| CNN1D | Convolutional | sequence | 0.983 | 0.303 | 0.301 | 0.396 |
| BiLSTM | Recurrent | sequence | 0.980 | 0.211 | 0.497 | 0.292 |
| LightMamba | State-space (SSM) | sequence | 0.969 | 0.428 | 0.441 | 0.130 |
| SVM (RBF) | Kernel | tabular | 0.948 | 0.061 | 0.466 | 0.472 |
| LightGBM | Tree ens. (boosting) | tabular | 0.940 | 0.232 | 0.459 | 0.309 |
| XGBoost | Tree ens. (boosting) | tabular | 0.935 | 0.148 | 0.419 | 0.433 |
| RandomForest | Tree ens. (bagging) | tabular | 0.931 | 0.084 | 0.456 | 0.460 |
| ExtraTrees | Tree ens. (bagging, extra) | tabular | 0.927 | 0.145 | 0.389 | 0.466 |
| GradBoost (sklearn) | Tree ens. (boosting) | tabular | 0.924 | 0.068 | 0.371 | 0.561 |
| LogReg (L2) | Linear | tabular | 0.920 | 0.060 | 0.407 | 0.534 |
| LogReg (L1) | Linear (sparse) | tabular | 0.920 | 0.057 | 0.412 | 0.531 |
| kNN | Instance-based | tabular | 0.918 | 0.129 | 0.391 | 0.480 |
| GaussianNB | Probabilistic | tabular | 0.900 | 0.051 | 0.446 | 0.503 |

AUC is out-of-fold and used only to order models (see [Model Cross-Validation](#model-cross-validation-16-architecturally-diverse-models)).

### Table 3 — Quasi-experimental placebo-pool comparison (OPS-SAT-AD; underlies Fig. 3a, Fig. 6a)

| Channel | level, p_bonf | diff, p_bonf | diff², p_bonf |
|---|---|---|---|
| CADC0872 | 1.000 (capped) | 7.94×10⁻²⁴ | 1.12×10⁻²² |
| CADC0873 | 1.000 (capped) | 2.40×10⁻¹⁷ | 3.09×10⁻¹⁶ |
| CADC0874 | 1.000 (capped) | 5.37×10⁻¹⁶ | 2.88×10⁻²² |
| CADC0888 | 1.000 (capped) | 6.92×10⁻⁶ | 3.24×10⁻¹³ |
| CADC0894 | 1.000 (capped) | 3.89×10⁻⁶ | 1.95×10⁻⁴ |

*One-sided Mann–Whitney U per feature on its own statistic; 15 tests, Bonferroni-corrected. See Table 12 / Fig. 6 for the SMAP/MSL and SMD equivalents (weaker `level` result).*

### Table 4 — Heterogeneity summary (underlies Fig. 4b, OPS-SAT-AD)

| Outcome | I² | Q-test p |
|---|---|---|
| Effect size \|d\| (level) | 65.7% | 0.0202 |
| Effect size \|d\| (diff) | 97.2% | <0.0001 |
| Effect size \|d\| (diff²) | 93.8% | 2.6×10⁻¹³ |
| frac_sig (level) | 82.8% | 1.1×10⁻⁴ |
| frac_sig (diff) | 93.8% | 3.2×10⁻¹³ |
| frac_sig (diff²) | 92.2% | 1.9×10⁻¹⁰ |

### Table 5 — Full/train/test triple-verification agreement (underlies Fig. 3b, OPS-SAT-AD)

| Channel | Feature | Sig. (full) | Sig. (train) | Sig. (test) | 3-way agree |
|---|---|---|---|---|---|
| CADC0872 | level | No | No | No | Yes |
| CADC0872 | diff | Yes | Yes | Yes | Yes |
| CADC0872 | diff² | Yes | Yes | Yes | Yes |
| CADC0873 | level | No | No | No | Yes |
| CADC0873 | diff | Yes | Yes | Yes | Yes |
| CADC0873 | diff² | Yes | Yes | Yes | Yes |
| CADC0874 | level | No | No | No | Yes |
| CADC0874 | diff | Yes | Yes | Yes | Yes |
| CADC0874 | diff² | Yes | Yes | Yes | Yes |
| CADC0888 | level | No | No | No | Yes |
| CADC0888 | diff | Yes | Yes | **No** (p=0.051) | **No** (power loss) |
| CADC0888 | diff² | Yes | Yes | Yes | Yes |
| CADC0894 | level | No | No | No | Yes |
| CADC0894 | diff | Yes | Yes | n/a (n=4<5) | n/a |
| CADC0894 | diff² | Yes | Yes | n/a (n=4<5) | n/a |

### Table 6 — Model-cross-validation agreement and stability statistics (OPS-SAT-AD)

| Statistic | Value |
|---|---|
| Kendall's *W* | **0.609** |
| χ² approximation (df=2) | χ²=19.50, p<.001 |
| Models ranking `level` top | 0/16 (tabular 0/10, sequence 0/6) |
| Binomial test, (2/3)¹⁶ | p=.0015 (nominal; models share data) |
| Models with `level` share < 1/3 | 15/16 |
| Bootstrap stability, RandomForest | 100% favor diff+diff² > level |
| Bootstrap stability, LogisticRegression | 100% favor diff+diff² > level |

### Table 7 — Grouping-level aggregation (13 finer-grained groupings within 9 coarse families, OPS-SAT-AD)

| Grouping | Coarse family | n models | mean level | mean diff | mean diff² | diff+diff² > level |
|---|---|---|---|---|---|---|
| Linear | Linear | 1 | 0.060 | 0.407 | 0.534 | Yes |
| Linear (sparse) | Linear | 1 | 0.057 | 0.412 | 0.531 | Yes |
| Probabilistic | Probabilistic | 1 | 0.051 | 0.446 | 0.503 | Yes |
| Instance-based | Instance-based | 1 | 0.129 | 0.391 | 0.480 | Yes |
| Kernel | Kernel | 1 | 0.061 | 0.466 | 0.472 | Yes |
| Tree ensemble (bagging) | Tree ensemble | 1 | 0.084 | 0.456 | 0.460 | Yes |
| Tree ensemble (bagging, extra) | Tree ensemble | 1 | 0.145 | 0.389 | 0.466 | Yes |
| Tree ensemble (boosting) | Tree ensemble | 3 | 0.149 | 0.416 | 0.434 | Yes |
| Convolutional | Convolutional | 1 | 0.303 | 0.301 | 0.396 | Yes |
| Convolutional (dilated causal) | Convolutional | 1 | 0.200 | 0.380 | 0.419 | Yes |
| Recurrent | Recurrent | 2 | 0.248 | 0.443 | 0.310 | Yes |
| Attention | Attention | 1 | 0.178 | 0.438 | 0.384 | Yes |
| State-space (SSM) | State-space | 1 | 0.428 | 0.441 | 0.130 | Yes (narrowest margin) |

### Table 8 — 2×2 ablation grid (train macro-MCC; OPS-SAT-AD)

| mixture | forgetting | train_macro_mcc | n_scoring_channels |
|---|---|---|---|
| False | True (**locked**) | 0.308 | 6 |
| True | True | 0.275 | 6 |
| False | False | 0.238 | 6 |
| True | False | 0.199 | 6 |

### Table 9 — Pooled temporal-precedence test (OPS-SAT-AD; underlies Fig. 5b)

| Comparison | n pairs | p (Wilcoxon) | Interpretation |
|---|---|---|---|
| level vs. diff | 70 | 1.4×10⁻⁵ | diff precedes level in 84.3% of pairs |
| level vs. diff² | 70 | 3.2×10⁻⁵ | diff² precedes level in 82.9% of pairs |
| diff vs. diff² | 161 | 0.368 (n.s.) | no reliable ordering by the signed-rank test |

*See Table 13 / Fig. 7 for the SMAP/MSL and SMD equivalents (no reversal in normal control).*

### Table 10 — Normal-segment control: precedence ordering (OPS-SAT-AD only; underlies Fig. 5b)

| Comparison | Regime | n pairs | frac(first precedes second) |
|---|---|---|---|
| level vs. diff | anomalous | 70 | level 15.7% / diff **84.3%** |
| level vs. diff | normal | 224 | level **55.8%** / diff 44.2% |
| level vs. diff² | anomalous | 70 | level 17.1% / diff² **82.9%** |
| level vs. diff² | normal | 171 | level **74.3%** / diff² 25.7% |
| diff vs. diff² | anomalous | 161 | diff 7.5% / diff² **92.5%** |
| diff vs. diff² | normal | 211 | diff **64.5%** / diff² 35.5% |

*Reporting note on diff vs. diff².* Table 10 gives sign fractions (diff² first in 92.5% of anomalous pairs) while Table 9 gives a signed-rank test on the same comparison (n.s.). The two statistics weight pairs differently: the sign fraction ignores lag magnitude, while the signed-rank test weights by it, and the cross-correlation analysis (§3.4) finds no anomalous-vs-normal difference in lag distribution. We draw **no** claim from the diff-vs-diff² ordering and report both rows descriptively. The reversal pattern in the `level` rows does not replicate in SMAP/MSL or SMD (Table 13, Fig. 7).

<!-- VERIFY before submission: reconcile Table 9 (diff vs diff², n=161, Wilcoxon p=0.368) with Table 10 (diff² first in 92.5% of the same 161 pairs). Check tie handling and the definition of the sign fraction in temporal_precedence*.csv, and replace the reporting note above with the confirmed explanation. -->

### Table 11 — Layer 1 noise estimates (OPS-SAT-AD)

| Channel | Type | r_robust | q_robust |
|---|---|---|---|
| CADC0872 | float_noise_suspect | 2.65×10⁻¹² | 4.83×10⁻¹³ |
| CADC0873 | float_noise_suspect | 2.51×10⁻¹² | 2.94×10⁻¹³ |
| CADC0874 | float_noise_suspect | 8.02×10⁻¹³ | 2.73×10⁻¹³ |
| CADC0884 | quantized | 4.86×10⁻⁴ | 2.83×10⁻⁵ |
| CADC0886 | quantized | 1.33×10⁻³ | 7.37×10⁻⁵ |
| CADC0888 | quantized | 1.28×10⁻³ | 8.23×10⁻⁵ |
| CADC0890 | continuous | 2.32×10⁻³ | 2.69×10⁻⁴ |
| CADC0892 | quantized | 1.51×10⁻⁴ | 3.79×10⁻⁵ |
| CADC0894 | quantized | 1.33×10⁻⁴ | 1.42×10⁻⁵ |

*Table 12 (placebo-pool comparison across datasets; underlies Fig. 6) and Table 13 (precedence ordering and baseline decomposition; underlies Fig. 7) are in [External Validation](#external-validation--generalization-study-smapmsl-smd).*

---

## Reproducing every number in this README

```bash
# 1. Clone and set up the environment
git clone https://github.com/USERNAME/opssat-ad-onset-signatures.git
cd opssat-ad-onset-signatures
conda env create -f environment.yml && conda activate opssat-ad-signatures   # or: pip install -r requirements.txt

# 2. Download all three datasets (none are redistributed in this repo)
bash data/download_opssat_ad.sh
bash data/download_smap_msl.sh
bash data/download_smd.sh

# 3. Run the full OPS-SAT-AD Layer 1 → 2 → 3 pipeline
python -m layer1_signal_estimation.run
python -m layer2_anomaly_detection.run
python -m layer3_signature_analysis.run

# 4. Run the labeling-protocol audit + train/test triple re-verification
python -m layer3_signature_analysis.labeling_protocol_audit
python -m layer3_signature_analysis.train_test_triple_reverification

# 5. Run the 16-model cross-validation benchmark (OPS-SAT-AD)
python -m model_cross_validation.run_all

# 6. Run the external-validation pipeline (SMAP/MSL, SMD)
python -m external_validation.run_external_validation

# 7. Regenerate every figure in this README from the results/ artifacts above
python results/figures/fig01_channel_scope.py
python results/figures/fig02_model_cross_validation.py
python results/figures/fig03_quasi_experimental.py
python results/figures/fig04_scm_and_heterogeneity.py
python results/figures/fig05_dataset_and_precedence.py
python results/figures/fig06_cross_dataset_placebo.py
python results/figures/fig07_precedence_baseline.py

# ...or simply:
make all
```

`REPRODUCIBILITY_CHECKLIST.md` documents random seeds, package versions, and expected runtime (CPU-only: ~40 min for OPS-SAT-AD Layers 1–3; the 16-model benchmark is the long pole at ~2–3 hours on CPU / ~25 min on a single GPU; the external-validation pipeline runs in ~15–30 min on CPU, dominated by SMD's 1,064-channel loop; Figures 6–7 regenerate in seconds once `results/external_validation/` exists).

---

## Threats to validity and how each was addressed

Each row states the mitigation and, where it is partial, what remains open.

| Threat | Mitigation | Residual status |
|---|---|---|
| **Onset-estimation circularity (OPS-SAT-AD):** onsets come from BOCPD on Kalman innovations, which may favor transient variance changes; placebo pivots are position-matched, not detector-selected | Replication on SMAP/MSL and SMD with ground-truth onsets (Table 12–13, Fig. 6–7) | **Open for OPS-SAT-AD strong-form claims**; the directional claim is supported by onset-independent data |
| **Non-independence of pooled external tests** (windows nested in channels/machines) | Channel-level counts reported alongside pooled p-values (Fig. 6b); pooled p-values labeled nominal; confirmatory/exploratory families separated | Cluster-robust re-analysis not yet run ([Planned extensions](#planned-extensions-not-in-this-release)) |
| **Operator-level advantage of differencing** ("`diff` is just faster") | Normal-segment controls in all three datasets; baseline decomposition (Table 13, Fig. 7); analytical baseline stated | Regime-mapping simulation not yet run |
| **Feature statistics on different scales** (|d| vs. log-variance ratio) | Per-feature rank tests; p-values interpreted as distinguishability from placebo, not cross-feature effect size | Common-scale metric (e.g., AUROC, detection delay) not yet run |
| **Train/test leakage in channel selection (OPS-SAT-AD)** | Scoping re-run on train-only and test-only splits; identical 5-channel scope in all three passes | Addressed |
| **Train/test leakage in the placebo-pool result (OPS-SAT-AD)** | §3.7 re-run on train-only/test-only slices; 12/13 decidable comparisons agree, the disagreement diagnosed as a power artifact | Addressed |
| **Result being a labeling artifact (OPS-SAT-AD)** | External audit of the labeling protocol: no documented cutting rule | Undecidable; not excluded |
| **Classifier inductive bias in feature ranking** | 16 models, 9 coarse families, 2 representations, 2 attribution methods | Addressed for the classifier side only; models share data and labels |
| **`level` looking important only because of a weaker null (OPS-SAT-AD)** | Same-channel placebo-pool design | Addressed; "no detectable excess", not equivalence |
| **`level`'s within-segment significance contradicting the precedence result (OPS-SAT-AD)** | Transient-vs-persistent profile classification, early-window sensitivity, normal-segment control | Addressed |
| **Small-sample channels inflating effect sizes (OPS-SAT-AD)** | Structural/underpowered channels excluded rather than down-weighted; CADC0890's exclusion despite the highest MCC is a deliberate conservative choice | Addressed; limits scope to 5 channels |
| **Between-channel heterogeneity treated as noise (OPS-SAT-AD)** | I²/Q-test, dual-criterion LOO sensitivity | Moderator confounded with sensor family; k=5 |
| **Multiple-comparisons inflation (OPS-SAT-AD)** | Bonferroni across all 15 placebo-pool tests and their train/test replications; pooled Wilcoxon for the precedence claim | Addressed |
| **Multiple-comparisons inflation across many external channels** | Benjamini–Hochberg FDR; confirmatory and exploratory families corrected separately | Addressed |
| **Underpowered per-channel testing in SMAP/MSL** | Dataset-pooled and channel-type-stratified tests as the primary design | Addressed; SMAP/MSL precedence result remains underpowered |
| **Sliding-window onset refinement contaminating the pre-onset baseline (SMAP/MSL; caught during analysis)** | Diagnosed via paired jointly-valid-candidate and window-length-stratification checks; replaced by ground-truth-onset, non-sliding, adaptive-window method; censoring bias ruled out by a 150–5,000-sample sweep | Addressed; full trail in `docs/dev-log/step_external_validation.md` |
| **Visually trivial point outliers inflating effects in SMAP/MSL** | Triviality filter (>20 nominal-SD exclusion) per Wu & Keogh (2021) | Addressed |
| **Result being an artifact of OPS-SAT-AD's instrument physics** | Independent replication on SMAP/MSL and SMD (Fig. 6–7) | Direction confirmed 3/3; strict specificity and reversal confirmed only in OPS-SAT-AD |

---

## Limitations

Ordered by their importance for interpreting the results.

1. **Onset-estimation circularity in OPS-SAT-AD.** Onsets are estimated by BOCPD on Kalman innovations, and anomalous onsets are detector-selected while placebo pivots are position-matched resamples. The direction of possible bias (favoring transient variance changes over persistent level shifts) is plausible but is not quantified. Only 178/386 (46.1%) anomalous segments are scoreable, and all Layer 3 results and the 16-model labels are conditional on this subset and estimator. The onset-independent evidence is the SMAP/MSL and SMD replication (Fig. 6–7).
2. **Pooled external p-values are nominal.** Anomaly windows are nested within channels and machines, so SMD p-values of 10⁻¹⁶⁷ overstate evidence strength (Fig. 6a). The channel-level counts (97 / 142 / 145 of 1,038 channels for `level` / `diff` / `diff²`; Fig. 6b) are the more conservative summary and show a gap of roughly 1.5×.
3. **Much of the external `diff`-first ordering is present in normal controls.** The anomaly-attributable increment is +3.9 pp in SMD and +11.0 pp (n.s.) in SMAP/MSL, versus +40.1 pp in OPS-SAT-AD (Fig. 7b). Outside OPS-SAT-AD the data are consistent with a large operator-level baseline and a small anomaly-specific component, and cannot rule out that differencing is faster for any local disturbance.
4. **The strict-specificity claim and the normal-control reversal are OPS-SAT-AD-specific.** `level` retains weak but detectable signal above placebo in SMAP/MSL (p_fdr=0.031) and SMD, and no reversal occurs in either (Fig. 7a). Non-significance of `level` in OPS-SAT-AD is a failure to reject (p_bonf capped at 1.000), not a demonstrated equivalence.
5. **Feature statistics are not on a common scale.** `level` uses |Cohen's d| and `diff`/`diff²` use log-variance ratios; per-feature rank tests support "distinguishable from placebo" statements, not cross-feature effect-size comparisons.
6. **The channel-type moderator is confounded with sensor family.** `float_noise_suspect` = magnetometer (3 channels) and `quantized` = photodiode (2 channels in scope) in OPS-SAT-AD, with *I²* of 92–97% for the `diff`/`diff²` effects. The pattern could not be tested in SMAP/MSL (1 of 82 channels) and was not confirmed in SMD (no significant effect; 162 anomaly windows across 17 channels).
7. **The 16-model cross-validation is a classifier-side check only.** It shares data and onset-derived labels across models, uses a 3-feature design (so "`diff`+`diff²` > `level`" is expected at 2:1 under a uniform-share null), relies on non-independent binomial and Kendall statistics, and has out-of-fold AUCs from placebo pivots that are multiple per normal segment. The LightMamba model is a lightweight non-CUDA substitute for `mamba_ssm`, so its narrower margin should not be over-interpreted. The check was not replicated on SMAP/MSL or SMD, and the labeling-protocol audit was not repeated externally.
8. **Scope of OPS-SAT-AD analysis.** 5 of 9 channels are excluded for statistical-power reasons (Table 1), so OPS-SAT-AD-scoped claims cover these 5 channels only. CADC0894 has a small anomalous count (n=21 full-data, n=4 in the test-only slice), which limits power in every test-split replication. CADC0874's `diff²` effect is likely under-estimated by the full-window Welch design given its especially short-lived transient; treat its meta-analytic estimate as a conservative lower bound.
9. **The labeling-protocol audit is undecidable.** No public documentation specifies a quantitative cutting rule, so a non-systematic labeling influence on the onset-position skew (mean position ratio ≈0.569) cannot be excluded.
10. **The SMAP/MSL anomalous-regime precedence result is underpowered** (n=40, p=0.223, CI spans zero in Fig. 7b) and is described as undecided; SMD shows the same non-reversal pattern at high power and is described as a non-replication. SMD's domain (industrial servers) differs substantially from spacecraft instrumentation and is weighted as supporting evidence.
11. **The diff-vs-diff² ordering is unresolved** (Table 9 vs. Table 10) and no claim rests on it.
12. **No detector benchmark.** Detection performance is modest in absolute terms on some channels (CADC0888 and CADC0894 sit well below CADC0874), and no comparison with published OPS-SAT-AD detectors is made; detection results serve channel scoping only.

---

## Planned extensions (not in this release)

These would directly address the limitations above and are not part of the results reported here.

- **Synthetic regime map**: matched-SNR simulations of level shifts vs. variance surges on autocorrelated baselines, to show when the operator-level baseline alone favors `diff`.
- **Cluster-robust inference**: bootstrap or mixed-effects analysis with segments (OPS-SAT-AD) and channels/machines (SMAP/MSL, SMD) as clusters.
- **Common-scale comparison**: AUROC and detection delay computed identically for `level`, `diff`, `diff²`.
- **Onset-estimator sensitivity** for OPS-SAT-AD (alternative estimators, or detector-matched placebo pivots).
- **External model cross-validation**: reduced (2–4 family) replication of the model-diversity check on SMAP/MSL or SMD; official `mamba_ssm` for the SSM family.
- **Published-baseline comparison** for OPS-SAT-AD detection metrics.

---

## License

Code in this repository is released under the MIT License (see `LICENSE`). This repository does not redistribute any of the three datasets used; each dataset is governed by its own upstream license and terms of use:

- **OPS-SAT-AD**: see the dataset's Zenodo record and `data/README.md` for license terms.
- **SMAP/MSL (Telemanom)**: see the dataset's public release terms and `data/README.md`.
- **SMD (OmniAnomaly)**: see the dataset's public release terms and `data/README.md`.

Users are responsible for independently confirming that their intended use complies with each dataset's license.
