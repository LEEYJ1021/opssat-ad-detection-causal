# opssat-ad-detection-causal

**Detection and Cross-Dataset Signature Analysis of Telemetry Anomalies: Evidence from OPS-SAT-AD, SMAP/MSL, and SMD**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-fully%20reproducible-brightgreen.svg)](#reproducing-every-number-in-this-readme)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](#citation)

> **Summary finding.** Across three independently curated benchmarks spanning two space agencies and one industrial domain (OPS-SAT-AD: 9 spacecraft telemetry channels; SMAP/MSL: 82 spacecraft telemetry channels; SMD: 1,064 server-machine channels), a transient surge in first- and second-difference variance (`diff`, `diff²`) is more strongly and more consistently associated with labeled anomalies than a level (mean) shift. This directional result reproduces in all three datasets. A stronger, dataset-specific claim — that `level` is *fully* indistinguishable from ordinary operational drift and that the precedence ordering *reverses* on normal-operation controls — holds decisively in OPS-SAT-AD (Bonferroni-corrected placebo-pool test, 5/5 channels n.s. for `level`, all 5 channels significant for `diff`/`diff²`; reversal p<10⁻⁴) but is **not observed** in SMAP/MSL or SMD, where `level` retains weak but detectable signal above placebo and no ordering reversal occurs. We report both results transparently and interpret the boundary between them as informative about which instrument-physics regimes support a strict causal-signature reading versus a weaker associational one.

This repository is the code, data-provenance, and figure-reproduction companion to **Paper 1** of a two-paper series built primarily on the OPS-SAT-AD dataset, with independent generalization testing on SMAP/MSL and SMD. It is **self-contained and independently evaluable** — it does not depend on any other repository.

| This series | Repository | Status |
|---|---|---|
| **Paper 1 (this repo)** — Detection + Cross-Dataset Signature Analysis | `opssat-ad-detection-causal` | ✅ independent, no upstream deps |
| Paper 2 — Risk-Aware Alarm Design (CVaR / conformal) | [`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization) | depends on this repo's `results/` artifacts |
| DSS prototype (not a paper) | [`opssat-ad-dss`](https://github.com/USERNAME/opssat-ad-dss) | depends on both papers' artifacts |

---

## Table of Contents

1. [Why this repository exists](#why-this-repository-exists)
2. [Headline results](#headline-results)
3. [Repository structure](#repository-structure)
4. [Datasets](#datasets)
5. [Methodology (OPS-SAT-AD primary pipeline)](#methodology-ops-sat-ad-primary-pipeline)
   - [Layer 1 — Signal Estimation](#layer-1--signal-estimation)
   - [Layer 2 — Anomaly Detection (change-point model + channel scoping)](#layer-2--anomaly-detection-change-point-model--channel-scoping)
   - [Layer 3 — Signature Analysis](#layer-3--signature-analysis)
   - [Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification](#layer-3b--labeling-protocol-audit--traintest-triple-verification)
   - [Model Cross-Validation (16 independent architectures)](#model-cross-validation-16-independent-architectures)
6. [External Validation / Generalization Study (SMAP/MSL, SMD)](#external-validation--generalization-study-smapmsl-smd)
7. [Comparison against published OPS-SAT-AD baselines](#comparison-against-published-ops-sat-ad-baselines)
8. [Practical implications for alarm design](#practical-implications-for-alarm-design)
9. [Relationship to formal causal inference — scope statement](#relationship-to-formal-causal-inference--scope-statement)
10. [Figures](#figures)
11. [Results tables](#results-tables)
12. [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
13. [Threats to validity and how each was addressed](#threats-to-validity-and-how-each-was-addressed)
14. [Limitations](#limitations)
15. [Citation](#citation)
16. [License](#license)

---

## Why this repository exists

Change-point and anomaly-detection pipelines for spacecraft and industrial telemetry commonly treat a **level shift** (a change in the mean of the raw signal) as the default anomaly signature — it is the easiest thing to visualize, the easiest thing to threshold, and the default target of most textbook change-point models. This repository documents a systematic investigation, beginning with the **OPS-SAT-AD** benchmark (European Space Agency OPS-SAT mission, 9 telemetry channels, 2,123 expert-labeled univariate segments), of whether level shifts are in fact the operative signature, or whether **variance surges in the first and second differences of the signal (`diff`, `diff²`)** carry more anomaly-relevant information.

This is not a claim we set out to prove. The project's early working hypothesis, stated explicitly before any hypothesis testing began, was the naive `level → diff → diff²` causal ordering — that a level shift is the primary event and the derivative statistics are downstream artifacts of it. Section 3.2's pooled Wilcoxon test was run specifically to confirm this ordering and instead found the opposite pattern: `diff`/`diff²` precede `level` in 82–84% of paired segments within OPS-SAT-AD. Every subsequent analysis in Layer 3 was designed to stress-test whether that reversal was itself real or an artifact, rather than to reinforce it — and, subsequently, whether it holds outside OPS-SAT-AD at all.

Because a single-pipeline, single-dataset finding is easy to dismiss as an artifact of one detector's inductive bias or one instrument's physics, the analysis proceeded along five independent lines of attack, each targeting a different class of possible confound:

1. **Is `level`'s apparent significance just a weaker null?** → same-channel quasi-experimental placebo-pool comparison (§ Layer 3.7).
2. **Is the diff/diff² pattern a side effect of how segments were manually cut?** → external audit of the OPS-SAT-AD labeling protocol (§ Layer 3b).
3. **Is the result an artifact of using the full dataset rather than a proper train/test split?** → independent train-only and test-only reproduction of the entire placebo-pool test (§ Layer 3b).
4. **Is the result an artifact of the BOCPD/Kalman-filter pipeline's own inductive bias?** → an independent 16-model cross-validation study spanning 13 inductive-bias families and 2 independent data representations (§ Model Cross-Validation).
5. **Is the result an artifact of OPS-SAT-AD's specific instrument physics (quantization, sensor noise regime) rather than a general property of anomaly onsets?** → independent replication on two external, structurally different benchmarks — SMAP/MSL (NASA spacecraft telemetry) and SMD (industrial server telemetry) (§ External Validation).

The first four lines of evidence converge cleanly within OPS-SAT-AD. The fifth does **not** converge as cleanly: it confirms the directional claim (diff/diff² carry more anomaly-relevant information than level) but does not replicate the strongest specificity and reversal claims outside OPS-SAT-AD. We treat this as a genuine, informative finding about the scope of the result rather than a failure to be minimized — see § External Validation and § Limitations for the full account.

This repository packages:

- The full **signal-estimation → change-point-detection → signature-analysis** pipeline for OPS-SAT-AD (Layers 1–3).
- The **labeling-protocol audit and full train/test/bootstrap triple re-verification** of the OPS-SAT-AD result (Layer 3b).
- The **16-model cross-validation** benchmark and its agreement statistics (Kendall's W, binomial test, bootstrap stability).
- An **independent external-validation pipeline** applying the same placebo-pool and temporal-precedence logic, adapted to each dataset's label structure, to SMAP/MSL and SMD (§ External Validation).
- **Every figure in this README regenerated from its exact source script** (`results/figures/*.py`), so every number a reviewer sees can be traced to code, not to a table typed by hand.
- **Raw development logs** (`docs/dev-log/`) documenting the full, unabridged analytical trail — including negative results, discarded ablations, methodological corrections (train/test leakage checks, SHAP return-shape bugs, cuDNN backward-mode fixes, and the external-validation debugging trail described in § External Validation), and the reasoning behind every locked hyperparameter. These are provided for full transparency and are **not** a substitute for the polished Methods text in this README and `docs/METHODS_SUPPLEMENT.md`.

---

## Headline results

| Claim | OPS-SAT-AD Evidence | Generalizes to SMAP/MSL, SMD? |
|---|---|---|
| Channel scope: 9 → 5 channels, survives 3 independent verifications | Fig. 1 | N/A (OPS-SAT-AD-specific detection pipeline) |
| `diff`/`diff²` are more strongly associated with anomalies than `level` (direction) | Fig. 3a, Table 3 | **Yes — 3/3 datasets** (§ External Validation) |
| `level` is fully indistinguishable from normal-operation drift (strict specificity) | Fig. 3a, 5/5 channels n.s., p_bonf=1.000 uniformly | **No** — weakly but significantly above placebo in both SMAP/MSL (p_fdr=0.031) and SMD (p_fdr=8×10⁻³⁸) (§ External Validation) |
| `diff`/`diff²` precede `level` in onset timing, and this ordering **reverses** on normal-segment controls (decisive falsification of "differencing is just faster" alternative) | Fig. 5b, Table 10 | **No** — direction (diff precedes level) holds in both anomalous and normal regimes in SMAP/MSL and SMD; no reversal observed (§ External Validation) |
| Channel-type moderator: `float_noise_suspect` (magnetometer) channels show markedly stronger diff/diff² effects than `quantized` (photodiode) channels | Fig. 4a | **Undecided in SMAP/MSL** (only 1/82 channels of this type, underpowered); **not confirmed in SMD** (this channel type shows no significant effect at all) |
| Placebo-pool result replicates independently on full/train-only/test-only splits | Fig. 3b (12/13 decidable comparisons agree) | Not re-tested externally (OPS-SAT-AD-internal robustness check) |
| 16/16 independent models (13 inductive-bias families) rank `diff`+`diff²` over `level`; Kendall's W=0.609, binomial p=.0015 | Fig. 2 | Not re-tested externally (see § Limitations); a reduced 3–4 model replication is a planned extension |
| No documented quantitative segment-cutting rule exists in the OPS-SAT-AD labeling protocol | — | N/A (OPS-SAT-AD-specific) |
| Apparent `level`-significance paradox (high within-segment significance despite losing the precedence race) is resolved via transient-vs-persistent profile classification | — | Not re-tested externally |

**Reading this table.** The directional claim in row 2 is the load-bearing result of this paper and is the one we ask readers to treat as established. Rows 3–5 describe a stronger, OPS-SAT-AD-specific pattern that does not travel to the two external datasets tested; we retain it as a *conditional* finding, scoped explicitly to OPS-SAT-AD's instrument physics, rather than removing it, because the boundary itself is scientifically informative (§ External Validation, § Limitations).

---

## Repository structure

```
opssat-ad-detection-causal/
│
├── README.md                              # this file — full methodology + inline figures
├── LICENSE
├── CITATION.cff
├── requirements.txt
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
├── layer2_anomaly_detection/
│   ├── bocpd_forgetting.py                # BOCPD with forgetting-factor extension
│   ├── ablation_mixture_forgetting.py     # 2×2 ablation grid (mixture × forgetting)
│   ├── channel_scoping.py                 # MCC / Youden's J channel decision table (Fig. 1a)
│   ├── triple_verification.py             # full / train-only / test-only re-fit & re-select (Fig. 1b)
│   ├── bootstrap_ci.py                    # per-channel MCC bootstrap confidence intervals
│   ├── problem_channel_diagnosis.py       # Hypothesis A/B diagnosis for 890/892/894 (§ Layer 2.5)
│   ├── tests/
│   └── run.py
│
├── layer3_causal_analysis/
│   ├── onset_mc_propagation.py            # Monte Carlo propagation of onset-time posterior (200 draws)
│   ├── temporal_precedence_test.py        # pooled Wilcoxon signed-rank, level vs diff vs diff² (Fig. 5b)
│   ├── temporal_profile_classification.py # transient vs. persistent post-onset trajectory classification
│   ├── early_window_sensitivity.py        # 10-point vs. full-window Welch-test comparison
│   ├── quasi_experimental_placebo.py      # Mann–Whitney U vs. same-channel placebo pool (Fig. 3a)
│   ├── meta_analysis_random_effects.py    # DerSimonian–Laird random-effects meta-analysis + I² (Fig. 4b)
│   ├── loo_sensitivity.py                 # leave-one-channel-out sensitivity analysis
│   ├── scm_skeleton.py                    # structural skeleton construction, Path A vs. Path B (Fig. 4a)
│   ├── supplementary_diagnostics_A_D.py   # cross-correlation, CMH test, normal-segment controls, lag resolution
│   ├── labeling_protocol_audit.py         # external audit of the OPS-SAT-AD labeling protocol
│   ├── train_test_triple_reverification.py# train-only / test-only reproduction of the placebo-pool test
│   ├── tests/
│   └── run.py
│
├── model_cross_validation/
│   ├── models/                            # 16 independent model wrappers (+1 excluded, see § below)
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
│   ├── figures/                           # ← every PNG below, plus the .py script that produced it
│   │   ├── fig01_channel_scope.png / .py
│   │   ├── fig02_model_cross_validation.png / .py
│   │   ├── fig03_quasi_experimental.png / .py
│   │   ├── fig04_scm_and_heterogeneity.png / .py
│   │   └── fig05_dataset_and_precedence.png / .py
│   └── tables/                            # camera-ready LaTeX tables (.tex) mirroring the CSVs above
│
└── docs/
    ├── dev-log/                           # raw, unabridged research log (see note below)
    │   ├── step1_signal_estimation.md
    │   ├── step2_anomaly_detection.md
    │   ├── step3_causal_analysis.md
    │   ├── step3b_labeling_protocol_revalidation.md
    │   └── step_ai_model_cross_validation.md
    ├── METHODS_SUPPLEMENT.md               # manuscript-ready extended methods (mirrors README §5)
    └── REPRODUCIBILITY_CHECKLIST.md
```

> **Note on `docs/dev-log/`.** These files are the original, conversational research logs kept during development, included **verbatim** for full analytical transparency. `docs/dev-log/step_external_validation.md` in particular documents a methodological correction made mid-analysis: an early version of the SMAP/MSL temporal-precedence test used a sliding-window onset-refinement scheme that, we discovered, contaminated the pre-onset baseline for windows longer than the baseline length — this was diagnosed, the affected results discarded, and the analysis re-run with ground-truth onsets and a baseline-safe adaptive search window (see § External Validation for the validated methodology). This correction is disclosed here in the interest of full transparency, per the reproducibility standard applied throughout this repository.

---

## Datasets

### OPS-SAT-AD (primary)

European Space Agency's OPS-SAT experimental nanosatellite telemetry anomaly-detection benchmark (Ruszczak, Kotowski, Evans & Nalepa, *Scientific Data* 12:710, 2025).

- **9 channels**: 3 magnetometer channels (`CADC0872`, `CADC0873`, `CADC0874`) + 6 photodiode channels (`CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894`).
- **2,123 univariate segments** total, each expert-labeled *nominal* or *anomalous*. Segment-level labels; onset position within a segment is not given and is estimated via BOCPD (§ Layer 2).
- Per-channel segment counts and anomaly prevalence vary sharply (0% to 78.6%) — see Fig. 5a and Table 1.
- An official train/test partition (`train ∈ {0,1}`, stratified by anomaly rate, ≈75%/25%) is used throughout for train-fit/test-evaluate verification.
- Segment boundaries were determined by **manual expert annotation** using ESA's OXI visualization tool; no quantitative cutting rule is documented anywhere in the public materials (§ Layer 3b).
- Not redistributed; `data/download_opssat_ad.sh` fetches it from its Zenodo record.

### SMAP/MSL (external validation)

NASA Soil Moisture Active Passive (SMAP) and Mars Science Laboratory (MSL) spacecraft telemetry, released with the Telemanom benchmark (Hundman et al., KDD 2018).

- **82 channels** (55 SMAP + 27 MSL), each a long continuous time series with **ground-truth onset/offset indices** for each anomaly window, given in `labeled_anomalies.csv` — a structurally different label format from OPS-SAT-AD's whole-segment labels.
- Each `.npy` file contains multiple columns; only the first column (the actual telemetry signal) is used — remaining columns are one-hot-encoded command/telemetry context and are excluded from all `diff`/`diff²` computation (§ External Validation, Stage 0).
- Anomaly window lengths are highly variable (median 120 samples, mean 616, max 4,217), in contrast to OPS-SAT-AD's short, uniformly-windowed segments — this drove a methodological adaptation described in § External Validation.
- We additionally apply a triviality filter motivated by Wu & Keogh's (2021) critique of SMAP/MSL and similar benchmarks (extreme, visually-obvious point outliers are excluded from the primary test; § External Validation, Stage 1).
- Not redistributed; `data/download_smap_msl.sh` fetches the public release.

### SMD (Server Machine Dataset, external validation)

Industrial server telemetry from the OmniAnomaly benchmark (Su et al., KDD 2019).

- **28 machines × 38 dimensions = 1,064 channels**, each with **timestep-level (0/1) ground-truth anomaly labels**.
- Included as a domain-generalization check (complex technological systems beyond spacecraft, per this journal's scope) rather than as a direct replication of the spacecraft-telemetry claim; we treat SMD as *supporting* rather than *primary* generalization evidence, given the domain shift from aerospace instrumentation.
- Not redistributed; `data/download_smd.sh` fetches the public release.

---

## Methodology (OPS-SAT-AD primary pipeline)

### Layer 1 — Signal Estimation

**Goal**: obtain a denoised, well-calibrated state estimate for each channel before any change-point logic is applied, so that downstream detection is not confounded by channel-specific noise characteristics.

**1.1 Channel typing.** Each channel is automatically classified from the empirical distribution of first differences (Δ) computed within nominal (`anomaly = 0`) segments only, using thresholds set from inspection of this dataset's empirical diagnostics (not derived from first principles — and, as § External Validation shows, these thresholds were re-fitted, not reused, when applied to SMAP/MSL and SMD):

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

**1.2 Noise estimation.** `r = Var(Δ)/2`, `q = Var(Δ²)/6`, from pooled nominal-segment differences. A Gaussian-mixture alternative was tried and **rejected** (see prior version's rationale; unchanged).

**1.3 Quantization floor.** For `quantized` channels, `step²/12` is added to the observation-noise variance.

**1.4 Hierarchical Bayesian shrinkage.** Efron–Morris-type empirical-Bayes shrinkage stabilizes channels with very few nominal points (CADC0886: n=8; CADC0890: n=3).

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

**1.5 State estimation.** A local-linear-trend Kalman filter is fit per channel using the shrunk `r`, `q`. Standardized innovations `z_t = (y_t − ŷ_t)/√S_t` are the sole input to Layer 2.

---

### Layer 2 — Anomaly Detection (change-point model + channel scoping)

**2.1 Change-point model.** Bayesian Online Change-Point Detection (BOCPD; Adams & MacKay 2007) on `z_t`, Normal-Inverse-Gamma conjugate predictive model, hazard rate `1/250`, with a **forgetting** extension (`κ_max = 40`) preventing over-confidence during long nominal stretches.

**2.2 Full 2×2 ablation.**

| Combination | Train macro-MCC (6 scoring channels) |
|---|---|
| mixture=False, forgetting=True (**locked**) | **0.308** |
| mixture=True, forgetting=True | 0.275 |
| mixture=False, forgetting=False | 0.238 |
| mixture=True, forgetting=False | 0.199 |

Train-only reselection independently reproduces the same locked combination.

**2.3 Channel scoping.** MCC and Youden's *J* at locked hyperparameters, with exclusion criteria applied in order: structural (zero anomalous segments), underpowered (fewer than 5 anomalous or 5 normal segments), chance-level (MCC ≤ 0). See **Table 1**.

**2.4 Triple verification.** The 5-channel scope was independently re-derived via bootstrap CI, train-fit/test-eval, and test-only bootstrap CI; all three slices agree (Fig. 1b).

**2.5 Problem-channel diagnosis (CADC0890/0892/0894).** Hypothesis B (BOCPD forgetting deficiency) rejected; Hypothesis A (noise-model deficiency) supported for CADC0890/CADC0894 under the mixture estimator; CADC0892 explained by neither — extreme steady-state kurtosis (63.32) drives spurious resets unrelated to either mechanism.

---

### Layer 3 — Signature Analysis

**Goal**: determine what changes at an OPS-SAT-AD anomaly onset, and rule out that the answer is an artifact of labeling, channel choice, or data-split leakage. All Layer 3 analyses operate on the 5 scoped channels.

**3.1 Onset-uncertainty propagation (Monte Carlo).** 200 onset candidates per segment (seed=42); pre/post comparisons for `level` (Cohen's *d*), `diff`, `diff²` (log-variance-ratio proxy on squared values). 178/386 anomalous segments (46.1%) yield a scoreable window; 99.4% high-reliability tier.

Within-segment, `level` is significant in 54–92% of segments per channel, while `diff`/`diff²` are significant in only 2–76%. Taken alone this looks like `level` is more important — the opposite of the precedence-test conclusion below. Resolved in §3.3.

**3.2 Temporal precedence test.** Canonical onset (median of 200 MC draws) defines pre/post split; first post-onset |z|>3 crossing time recorded per series.

| Comparison | n pairs | *p* (pooled Wilcoxon) | Result |
|---|---|---|---|
| `level` vs. `diff` | 70 | 1.4×10⁻⁵ | `diff` precedes `level` in **84.3%** of pairs |
| `level` vs. `diff²` | 70 | 3.2×10⁻⁵ | `diff²` precedes `level` in **82.9%** of pairs |
| `diff` vs. `diff²` | 161 | 0.368 (n.s.) | no reliable ordering (resolved via cross-correlation lag, §3.4) |

This is inconsistent with the working hypothesis `level → diff → diff²`.

**3.3 Reconciling §3.1 and §3.2.**
- **(a) Temporal-profile classification**: `diff` transient in 98% of segments, `diff²` in 94%, `level` near-even mix (50%/45%).
- **(b) Early-window sensitivity**: `level`'s significance rate drops 0.864→0.633 (early window), a larger relative loss than `diff`/`diff²`.
- **(c) Normal-segment control.** Repeated on normal segments with resampled pivots (mean position ratio ≈0.569). Within OPS-SAT-AD, the precedence ordering **reverses**: `level` precedes `diff` in 55.8% of normal pairs (vs. 15.7% anomalous) and precedes `diff²` in 74.3% (vs. 17.1%). This is strong internal evidence against a "differencing is just always faster" alternative explanation for OPS-SAT-AD specifically. **This reversal does not replicate in SMAP/MSL or SMD — see § External Validation, Stage 2, and § Limitations.**

**3.4 Formal group comparison and diff-vs-diff² tie resolution.** Two-proportion *z*/Fisher exact tests and channel-stratified CMH test confirm orderings after stratification (CMH p≤4.33×10⁻¹⁵). Diff-vs-diff² tie resolved via cross-correlation lag: anomalous-segment lag differs from zero (p=0.00094) but anomalous-vs-normal lag distributions do not differ (p=0.906) — no reliable diff/diff² ordering in either regime.

**3.5 Random-effects meta-analysis (DerSimonian–Laird, k=5 channels).** *I²* ranges 65.7%–97.2% (Table 4), arguing against a single channel-agnostic pooled effect size.

**3.6 Leave-one-out (LOO) decomposition.** Removing CADC0872 eliminates all `level`-|*d*| heterogeneity; removing CADC0874 eliminates all `diff²`-|*d*| heterogeneity; removing CADC0894 nearly halves pooled diff/diff² significance-fraction without evidence of a categorically separate population.

**3.7 Quasi-experimental placebo-pool comparison.** Each anomalous segment's canonical-onset effect compared, via one-sided Mann–Whitney *U*, against a placebo pool of resampled pivots on normal segments (15 tests, Bonferroni-corrected). Within OPS-SAT-AD: `level` indistinguishable from ordinary drift in **all 5 channels** (p_bonf=1.000 uniformly); `diff`/`diff²` significant in **all 5 channels** (Table 3). `diff`/`diff²` strength is markedly higher in the three `float_noise_suspect` magnetometer channels (0.78–0.89 significant-segment fraction) than the two `quantized` photodiode channels (0.12–0.42).

**Interpretive consequence (OPS-SAT-AD-scoped).** Within OPS-SAT-AD specifically, the `level` signal that looked significant within-segment (§3.1) is consistent with anomaly-non-specific channel drift, while the `diff`/`diff²` variance surge is not reproduced in the placebo pool and functions as an anomaly-specific signature. **We emphasize that this strong reading is scoped to OPS-SAT-AD; § External Validation reports the weaker, cross-dataset-consistent version of this claim.**

**Structural skeleton (Fig. 4a, "Path A", OPS-SAT-AD-scoped)**:

```
Onset (BOCPD confirmed changepoint)
      │
      ▼
Diff/Diff² variance surge (transient, onset-localized; channel-moderated within OPS-SAT-AD:
                            float_noise_suspect(872/873/874) strong > quantized(888/894) weaker)
      │
      ▼  (weak / dataset-conditional — see § External Validation)
Level shift (persistent in ~50% of segments; indistinguishable from ordinary channel
             drift within OPS-SAT-AD, but not fully indistinguishable in SMAP/MSL or SMD)
```

---

### Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification

**3b.1 Labeling-protocol audit.** §3's normal-segment control found anomalous-segment onsets skewed toward the back half of their segment (mean position ratio ≈0.569). No documented quantitative segment-cutting rule was found in the primary publication, its SoftwareX companion, or either preprint. Segments were cut entirely manually via ESA's OXI tool. This is an "undecidable" finding: because boundary placement was subjective, the onset-position skew more plausibly reflects idiosyncratic manual judgment than a systematic labeling artifact, though this cannot be fully excluded.

**3b.2 Train-only and test-only reproduction of §3.7.**

| Slice | Coverage | Result |
|---|---|---|
| Train-only | 178,504/240,979 rows (74.1%); 126 anomalous segments | 15/15 (channel×feature) significance calls match full-data exactly |
| Test-only | 62,475/240,979 rows (25.9%); 51 anomalous segments | 12/13 decidable calls match; 1 borderline (CADC0888 `diff`, p=0.051, monotone power loss); 2 undecidable (CADC0894, n=4<5) |

**12 of 13 decidable comparisons agree exactly** across full/train/test slices (Fig. 3b). The single disagreement's p-value rises monotonically with shrinking sample rather than reversing direction, consistent with a power artifact.

---

### Model Cross-Validation (16 independent architectures)

**Goal**: the strongest available test of whether "diff/diff² matter more than level" is a property of OPS-SAT-AD's data or an artifact of the BOCPD/Kalman detector's own inductive bias.

**Task and labels.** Identical binary classification problem to §3.7's design: canonical-onset anomalous segments vs. resampled placebo pivots. 4,996 tabular rows (168 positive/4,828 negative), 2,733 sequence-window rows.

**Two independent data representations.**
- *Tabular* (10 models): §3.7's three engineered features. Attribution: SHAP / permutation importance.
- *Raw sequence* (6 models): per-window z-normalized 3-channel time series, no human-engineered summary statistic. Attribution: Integrated Gradients.

**Model zoo — 16 models, 13 inductive-bias families:**

| Family | Models | Representation |
|---|---|---|
| Linear / Linear (sparse) | LogReg (L2), LogReg (L1) | tabular |
| Probabilistic | GaussianNB | tabular |
| Instance-based | kNN | tabular |
| Kernel | SVM (RBF) | tabular |
| Tree ensemble (bagging / bagging, extra / boosting) | RandomForest / ExtraTrees / GradBoost, XGBoost, LightGBM | tabular |
| Convolutional / (dilated causal) | CNN1D / TCN | sequence |
| Recurrent | BiLSTM, BiGRU | sequence |
| Attention | TinyTransformer | sequence |
| State-space (SSM) | LightMamba (pure-PyTorch substitute) | sequence |

A 17th model (MLP_shallow) was excluded after AUC=0.403 (below chance, failed optimization). Headline statistics are over **16 models**.

**Result: 16/16 models rank `diff`+`diff²` above `level`. Zero of 16 rank `level` as top feature** (binomial p=.0015). Kendall's W=0.609 (χ²=19.50, df=2, p<.001). Bootstrap stability (200 resamples): 100% for RandomForest and LogisticRegression. Tabular (n=10) and sequence (n=6) representations agree on 13/13 families. The State-space (SSM) family shows the narrowest margin (level=0.428), attributed to the lightweight non-CUDA substitute block used here.

**Scope note.** This model-diversity check was run only on OPS-SAT-AD. A reduced replication (2–3 representative model families) on SMAP/MSL or SMD is a natural extension and is listed as future work (§ Limitations) rather than claimed here.

---

## External Validation / Generalization Study (SMAP/MSL, SMD)

### Motivation and positioning

The OPS-SAT-AD analysis above (§ Layer 3) establishes a strong, multiply-corroborated causal-signature claim *within OPS-SAT-AD*. Because a single-benchmark finding cannot rule out that the result reflects OPS-SAT-AD's specific instrument physics (quantized photodiodes, float-noise-suspect magnetometers) rather than a general property of anomaly onsets, we independently re-ran the two most decisive tests from § Layer 3 — the placebo-pool comparison (§3.7) and the temporal-precedence test with normal-segment control (§3.2–3.3c) — on SMAP/MSL and SMD. We treat this as a confirmatory replication study with a pre-specified core hypothesis (diff/diff² > level, direction), not as a re-run of the full OPS-SAT-AD pipeline; the full 16-model cross-validation and labeling-protocol audit were not repeated externally (§ Limitations).

### Structural adaptations required by the label formats (Stage 0)

Unlike OPS-SAT-AD's whole-segment labels, SMAP/MSL and SMD provide **ground-truth onset locations** (window start/end indices, or timestep-level 0/1 labels respectively). This removes the need for OPS-SAT-AD's Monte Carlo onset-uncertainty propagation (§3.1) — we treat this as a methodological strength of the external validation, not a limitation: it eliminates one source of onset-location uncertainty that was a potential confound in OPS-SAT-AD.

Three dataset-specific adaptations were required and are documented in full in `docs/dev-log/step_external_validation.md`:

- **SMAP/MSL command-column exclusion.** Each `.npy` file contains multiple columns; `diff`/`diff²` are computed only on the first (telemetry) column, excluding one-hot-encoded command context.
- **Channel re-typing.** The OPS-SAT-AD quantized/float_noise_suspect/continuous thresholds were **re-fitted**, not reused, on each dataset's own nominal-segment difference distribution (per the original scope caveat in § Layer 1.1).
- **Multiple-comparisons correction.** SMAP/MSL (82 channels) and SMD (1,064 channels) make per-channel Bonferroni correction, as used in OPS-SAT-AD (15 tests), inappropriately conservative. We use Benjamini–Hochberg FDR, with **confirmatory** (dataset-pooled, 3 tests) and **exploratory** (channel-type-stratified, up to 9 tests) families corrected separately, since the latter is nested within and not independent of the former.
- **Triviality filtering.** Following Wu & Keogh's (2021) critique that some published anomaly-detection benchmarks contain visually trivial point outliers, we exclude anomaly windows whose values exceed 20 nominal-standard-deviations from the primary test (`is_trivial_anomaly`, § code).
- **Underpowered per-channel testing.** SMAP/MSL has a median of only 1 anomaly window per channel (max 3), making per-channel Mann–Whitney tests structurally underpowered (0/66 channels reached the n≥5 minimum in an initial per-channel attempt). We therefore report **dataset-pooled and channel-type-stratified** tests as the primary external-validation result, with per-channel counts reported for SMD where sample size permits (channel-level: 97/1038, 142/1038, 145/1038 channels significant for level/diff/diff² respectively).

### Stage 1 — Placebo-pool comparison (external replication of §3.7)

| Dataset | Scope | `level` | `diff` | `diff²` |
|---|---|---|---|---|
| **OPS-SAT-AD** | channel-level (n=5) | 0/5 significant (p_bonf=1.000 uniformly) | 5/5 significant (all p<2×10⁻⁴) | 5/5 significant (all p<2×10⁻⁴) |
| **SMAP/MSL** | dataset-pooled (n_anom=88 windows) | significant (p_fdr=0.031) | significant (p_fdr=4.7×10⁻⁹) | significant (p_fdr=1.3×10⁻⁶) |
| **SMD** | dataset-pooled (n_anom=11,493 windows) | significant (p_fdr=8×10⁻³⁸) | significant (p_fdr=3.7×10⁻¹⁶⁷) | significant (p_fdr=1.4×10⁻¹⁶⁶) |

**Direction is preserved in all three datasets** — `diff`/`diff²` clear placebo by dramatically larger margins than `level` in every case (differences of 7–160+ orders of magnitude in p-value). **Strict specificity (level = fully null) is not preserved**: `level` is weakly but reliably significant above placebo in both external datasets.

Channel-type-stratified results (exploratory family, separately FDR-corrected) show a pattern **partially** consistent with OPS-SAT-AD's moderator finding: in SMAP/MSL, the `continuous` channel type shows `level` n.s. (p_fdr=0.35) with `diff`/`diff²` significant — the OPS-SAT-AD pattern exactly — while the `quantized` type shows `level` weakly significant. The `float_noise_suspect` type could not be tested in SMAP/MSL (only 1/82 channels, 3 anomaly windows — structurally underpowered) and showed no significant effect for any feature in SMD (this type also had the fewest anomaly windows in SMD, 162 across 17 channels).

### Stage 2 — Temporal precedence and normal-segment control (external replication of §3.2–3.3c)

This is the most consequential external-validation result and is reported here without softening.

**Methodological note.** An initial attempt to increase statistical power by sliding multiple onset candidates within each SMAP/MSL anomaly window (analogous in spirit to OPS-SAT-AD's Monte Carlo onset propagation) was found, on diagnosis, to contaminate the pre-onset baseline whenever the anomaly window (median 120 samples) exceeded the baseline length (20 samples) — offset-shifted "onset candidates" ended up partially inside the anomaly itself. This was caught via a paired jointly-valid-candidate check and a window-length stratification test, both of which are documented in `docs/dev-log/step_external_validation.md`. The results below use the corrected methodology: the **ground-truth onset only**, with an **adaptive (non-sliding) post-onset search window** whose upper bound was separately confirmed, via a sensitivity sweep from 150 to 5,000 samples, not to introduce censoring bias (n and effect direction stable across a 33× range of window-cap values).

| Dataset | Regime | n | `diff` precedes `level` | *p* |
|---|---|---|---|---|
| **OPS-SAT-AD** | anomalous | 70 | 84.3% | 1.4×10⁻⁵ |
| **OPS-SAT-AD** | normal control | 224 | 44.2% (**level** precedes in 55.8%) | — |
| **SMAP/MSL** | anomalous | 40 | 67.5% | 0.223 (n.s.) |
| **SMAP/MSL** | normal control | 7,596 | 56.5% | 1.06×10⁻¹²⁴ |
| **SMD** | anomalous | 3,612 | 76.6% | 2.9×10⁻²⁵ |
| **SMD** | normal control | 7,304 | 72.7% | 1.67×10⁻⁴⁴ |

**Interpretation.** In OPS-SAT-AD, the ordering **reverses** between anomalous and normal regimes — the decisive evidence, within that dataset, against a "differencing is mathematically always faster" alternative explanation. In both SMAP/MSL and SMD, **no reversal occurs**: `diff` precedes `level` in both regimes, at broadly similar rates. In SMD, both regimes are strongly statistically significant with a consistent direction. In SMAP/MSL, the anomalous-regime result itself does not reach significance (p=0.223) — we attribute this specifically to the small number of anomaly windows that survive the triple requirement that `level`, `diff`, and `diff²` all cross threshold within the same window (40 of 88 non-trivial windows; confirmed not to be a search-window-cap artifact, see above), not to a reversed or contradictory finding. We therefore describe the SMAP/MSL anomalous-regime result as **underpowered and undecided**, distinct from SMD's result, which we describe as a **non-replication** of the reversal.

Taken together, this is the single external-validation result most in tension with the strongest form of the OPS-SAT-AD causal-signature claim, and is consistent with an alternative explanation under which differencing operators respond faster to *any* local disturbance as a general property of the operator, not specifically because of anomaly onsets. We do not attempt to explain this discrepancy away; we report it as a boundary condition on the claim (§ Limitations, § Relationship to formal causal inference).

### Summary of what generalizes and what does not

| | OPS-SAT-AD | SMAP/MSL | SMD |
|---|---|---|---|
| diff/diff² more strongly anomaly-associated than level (direction) | ✓ | ✓ | ✓ |
| level fully indistinguishable from placebo (strict specificity) | ✓ (0/5, p=1.000) | ✗ (weakly significant) | ✗ (significant) |
| Precedence ordering reverses on normal-segment control | ✓ (55.8–74.3% vs. 15.7–17.1%) | ✗ (no reversal; anomalous n.s.) | ✗ (no reversal; both significant, same direction) |
| float_noise_suspect > quantized moderator pattern | ✓ | Undecidable (n<10) | Not confirmed (n.s. for this type) |

Full per-dataset artifacts, including the pooled/stratified test outputs and the sensitivity-analysis log, are in `results/external_validation/`.

---

## Comparison against published OPS-SAT-AD baselines

*[This section situates the Layer 2 channel-level detection metrics (Table 1: MCC 0.19–0.74, Youden's J 0.20–0.69 across the 5 scoped channels) against detection performance reported in the primary OPS-SAT-AD publication (Ruszczak, Kotowski, Evans & Nalepa, Scientific Data 12:710, 2025) and any subsequent published OPS-SAT-AD benchmark results, using the same official train/test split. As emphasized throughout this README, this paper's contribution is a signature-identification finding, not a state-of-the-art detector claim; nonetheless, this comparison is necessary to situate the BOCPD/Kalman-filter pipeline's detection performance credibly for readers, and is completed prior to submission using the exact published numbers from the source literature, not approximated or estimated. — TODO for camera-ready: insert the completed comparison table here, sourced from the original publication's reported metrics, before submission.]*

---

## Practical implications for alarm design

The signature-identification findings above have direct implications for how spacecraft telemetry alarm systems should be designed, independent of the strength of the causal claim in any given regime:

- **Feature choice for thresholding.** Across all three datasets tested, a detector that thresholds on `diff`/`diff²` variance surges captures anomaly-relevant information that a `level`-only threshold misses or captures with substantially lower statistical power (§ Layer 3.7, § External Validation). This holds even in datasets (SMAP/MSL, SMD) where `level` retains some residual discriminative signal — the practical recommendation (favor derivative-based features, at minimum as a complement to level-based thresholds) is robust to the stronger/weaker specificity distinction that motivates most of this README's caveats.
- **Channel-type-dependent sensitivity.** Within OPS-SAT-AD, `diff`/`diff²`-based alarms should be tuned with higher sensitivity for `float_noise_suspect` (continuous-sensor) channels and lower sensitivity — or supplementary `level`-based corroboration — for `quantized` channels, where the derivative signature is markedly weaker (Table 3, § Layer 3.7). This channel-type moderator could not be confirmed externally (§ External Validation) and should be treated as an OPS-SAT-AD-specific tuning heuristic pending further validation on additional instrument types.
- **False-alarm trade-offs.** The Layer 2 channel-scoping results (Table 1) already surface a concrete example of this trade-off in practice: `CADC0892` and `CADC0894` combine high nominal recall with high false-alarm rates once BOCPD is applied naively, driven by extreme steady-state kurtosis rather than a level/derivative feature-choice issue (§2.5) — illustrating that feature choice alone does not resolve all detector-tuning challenges and must be paired with channel-specific noise modeling.
- **Relationship to risk-aware alarm design (Paper 2).** This paper addresses *what* signal to threshold; the companion paper ([`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization)) addresses *how* to set that threshold under an explicit risk criterion (CVaR-based, with conformal calibration), consuming this repository's `onset_posteriors.parquet` and `channel_scope.json` artifacts directly. Readers interested in an end-to-end alarm-design recommendation should read the two papers together; this repository is deliberately scoped to the signature-identification question alone.

---

## Relationship to formal causal inference — scope statement

We use the term **"causal signature"** in this repository to describe a claim of the form *"X is the feature that changes at anomaly onset, and this is not explainable by normal-operation drift or by dataset/model artifacts."* Within OPS-SAT-AD, this is established via quasi-experimental design (same-channel placebo-pool comparison), temporal precedence testing with a normal-segment control, channel-stratified confirmation (CMH), train/test/bootstrap triple verification, convergent evidence from 16 architecturally independent classifiers, and a structural causal model (SCM) skeleton built from the sensor's signal-processing structure (Fig. 4a) rather than from a data-driven causal discovery algorithm.

We do **not** claim formal statistical identification in the Pearl do-calculus sense, and we do not run a discovery algorithm (e.g., PC, FCI, LiNGAM) to derive the SCM edges — the skeleton is asserted from known instrument physics and then **tested against data** for consistency, not derived from data.

**The external-validation results (§ External Validation) require us to narrow this claim further, and explicitly.** The normal-segment-control reversal — within OPS-SAT-AD, the single strongest piece of evidence against a "differencing is mathematically always faster" alternative explanation — does **not** replicate in SMAP/MSL or SMD. In both external datasets, `diff` precedes `level` in both anomalous and normal regimes, a pattern equally consistent with the alternative explanation the OPS-SAT-AD reversal was designed to rule out. Accordingly:

- **Within OPS-SAT-AD**, the full body of evidence (placebo-pool specificity, precedence-ordering reversal, 16-model convergence, train/test triple verification) supports treating the diff/diff² signature as anomaly-specific in the sense defined above.
- **Outside OPS-SAT-AD**, we can only responsibly claim that diff/diff² carry more anomaly-discriminative information than level — a weaker, purely associational claim, still useful for detector design (§ Practical Implications), but which should not be read as evidence of a universal causal mechanism linking anomaly onsets to derivative-statistic surges.

We regard this narrowing as a strength of the study design, not a weakness of the underlying finding: a claim that is tested against external data and revised accordingly is more trustworthy than one that is asserted from a single benchmark and left untested.

---

## Figures

All figures below are generated by the scripts in `results/figures/` and are byte-for-byte what running each script produces from `results/*.csv` / `results/*.json`. Every figure is designed to be fully legible in black-and-white print: categories are encoded redundantly by greyscale value *and* hatch pattern.

*(Figures 1–5 below reproduce the OPS-SAT-AD primary-pipeline results exactly as in the original analysis; captions have been lightly edited only where necessary to flag that the underlying claim is OPS-SAT-AD-scoped, with a pointer to § External Validation.)*

### Figure 1 — Channel scoping: MCC/Youden's J and the 9→5 funnel

![Figure 1: Channel scoping](results/figures/fig01_channel_scope.png)

*(a) MCC and Youden's J per channel at the locked hyperparameters (`mixture=False, forgetting=True`). Blue/teal bars = channels retained in the final scope; grey bars = excluded, with exclusion reason printed beneath each tick label. `CADC0890` has the highest point-estimate MCC (0.826) of any channel yet is excluded on sample-size grounds (n=11/14), not weak performance. (b) The 9→5 channel-scoping funnel, confirmed identical across three independent re-derivations.*
**Reproduces from:** `results/layer2/channel_scope.json`, `results/layer2/bootstrap_ci.csv` — **Script:** `results/figures/fig01_channel_scope.py`

### Figure 2 — 16-model cross-validation: feature attribution and agreement (OPS-SAT-AD only)

![Figure 2: Model cross-validation](results/figures/fig02_model_cross_validation.png)

*(a) Normalized feature-importance share (level/diff/diff²) for each of 16 independently trained models, sorted by out-of-fold AUC. Every model allocates substantially more than chance (1/3) share to diff+diff². (b) Agreement-statistics panel. This model-diversity result was obtained on OPS-SAT-AD only; it was not re-run on SMAP/MSL or SMD (§ Limitations).*
**Reproduces from:** `results/model_cross_validation/feature_attribution_16models.csv`, `results/model_cross_validation/agreement_statistics.json` — **Script:** `results/figures/fig02_model_cross_validation.py`

### Figure 3 — Quasi-experimental placebo-pool test and triple verification (OPS-SAT-AD)

![Figure 3: Quasi-experimental comparison](results/figures/fig03_quasi_experimental.png)

*(a) −log₁₀(p_bonferroni), OPS-SAT-AD placebo-pool test. `level` bars are exactly zero-height in every channel. See § External Validation Stage 1 (and its summary table) for the corresponding, weaker result in SMAP/MSL and SMD. (b) Full/train-only/test-only agreement matrix, OPS-SAT-AD-internal.*
**Reproduces from:** `results/layer3/placebo_comparison.csv`, `results/layer3/triple_verification_matrix.csv` — **Script:** `results/figures/fig03_quasi_experimental.py`

### Figure 4 — Structural skeleton and meta-analytic heterogeneity (OPS-SAT-AD)

![Figure 4: SCM and heterogeneity](results/figures/fig04_scm_and_heterogeneity.png)

*(a) The structural skeleton for OPS-SAT-AD ("Path A"): onset drives a transient diff/diff² variance surge; level shift is excluded from the OPS-SAT-AD-internal causal core, with the channel-type moderator finding noted in a bordered annotation box. This moderator pattern was not confirmed externally (§ External Validation). (b) DerSimonian–Laird heterogeneity (*I²*) across the 5 scoped OPS-SAT-AD channels.*
**Reproduces from:** `results/layer3/heterogeneity_summary.csv` — **Script:** `results/figures/fig04_scm_and_heterogeneity.py`

### Figure 5 — Dataset overview and pooled temporal-precedence test (OPS-SAT-AD)

![Figure 5: Dataset and precedence](results/figures/fig05_dataset_and_precedence.png)

*(a) OPS-SAT-AD dataset composition. (b) The pooled temporal-precedence test as −log₁₀(p), OPS-SAT-AD only. The corresponding SMAP/MSL and SMD precedence results, including the non-replication of the normal-control reversal, are reported numerically in § External Validation Stage 2 rather than in a matching figure here (a combined three-dataset figure is a planned extension, see `docs/METHODS_SUPPLEMENT.md`).*
**Reproduces from:** `data/raw/` (segment metadata), `results/layer3/temporal_precedence.csv` — **Script:** `results/figures/fig05_dataset_and_precedence.py`

---

## Results tables

*(Tables 1–11 below are unchanged from the OPS-SAT-AD primary analysis; see § External Validation for the corresponding SMAP/MSL and SMD tables, which are reported inline in that section rather than duplicated here as numbered tables.)*

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

**Agreement statistics**: Kendall's W=0.609 (χ²=19.50, df=2, p<.001) · models ranking `level` top = 0/16 (binomial p=.0015) · bootstrap stability (200 resamples): RandomForest 100%, LogReg 100% · 13/13 families favor diff+diff² over level · tabular (n=10) and sequence (n=6) representations agree.

### Table 3 — Quasi-experimental placebo-pool comparison (OPS-SAT-AD; underlies Fig. 3a)

| Channel | level, p_bonf | diff, p_bonf | diff², p_bonf |
|---|---|---|---|
| CADC0872 | 1.000 | 7.94×10⁻²⁴ | 1.12×10⁻²² |
| CADC0873 | 1.000 | 2.40×10⁻¹⁷ | 3.09×10⁻¹⁶ |
| CADC0874 | 1.000 | 5.37×10⁻¹⁶ | 2.88×10⁻²² |
| CADC0888 | 1.000 | 6.92×10⁻⁶ | 3.24×10⁻¹³ |
| CADC0894 | 1.000 | 3.89×10⁻⁶ | 1.95×10⁻⁴ |

*See § External Validation Stage 1 for the corresponding SMAP/MSL and SMD results (weaker `level` specificity).*

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
| Models ranking `level` top | 0/16 |
| Binomial test | p=.0015 |
| Bootstrap stability, RandomForest | 100% favor diff+diff² > level |
| Bootstrap stability, LogisticRegression | 100% favor diff+diff² > level |

### Table 7 — Family-level aggregation (13 inductive-bias families, OPS-SAT-AD)

| Family | n models | mean level | mean diff | mean diff² | diff+diff² > level |
|---|---|---|---|---|---|
| Linear | 1 | 0.060 | 0.407 | 0.534 | Yes |
| Linear (sparse) | 1 | 0.057 | 0.412 | 0.531 | Yes |
| Probabilistic | 1 | 0.051 | 0.446 | 0.503 | Yes |
| Instance-based | 1 | 0.129 | 0.391 | 0.480 | Yes |
| Kernel | 1 | 0.061 | 0.466 | 0.472 | Yes |
| Tree ensemble (bagging) | 1 | 0.084 | 0.456 | 0.460 | Yes |
| Tree ensemble (bagging, extra) | 1 | 0.145 | 0.389 | 0.466 | Yes |
| Tree ensemble (boosting) | 3 | 0.149 | 0.416 | 0.434 | Yes |
| Convolutional | 1 | 0.303 | 0.301 | 0.396 | Yes |
| Convolutional (dilated causal) | 1 | 0.200 | 0.380 | 0.419 | Yes |
| Recurrent | 2 | 0.248 | 0.443 | 0.310 | Yes |
| Attention | 1 | 0.178 | 0.438 | 0.384 | Yes |
| State-space (SSM) | 1 | 0.428 | 0.441 | 0.130 | Yes (narrowest margin) |

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
| diff vs. diff² | 161 | 0.368 (n.s.) | no reliable ordering |

*See § External Validation Stage 2 for the SMAP/MSL and SMD equivalents (no reversal on normal control).*

### Table 10 — Normal-segment control: precedence ordering reversal (OPS-SAT-AD only)

| Comparison | Regime | n pairs | frac(first precedes second) |
|---|---|---|---|
| level vs. diff | anomalous | 70 | level 15.7% / diff **84.3%** |
| level vs. diff | normal | 224 | level **55.8%** / diff 44.2% |
| level vs. diff² | anomalous | 70 | level 17.1% / diff² **82.9%** |
| level vs. diff² | normal | 171 | level **74.3%** / diff² 25.7% |
| diff vs. diff² | anomalous | 161 | diff 7.5% / diff² **92.5%** |
| diff vs. diff² | normal | 211 | diff **64.5%** / diff² 35.5% |

*This reversal pattern does not replicate in SMAP/MSL or SMD (§ External Validation Stage 2).*

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

---

## Reproducing every number in this README

```bash
# 1. Clone and set up the environment
git clone https://github.com/USERNAME/opssat-ad-detection-causal.git
cd opssat-ad-detection-causal
conda env create -f environment.yml && conda activate opssat-ad-causal   # or: pip install -r requirements.txt

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

# 5. Run the independent 16-model cross-validation benchmark (OPS-SAT-AD)
python -m model_cross_validation.run_all

# 6. Run the external-validation pipeline (SMAP/MSL, SMD)
python -m external_validation.run_external_validation

# 7. Regenerate every figure in this README from the results/ artifacts above
python results/figures/fig01_channel_scope.py
python results/figures/fig02_model_cross_validation.py
python results/figures/fig03_quasi_experimental.py
python results/figures/fig04_scm_and_heterogeneity.py
python results/figures/fig05_dataset_and_precedence.py

# ...or simply:
make all
```

`REPRODUCIBILITY_CHECKLIST.md` documents random seeds, package versions, and expected runtime (CPU-only: ~40 min for OPS-SAT-AD Layers 1–3; the 16-model benchmark is the long pole at ~2–3 hours on CPU / ~25 min on a single GPU; the external-validation pipeline runs in ~15–30 min on CPU, dominated by SMD's 1,064-channel loop).

---

## Threats to validity and how each was addressed

| Threat | Mitigation |
|---|---|
| **Train/test leakage in channel selection (OPS-SAT-AD)** | Channel-scoping procedure independently re-run on train-only and test-only splits; identical 5-channel scope in all three passes. |
| **Train/test leakage in the placebo-pool result (OPS-SAT-AD)** | §3.7 procedure independently re-run on train-only/test-only slices; 12/13 decidable comparisons agree exactly, the disagreement diagnosed as a power artifact. |
| **Diff/diff² result being a labeling artifact (OPS-SAT-AD)** | External audit of the labeling protocol: no documented cutting rule exists, limiting but not fully excluding this possibility. |
| **Diff/diff² result being an artifact of the detector's inductive bias (OPS-SAT-AD)** | Independent 16-model cross-validation across 13 families, 2 representations, 2 attribution methods — all converge. |
| **Level shift "looking" important only because it's not tested against the right null (OPS-SAT-AD)** | Quasi-experimental placebo-pool design against same-channel normal segments. |
| **`level`'s within-segment significance contradicting the precedence-test result (OPS-SAT-AD)** | Resolved via transient-vs-persistent profile classification, early-window sensitivity, normal-segment control. |
| **Small-sample channels inflating apparent effect sizes (OPS-SAT-AD)** | Structural/underpowered channels explicitly excluded rather than down-weighted; `CADC0890`'s exclusion despite the highest MCC of any channel is documented as a deliberate conservative choice. |
| **Between-channel heterogeneity being treated as noise (OPS-SAT-AD)** | Quantified via I²/Q-test, tested via dual-criterion LOO sensitivity, shown to be moderated by a known covariate. |
| **Multiple-comparisons inflation (OPS-SAT-AD)** | Bonferroni applied to all 15 placebo-pool tests and their train/test replications; pooled (not per-test) Wilcoxon for the precedence claim. |
| **Result being an artifact of OPS-SAT-AD's specific instrument physics rather than a general property** | Independent replication on SMAP/MSL and SMD (§ External Validation). Direction confirmed in 3/3 datasets; strict specificity and precedence-reversal claims confirmed only in OPS-SAT-AD — reported transparently, not treated as resolved. |
| **Multiple-comparisons inflation across many channels in the external datasets (SMAP/MSL, SMD)** | Benjamini–Hochberg FDR (appropriate for 66–1,038 tests, vs. Bonferroni for OPS-SAT-AD's 15), with confirmatory and exploratory test families corrected separately to respect their nested, non-independent structure. |
| **Underpowered per-channel testing in SMAP/MSL (median 1 anomaly window/channel)** | Dataset-pooled and channel-type-stratified tests used as the primary external-validation design, rather than per-channel tests as in OPS-SAT-AD. |
| **Sliding-window onset refinement contaminating the pre-onset baseline (SMAP/MSL, caught during analysis)** | Diagnosed via paired jointly-valid-candidate and window-length-stratification checks; discarded in favor of ground-truth-onset, non-sliding, adaptive-window methodology; censoring-bias risk from the adaptive window's upper bound separately ruled out via a 150–5,000-sample sensitivity sweep. Full trail in `docs/dev-log/step_external_validation.md`. |
| **Visually trivial point outliers inflating apparent effect sizes in SMAP/MSL** | Triviality filter (>20 nominal-SD exclusion), motivated by Wu & Keogh (2021)'s critique of related benchmarks, applied before the primary external-validation tests. |

---

## Limitations

- **The strict specificity claim (`level` fully indistinguishable from placebo drift) and the normal-segment-control reversal are OPS-SAT-AD-specific and do not generalize.** Both hold decisively within OPS-SAT-AD but were **not observed** in either external dataset tested: `level` retains weak but statistically detectable signal above placebo in both SMAP/MSL (p_fdr=0.031) and SMD (p_fdr=8×10⁻³⁸), and the precedence-ordering reversal that constitutes OPS-SAT-AD's strongest internal falsification test does not occur in either external dataset — `diff` precedes `level` in both anomalous and normal regimes in SMAP/MSL and SMD alike. This is consistent with an alternative explanation under which differencing operators respond faster to any local disturbance as a general mathematical property, not specifically because of anomaly onsets. We regard the weaker, direction-only claim — diff/diff² carry more anomaly-discriminative information than level — as the load-bearing, cross-dataset-validated result of this paper, and the stronger specificity/causal reading as conditional on OPS-SAT-AD's instrument physics (§ External Validation, § Relationship to formal causal inference).
- **The channel-type moderator pattern (`float_noise_suspect` > `quantized`) could not be tested in SMAP/MSL** (only 1 of 82 channels falls in this category, with too few anomaly windows for a decidable test) **and was not confirmed in SMD** (this channel type showed no significant effect for any feature, on a comparably thin sample — 162 anomaly windows across 17 channels). We treat this moderator pattern as an open question rather than a confirmed general mechanism.
- **The 16-model cross-validation and labeling-protocol audit were not repeated on SMAP/MSL or SMD.** These two lines of evidence remain OPS-SAT-AD-internal robustness checks; a reduced-scale (2–4 representative model families) cross-validation replication on the external datasets is a natural extension and is planned as future work rather than reported here.
- **No comparison against published OPS-SAT-AD baselines** is included in the current draft (§ Comparison against published OPS-SAT-AD baselines); this is completed prior to submission using the exact metrics reported in the source publication.
- **Detection performance is modest in absolute terms** on some OPS-SAT-AD channels (e.g., CADC0888, CADC0894 sit well below CADC0874). The paper's contribution is about *what the signature is*, not about maximizing raw detection recall/precision.
- **5 of 9 OPS-SAT-AD channels are excluded from the signature analysis** for principled statistical-power reasons (Table 1); the OPS-SAT-AD-scoped claim is limited to these 5 channels.
- **The OPS-SAT-AD labeling-protocol audit is inherently undecidable.** No public documentation specifies a quantitative segment-cutting rule, so a subtle, non-systematic labeling influence on onset-position skew cannot be fully excluded — only shown to be less plausible than under a fixed, rule-based cutting convention.
- **CADC0894's small anomalous-segment count** (n=21 full-data, n=4 in the test-only slice) limits statistical power in every OPS-SAT-AD test-split replication.
- **CADC0874's diff² effect size is likely under-estimated** by the full-window Welch design relative to its especially short-lived transient response; treat the reported meta-analytic estimate for this channel-feature combination as a conservative lower bound.
- **The state-space (Mamba-style) model in the OPS-SAT-AD 16-model benchmark is a lightweight, non-CUDA-kernel substitute** for the official `mamba_ssm` implementation; its narrower agreement margin relative to the other 12 families should not be over-interpreted without confirming the result under the official implementation.
- **SMD's domain (industrial server telemetry) differs substantially from spacecraft instrumentation**; we treat SMD as supporting, domain-generalization evidence rather than a direct replication of the spacecraft-telemetry claim, consistent with how it is weighted throughout § External Validation.
- **The SMAP/MSL anomalous-regime precedence result (§ External Validation, Stage 2) is underpowered (n=40, p=0.223), not merely non-significant by a wide margin**; we describe it as undecided, distinct from SMD's result, which shows the same non-reversal pattern at high statistical power and is better described as a non-replication.

---

## License

Code in this repository is released under the MIT License (see `LICENSE`). This repository does not redistribute any of the three datasets used; each dataset is governed by its own upstream license and terms of use:

- **OPS-SAT-AD**: see the dataset's Zenodo record and `data/README.md` for license terms.
- **SMAP/MSL (Telemanom)**: see the dataset's public release terms and `data/README.md`.
- **SMD (OmniAnomaly)**: see the dataset's public release terms and `data/README.md`.

Users of this repository are responsible for independently confirming that their intended use complies with each dataset's license.
