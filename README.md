# opssat-ad-detection-causal

**Detection and Causal Signature Identification of Telemetry Anomalies in the OPS-SAT-AD Benchmark**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-fully%20reproducible-brightgreen.svg)](#reproducing-every-number-in-this-readme)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](#citation)

> **One-sentence finding.** In the OPS-SAT-AD spacecraft telemetry benchmark, the true signature of an anomaly is **not a level (mean) shift** but a **transient surge in first- and second-difference variance** (`diff`, `diff²`); `level` is statistically indistinguishable from ordinary operational drift (Bonferroni-corrected placebo-pool test, 5/5 channels n.s.), while `diff`/`diff²` are significant in all 5 channels, precede `level` in onset timing (pooled Wilcoxon, p < 10⁻⁴), and are reproducible across full-data/train-only/test-only splits and **16 independently trained model architectures spanning 13 inductive-bias families** (0/16 rank `level` as the top feature; Kendall's W = 0.609, p < .001).

This repository is the code, data-provenance, and figure-reproduction companion to **Paper 1** of a two-paper series built on the OPS-SAT-AD dataset. It is **self-contained and independently evaluable** — it does not depend on any other repository.

| This series | Repository | Status |
|---|---|---|
| **Paper 1 (this repo)** — Detection + Causal Signature | `opssat-ad-detection-causal` | ✅ independent, no upstream deps |
| Paper 2 — Risk-Aware Alarm Design (CVaR / conformal) | [`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization) | depends on this repo's `results/` artifacts |
| DSS prototype (not a paper) | [`opssat-ad-dss`](https://github.com/USERNAME/opssat-ad-dss) | depends on both papers' artifacts |

---

## Table of Contents

1. [Why this repository exists](#why-this-repository-exists)
2. [Headline results](#headline-results)
3. [Repository structure](#repository-structure)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
   - [Layer 1 — Signal Estimation](#layer-1--signal-estimation)
   - [Layer 2 — Anomaly Detection (change-point model + channel scoping)](#layer-2--anomaly-detection-change-point-model--channel-scoping)
   - [Layer 3 — Causal Signature Analysis](#layer-3--causal-signature-analysis)
   - [Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification](#layer-3b--labeling-protocol-audit--traintest-triple-verification)
   - [Model Cross-Validation (16 independent architectures)](#model-cross-validation-16-independent-architectures)
6. [Figures](#figures) (all reproduced inline, with full captions)
7. [Results tables](#results-tables) (11 tables, mirroring every CSV in `results/`)
8. [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
9. [Threats to validity and how each was addressed](#threats-to-validity-and-how-each-was-addressed)
10. [Relationship to formal causal inference — scope statement](#relationship-to-formal-causal-inference--scope-statement)
11. [Limitations](#limitations)
12. [Suggested target journals](#suggested-target-journals)
13. [Citation](#citation)
14. [License](#license)

---

## Why this repository exists

Change-point and anomaly-detection pipelines for spacecraft telemetry almost universally treat a **level shift** (a change in the mean of the raw signal) as the canonical anomaly signature — it is the easiest thing to visualize, the easiest thing to threshold, and the default target of most textbook change-point models. This repository documents a systematic, triple-validated investigation of the **OPS-SAT-AD** benchmark (European Space Agency OPS-SAT mission, 9 telemetry channels, 2,123 expert-labeled univariate segments) that found the opposite is true here: level shifts are **not statistically distinguishable from normal-operation drift** once tested against a same-channel placebo pool, while **variance surges in the first and second differences of the signal are the actual, reproducible fingerprint of an anomaly**.

This is not a claim we set out to prove. The project's early working hypothesis, stated explicitly before any hypothesis testing began, was the naive `level → diff → diff²` causal ordering — that a level shift is the primary event and the derivative statistics are downstream artifacts of it. Section 3.2's pooled Wilcoxon test was run specifically to confirm this ordering and instead falsified it outright: `diff`/`diff²` precede `level` in 82–84% of paired segments, not the other way around. Every subsequent analysis in Layer 3 was designed to stress-test whether that reversal was itself real or an artifact, rather than to reinforce it.

Because a single-pipeline finding is easy to dismiss as an artifact of one detector's inductive bias, the analysis did not stop at the detection pipeline. Four independent lines of attack were run against the core finding, each targeting a different class of possible confound:

1. **Is `level`'s apparent significance just a weaker null?** → same-channel quasi-experimental placebo-pool comparison (Layer 3c).
2. **Is the diff/diff² pattern a side effect of how segments were manually cut?** → external audit of the labeling protocol (Layer 3b).
3. **Is the result an artifact of using the full dataset rather than a proper train/test split?** → independent train-only and test-only reproduction of the entire placebo-pool test (Layer 3b).
4. **Is the result an artifact of the BOCPD/Kalman-filter pipeline's own inductive bias?** → an independent 16-model cross-validation study spanning 13 inductive-bias families and 2 independent data representations, each asked the same question blind to the others and to the statistical machinery of Layer 3.

All four converge on the same answer. This repository packages:

- The full **signal-estimation → change-point-detection → causal-signature** pipeline (Layers 1–3).
- The **labeling-protocol audit and full train/test/bootstrap triple re-verification** of the causal-signature result (Layer 3b).
- The **16-model cross-validation** benchmark and its agreement statistics (Kendall's W, binomial test, bootstrap stability).
- **Every figure in this README regenerated from its exact source script** (`results/figures/*.py`), so every number a reviewer sees can be traced to code, not to a table typed by hand.
- **Raw development logs** (`docs/dev-log/`) documenting the full, unabridged analytical trail — including negative results, ablations that were discarded, the exact point at which each methodological correction (e.g., train/test leakage checks, SHAP return-shape bugs, cuDNN backward-mode fixes) was introduced, and the reasoning behind every locked hyperparameter. These are provided for full transparency and are **not** a substitute for the polished Methods text in the eventual manuscript, which draws on them (`docs/METHODS_SUPPLEMENT.md`).

---

## Headline results

| Claim | Evidence | Section |
|---|---|---|
| Channel scope: 9 → 5 channels, survives 3 independent verifications (full-data bootstrap, train-only reselection, test-only confirmation) | Fig. 1 | [§ Layer 2](#layer-2--anomaly-detection-change-point-model--channel-scoping) |
| `diff`/`diff²` precede `level` in onset timing (pooled Wilcoxon, 5 channels, n = 70–161 pairs); ordering **reverses** on normal-segment controls | Fig. 5b | [§ Layer 3](#layer-3--causal-signature-analysis) |
| `level` is **not** distinguishable from normal-operation drift (quasi-experimental placebo test, 5/5 channels n.s., p_bonf = 1.000 uniformly) | Fig. 3a | [§ Layer 3](#layer-3--causal-signature-analysis) |
| `diff`/`diff²` **are** distinguishable from normal drift (5/5 channels, Bonferroni-corrected, all p < 2×10⁻⁴) | Fig. 3a | [§ Layer 3](#layer-3--causal-signature-analysis) |
| Placebo-pool result replicates independently on full / train-only / test-only splits (12/13 decidable comparisons agree exactly) | Fig. 3b | [§ Layer 3b](#layer-3b--labeling-protocol-audit--traintest-triple-verification) |
| No documented quantitative segment-cutting rule exists in the OPS-SAT-AD labeling protocol (OXI tool); onset-position skew more plausibly reflects idiosyncratic manual judgment than a systematic labeling artifact | — | [§ Layer 3b](#layer-3b--labeling-protocol-audit--traintest-triple-verification) |
| 16/16 independent models (13 inductive-bias families, 2 data representations) rank `diff`+`diff²` over `level`; Kendall's W = 0.609 (p < .001); binomial p = .0015 | Fig. 2 | [§ Model cross-validation](#model-cross-validation-16-independent-architectures) |
| Bootstrap stability of the diff/diff² > level ranking: 100% (RandomForest, 200 resamples), 100% (LogReg, 200 resamples) | Table 6 | [§ Model cross-validation](#model-cross-validation-16-independent-architectures) |
| Channel-level heterogeneity is real and interpretable (I² up to 97.2%, moderated by channel noise/quantization regime, confirmed by leave-one-out sensitivity) | Fig. 4b | [§ Layer 3](#layer-3--causal-signature-analysis) |
| Apparent `level`-significance paradox (86–92% within-segment significance despite losing the precedence race) is resolved: `level` changes are persistent/sustained while `diff`/`diff²` are brief transients diluted by a full-window test | — | [§ Layer 3](#layer-3--causal-signature-analysis) |

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
│   ├── README.md                          # dataset layout, checksum, license notes
│   └── raw/                               # (gitignored) populated by download script
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

> **Note on `docs/dev-log/`.** These files are the original, conversational research logs kept during development. They are included **verbatim** for full analytical transparency (a reviewer or replicator can see every dead end, discarded ablation, and correction as it happened) but are explicitly **not** the manuscript's Methods section — the polished, publication-ready methodology is what appears in this README and in `docs/METHODS_SUPPLEMENT.md`.

---

## Dataset

**OPS-SAT-AD** — European Space Agency's OPS-SAT experimental nanosatellite telemetry anomaly-detection benchmark (Ruszczak, Kotowski, Evans & Nalepa, *Scientific Data* 12:710, 2025).

- **9 channels**: 3 magnetometer channels (`CADC0872`, `CADC0873`, `CADC0874`) + 6 photodiode channels (`CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894`).
- **2,123 univariate segments** total, each expert-labeled as *nominal* or *anomalous*.
- Per-channel segment counts and anomaly prevalence vary sharply (0% to 78.6%) — see Fig. 5a and Table 1.
- An official train/test partition (`train ∈ {0,1}`, stratified by anomaly rate, ≈ 75%/25%) is provided and used throughout for train-fit/test-evaluate verification.
- Every segment belongs to exactly one channel (verified: 0 multi-channel segments across all 2,123 segments and 303,493 raw observations); `anomaly` is internally consistent within every segment (0 mismatches against `dataset.csv`).
- A secondary `label` field (`a2`/`a3`/`a4`/`anomaly`) partitions the normal class further but has no documented definition anywhere in the primary publication or released code, and is not used anywhere downstream, matching the source authors' own practice.
- Segment boundaries were determined by **manual expert annotation** using ESA's OXI visualization tool (see [§ Layer 3b](#layer-3b--labeling-protocol-audit--traintest-triple-verification)); no quantitative cutting rule is documented anywhere in the public materials.
- We do **not** redistribute the raw dataset. `data/download_opssat_ad.sh` fetches it directly from its Zenodo record and lays it out in the directory structure the pipeline expects.

---

## Methodology

### Layer 1 — Signal Estimation

**Goal**: obtain a denoised, well-calibrated state estimate for each channel before any change-point logic is applied, so that downstream detection is not confounded by channel-specific noise characteristics.

**1.1 Channel typing.** Each channel is automatically classified from the empirical distribution of first differences (Δ) computed within nominal (`anomaly = 0`) segments only, using thresholds set from inspection of this dataset's empirical diagnostics (not derived from first principles — should be re-examined for other telemetry sources):

- **`quantized`** if the fraction of exact-zero Δ is ≥ 0.10 (quantization step estimated as the 25th percentile of non-zero |Δ|).
- **`float_noise_suspect`** if not quantized and the smallest non-zero |Δ| is < 1 × 10⁻⁶.
- **`continuous`** otherwise.

| Channel | Type | frac_zero_diff | Quantization step |
|---|---|---|---|
| CADC0872 | float_noise_suspect | 0.053 | n/a |
| CADC0873 | float_noise_suspect | 0.052 | n/a |
| CADC0874 | float_noise_suspect | 0.044 | n/a |
| CADC0884 | quantized | 0.183 | ≈ 0.0144 |
| CADC0886 | quantized | 0.371 | ≈ 0.0274 |
| CADC0888 | quantized | 0.354 | ≈ 0.0144 |
| CADC0890 | continuous | 0.083 | n/a |
| CADC0892 | quantized | 0.416 | ≈ 0.0054 |
| CADC0894 | quantized | 0.583 | ≈ 0.0026 |

**1.2 Noise estimation.** Observation-noise variance `r` and process-noise variance `q` (increments of local level and local trend) are estimated from pooled nominal-segment differences as `r = Var(Δ)/2`, `q = Var(Δ²)/6`. A Gaussian-mixture ("jump probability × jump-size MAD") alternative was tried and **rejected**: applied to the full difference distribution of a zero-inflated (quantized) channel, MAD collapses to zero (the median of a ≥50%-zero sample is zero); restricting MAD to non-zero jumps avoids the collapse but inflates steady-state standardized-innovation SD to 2.25–2.88 (should be ≈1) on CADC0890/CADC0894, versus 0.95–0.98 under the naïve estimator.

**1.3 Quantization floor.** For `quantized` channels, `step²/12` (variance of Uniform(−step/2, step/2)) is added to the observation-noise variance, reflecting the finite resolution of the sensor's A/D conversion.

**1.4 Hierarchical Bayesian shrinkage.** Per-channel `log(r)`/`log(q)` are pooled via an Efron–Morris-type empirical-Bayes shrinkage estimator (inverse-variance-weighted global mean, moment-based between-channel variance τ², per-channel weight τ²/(τ²+SE²)) — stabilizing the two channels with very few nominal points (CADC0886: n=8; CADC0890: n=3).

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

**1.5 State estimation.** A local-linear-trend Kalman filter (state = [level, trend]; transition `[[1,1],[0,1]]`; observation `[1,0]`; `Q = diag(0, q_shrunk)`; `R = r_shrunk + quantization_floor`) is fit per channel using the shrunk `r`, `q`. This is a signal-processing convenience model, not a physical model of spacecraft attitude/orbital dynamics. Standardized innovations `z_t = (y_t − ŷ_t)/√S_t` are the sole input to Layer 2.

---

### Layer 2 — Anomaly Detection (change-point model + channel scoping)

**Goal**: convert the Layer-1 state estimate into a calibrated change-point/anomaly score, and rigorously determine which channels are even analyzable.

**2.1 Change-point model.** Bayesian Online Change-Point Detection (BOCPD; Adams & MacKay 2007) on `z_t`, using a Normal-Inverse-Gamma conjugate predictive model (`μ₀=0, κ₀=1, α₀=1, β₀=1`) and hazard rate `1/250` (constant across channels). A **forgetting** extension caps the run-length sufficient statistic `κ` at `κ_max = 40`, rescaling `κ, α, β` proportionally once exceeded — preventing the vanilla-BOCPD failure mode of over-confidence (and reduced sensitivity to later changepoints) during long nominal stretches. Onset is declared at the first time step where the MAP run-length collapses from >5 to ≤2; the posterior over the top-30 run-lengths at confirmation is retained for Layer 3's Monte Carlo propagation.

**2.2 Full 2×2 ablation.** All four combinations of `{mixture noise estimator: on/off} × {forgetting: on/off}` were exhaustively compared. The two factors act through **disjoint mechanisms on disjoint channel subsets**:

| Combination | Train macro-MCC (6 scoring channels) |
|---|---|
| mixture=False, forgetting=True (**locked**) | **0.308** |
| mixture=True, forgetting=True | 0.275 |
| mixture=False, forgetting=False | 0.238 |
| mixture=True, forgetting=False | 0.199 |

Forgetting is necessary for `float_noise_suspect` channels (disabling it nearly halves CADC0874 recall: 0.725 → 0.319); the noise-estimator choice is necessary for CADC0890/CADC0894 (mixture estimator inverts CADC0890's MCC: 0.826 → −0.213). Train-only reselection independently reproduces the same locked combination, mitigating in-sample-tuning risk from the original full-data comparison.

**2.3 Channel scoping.** Channels are screened using MCC (Chicco & Jurman, *BMC Genomics* 21:6, 2020; 0 under chance regardless of class balance) and Youden's *J*, at the locked hyperparameters, with exclusion criteria applied in order: (1) structural — zero anomalous segments; (2) underpowered — fewer than 5 anomalous or 5 normal segments; (3) chance-level or worse — MCC ≤ 0. See **Table 1** for the full decision table.

**2.4 Triple verification.** The 5-channel scope was independently re-derived three ways — (i) 2,000-resample bootstrap 95% CI on full-data MCC (all 5 included channels exclude 0; CADC0892's CI [−0.274, 0.053] includes 0, reinforcing its exclusion); (ii) train-fit / test-eval (parameters re-estimated on train split only, evaluated on held-out test split); (iii) test-only bootstrap MCC CI. All three slices agree; see Fig. 1b and `results/layer2/`.

**2.5 Problem-channel diagnosis (CADC0890/0892/0894).** Two hypotheses were tested for why these channels retain high false-alarm rates: (A) noise-model deficiency (steady-state SD ≫ 1); (B) BOCPD forgetting-mechanism deficiency. Hypothesis B was **rejected** — forgetting on/off produces negligible change in steady-state SD or FA rate for these channels. Hypothesis A is supported for CADC0890/CADC0894 specifically under the *mixture* estimator (steady-state SD inflates to 2.25/2.88 vs. 0.945/0.978 under the locked naïve estimator) — resolved by the Layer 1 noise-estimator choice, not by re-tuning BOCPD. **CADC0892 is explained by neither hypothesis**: its steady-state SD is close to 1 under either estimator, yet 99.4% of its normal segments (176/177) trigger at least one spurious run-length reset (averaging 1.93 resets/segment), consistent with its extreme steady-state kurtosis (63.32) — a "usually quiet, occasionally spikes" pattern the shared `hazard=1/250` onset-confirmation rule is not well matched to. CADC0894 shows the same qualitative failure at a markedly lower rate (79.7% of normal segments, 1.24 resets/segment, kurtosis 34.27) — a genuine but weaker version of the same pattern, not evidence the two channels share one root cause.

---

### Layer 3 — Causal Signature Analysis

**Goal**: determine *what actually changes* at an anomaly onset, and rule out that the answer is an artifact of labeling, channel choice, or data-split leakage. All Layer 3 analyses operate on the 5 scoped channels and combine five independent lines of evidence.

**3.1 Onset-uncertainty propagation (Monte Carlo).** BOCPD's onset is a posterior over run-lengths, not a point estimate. For each detected anomalous segment, 200 onset candidates are drawn in proportion to this posterior (fixed seed=42); pre/post comparisons are computed for `level` (Welch's *t*-test + Cohen's *d* on raw means), `diff`, and `diff²` (Welch's *t*-test + Cohen's *d* on **squared** pre/post values, a variance-increase proxy). Of 386 anomalous segments in the 5 scoped channels, 178 (46.1%) yield a detected onset and scoreable pre/post window; 99.4% of those fall in the high-reliability draw-count tier (≥100 usable draws). Two corrections are applied before channel-level aggregation: (i) **sign cancellation** — `level`'s direction is roughly balanced across segments (e.g., CADC0872: 39% positive/61% negative), so |*d*| is reported separately from the sign distribution; (ii) **draw-count reliability tiering** — only high/medium-tier segments are carried forward, draw-count-weighted.

Within-segment, `level` is significant in 54–92% of segments per channel (channel-level frac_sig 0.54–0.92, |*d*| 0.90–1.66), while `diff`/`diff²` are significant in only 2–76% (frac_sig 0.024–0.76). Taken alone this looks like `level` is the *more* important feature — the opposite of the precedence-test conclusion below. This apparent paradox is resolved in §3.3.

**3.2 Temporal precedence test.** If `level` shift were the primary driver and `diff`/`diff²` were downstream consequences, `level` should cross a fixed threshold (|standardized deviation from pre-onset baseline| > 3) *first*. Each segment's canonical onset (median of its 200 MC draws) defines the pre/post split; the first post-onset crossing time, in seconds, is recorded per series (offsets of 1/2 samples applied to `diff`/`diff²` to align time axes).

| Comparison | n pairs | *p* (pooled Wilcoxon) | Result |
|---|---|---|---|
| `level` vs. `diff` | 70 | 1.4 × 10⁻⁵ | `diff` precedes `level` in **84.3%** of pairs |
| `level` vs. `diff²` | 70 | 3.2 × 10⁻⁵ | `diff²` precedes `level` in **82.9%** of pairs |
| `diff` vs. `diff²` | 161 | 0.368 (n.s.) | no reliable ordering at 1–5s sampling resolution (resolved via cross-correlation lag, §3.4) |

This directly **falsifies** the working hypothesis `level → diff → diff²`.

**3.3 Reconciling §3.1 and §3.2.** Three diagnostics resolve the apparent contradiction between "`level` is significant more often" (§3.1) and "`diff`/`diff²` arrive first" (§3.2):

- **(a) Temporal-profile classification** (60-sample post-onset lookahead; *transient* = decays to ≤50% of peak, *persistent* = does not): `diff` is transient in 98% of segments, `diff²` in 94%, but `level` is a near-even mix (50% transient / 45% persistent). `diff`/`diff²` are overwhelmingly short-lived spikes; `level` is often a sustained shift.
- **(b) Early-window sensitivity check** (first 10 post-onset points vs. full post-onset window): `level`'s significance rate drops from 0.864 (full) to 0.633 (early, Δ=−0.232) — a much larger relative loss than `diff` (0.175→0.090) or `diff²` (0.220→0.130). The full-window test *dilutes* rather than misses `diff`/`diff²`'s brief spikes, while persistently-elevated `level` is detectable from a smaller sample.
- **(c) Normal-segment control (decisive).** The §3.2 procedure was repeated on normal segments, using pivots resampled from the anomalous onset-position-ratio distribution (mean ≈0.569). The precedence ordering **reverses**: `level` precedes `diff` in 55.8% of normal pairs (vs. 15.7% in anomalous pairs) and precedes `diff²` in 74.3% (vs. 17.1%). This is the decisive evidence against a "differencing is just always faster" artifact hypothesis — the same operation on normal data gives the opposite ordering.

**3.4 Formal group comparison and diff-vs-diff² tie resolution.** Two-proportion *z*/Fisher exact tests (anomalous vs. normal, Bonferroni ×3) and a channel-stratified Cochran–Mantel–Haenszel test (controlling for channel base rates) both confirm all three orderings remain highly significant after stratification (CMH p ≤ 4.33×10⁻¹⁵ for all three comparisons). The diff-vs-diff² tie (§3.2, p=0.368, driven by tied crossing times at 1–5s resolution) is resolved via the **lag of maximum cross-correlation** between the two series' post-onset trajectories (continuous, tie-free): anomalous-segment lag differs from zero (Wilcoxon p=0.00094, Bonf. 0.0028) but the anomalous-vs-normal lag *distributions* do not differ (Mann–Whitney p=0.906) — `diff` and `diff²` genuinely have no reliable ordering relative to each other, in either regime.

**3.5 Random-effects meta-analysis (DerSimonian–Laird, k=5 channels).** Pooled |Cohen's *d*| and significance-fraction (arcsine-square-root transformed) — see **Table 4**. *I²* ranges 65.7%–97.2% across all 6 outcomes (all Q-test p<.05), arguing against a single "channel-agnostic" pooled effect size without qualification.

**3.6 Leave-one-out (LOO) decomposition.** A relative-shift criterion (≥30% pooled-estimate shift) was added alongside the standard *I²*-drop (≥30pp) rule, since the latter alone missed cases where the pooled estimate moved substantially without a correspondingly large *I²* change. Key findings: removing CADC0872 eliminates all `level`-|d| heterogeneity (65.7%→0.0%); removing CADC0874 eliminates all `diff²`-|d| heterogeneity (93.8%→0.0%), correlating with CADC0874 having simultaneously the *lowest* diff² |d| and *highest* diff² transient_ratio of the five channels (Spearman r=−1.00, n=5, exploratory); removing CADC0894 nearly halves pooled diff/diff² significance-fraction (−64%/−69%) but sits atop an already-orderly hierarchy among the other four channels, i.e. it is an extreme point on a continuum, not a categorically separate population. CADC0872's own elevated `level` |*d*| was checked against two artifact hypotheses (onset-sampling dispersion, denominator shrinkage) and both were rejected; visual inspection confirms genuine, sustained step-like transitions coincident with confidently-localized BOCPD onsets.

**3.7 Quasi-experimental placebo-pool comparison — the single most decisive test.** All prior comparisons (§3.1–3.6) are within-segment and cannot distinguish "the anomaly changed this" from "this channel drifts like this anyway." Each anomalous segment's canonical-onset effect (`level` |*d*|; `diff`/`diff²` log-variance-ratio) is compared, via one-sided Mann–Whitney *U*, against a **placebo pool**: for each of up to 300 normal segments/channel, up to 5 pivots are drawn from the channel's empirical anomalous-onset-position-ratio distribution, and the identical pre/post effect is computed at each pivot. Result (15 tests, Bonferroni-corrected) — see **Table 3**: `level` is indistinguishable from ordinary drift in **all 5 channels** (p_bonf=1.000 uniformly); `diff`/`diff²` are significant in **all 5 channels**. Channel-level significant-segment fractions show `diff`/`diff²` strength is markedly higher in the three `float_noise_suspect` magnetometer channels (0.78–0.89) than the two quantized photodiode channels (0.12–0.42), consistent with the §3.5–3.6 heterogeneity pattern.

**Interpretive consequence.** The `level` "signal" that looked highly significant within-segment (§3.1) is real but **anomaly-non-specific channel drift — a confounder**, not a causal-chain endpoint. The `diff`/`diff²` variance surge is reproduced nowhere in the placebo pool and is the genuine anomaly-specific signature.

**Final SCM skeleton (Fig. 4a, "Path A")**: `Onset → {diff, diff² variance surge}` is the causal core; `level` shift is downstream/confounded and excluded from the core, based directly on the §3.7 placebo-pool result.

```
Onset (BOCPD confirmed changepoint)
      │
      ▼
Diff/Diff² variance surge (transient, onset-localized; channel-moderated:
                            float_noise_suspect(872/873/874) strong > quantized(888/894) weaker)
      │
      ▼  (weak / confounded — not a causal-chain endpoint)
Level shift (persistent in ~50% of segments, but statistically indistinguishable
             from ordinary channel drift; excluded from the causal core)
```

---

### Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification

Two threats-to-validity items are closed here, both documented in full in `docs/dev-log/step3b_labeling_protocol_revalidation.md`.

**3b.1 Labeling-protocol audit.** §3's normal-segment control found anomalous-segment onsets skewed toward the back half of their segment (mean position ratio ≈0.569). The primary OPS-SAT-AD publication, its SoftwareX companion describing the OXI annotation tool, and both preprint versions were reviewed for any documented, quantitative segment-cutting rule. **None was found.** Segments were cut entirely manually: ESA operations engineers first proposed candidate windows subjectively judged "interesting," and domain experts then used OXI to collaboratively extract and label segments. No document specifies a fixed margin or cutting convention. **Interpretation**: this is an "undecidable" finding whose implication is still informative — because boundary placement was subjective rather than rule-based, the onset-position skew more plausibly reflects the aggregate of idiosyncratic individual judgments than a systematic labeling artifact (though this cannot be fully excluded).

**3b.2 Train-only and test-only reproduction of §3.7.** The exact §3.7 placebo-pool procedure was independently re-run using only `train==1` segments (126 of 177 reliable anomalous segments) and, separately, only `train==0` segments (51 of 177). Results:

| Slice | Coverage | Result |
|---|---|---|
| Train-only | 178,504/240,979 rows (74.1%); 126 anomalous segments | 15/15 (channel×feature) significance calls match full-data exactly |
| Test-only | 62,475/240,979 rows (25.9%); 51 anomalous segments | 12/13 decidable calls match; 1 borderline (CADC0888 `diff`, p=0.051, monotone power loss — same-slice `diff²` still p=1.5×10⁻³); 2 undecidable (CADC0894, n=4<5) |

**12 of 13 decidable (channel × feature) comparisons agree exactly across full/train/test slices** (Fig. 3b, `results/layer3/triple_verification_matrix.csv`). The single disagreement's p-value rises monotonically with shrinking sample (6.9×10⁻⁶ → 4.5×10⁻⁴ → 0.051) rather than reversing direction, read as a power artifact of the smallest slice rather than an effect reversal. The 2 undecidable cells reflect CADC0894's already-documented thin sample size (Layer 2 scoping), not a new finding. This closes the same train/test-leakage concern for the causal-signature analysis that Layer 2 §2.4 closed for detection performance.

---

### Model Cross-Validation (16 independent architectures)

**Goal**: the strongest possible test of whether "diff/diff² matter more than level" is a property of the *data* or an artifact of *our specific detector's inductive bias*. Full detail in `docs/dev-log/step_ai_model_cross_validation.md`.

**Task and labels.** All models solve the identical binary classification problem defined by §3.7's design (canonical-onset anomalous segments vs. onset-position-ratio-resampled placebo pivots on normal segments) — the same labeling and pivot-sampling logic as the primary statistical test, not a separately motivated task. Resulting label set: 4,996 tabular rows (168 positive / 4,828 negative), 2,733 sequence-window rows, pooled across the 5 scoped channels.

**Two independent data representations.**
- *Tabular* (10 models): the three §3.7 engineered features (`level` |*d*|, `diff`/`diff²` log-variance-ratio). Attribution: SHAP (Tree/Linear/Kernel explainer) where computable, else permutation importance (20 repeats, ROC-AUC).
- *Raw sequence* (6 models): per-window z-normalized 3-channel time series (`level`, `diff`, `diff²` stacked as channels), ±15 samples around the pivot — **no human-engineered summary statistic**. Attribution: Integrated Gradients (Captum, else manual 32-step Riemann-sum), summed over time, normalized per channel.

If tabular (human-summarized) and sequence (raw, model-discovered) representations converge, this rules out "the summary-statistic design itself favored diff/diff²" as an explanation.

**Model zoo — 16 models, 13 inductive-bias families** (5-fold grouped cross-validation, grouped by channel–segment identity):

| Family | Models | Representation |
|---|---|---|
| Linear / Linear (sparse) | LogReg (L2), LogReg (L1) | tabular |
| Probabilistic | GaussianNB | tabular |
| Instance-based | kNN | tabular |
| Kernel | SVM (RBF) | tabular |
| Tree ensemble (bagging) / (bagging, extra) / (boosting) | RandomForest / ExtraTrees / GradBoost (sklearn), XGBoost, LightGBM | tabular |
| Convolutional / (dilated causal) | CNN1D / TCN | sequence |
| Recurrent | BiLSTM, BiGRU | sequence |
| Attention | TinyTransformer | sequence |
| State-space (SSM) | LightMamba (pure-PyTorch selective-SSM substitute; official `mamba_ssm` CUDA kernel unavailable) | sequence |

**Implementation notes (bugs found and fixed during this run — none affect reported numerics except #3).** (1) SHAP return-shape normalization: newer SHAP versions return a 3D `(n, features, classes)` array for some explainer/model pairs instead of the legacy list format; a normalization helper collapses both forms consistently, falling back to permutation importance on failure. (2) `XGBClassifier(use_label_encoder=False)` removed — this argument was deleted in xgboost 2.x. (3) **A 17th model, MLP_shallow, was excluded** after achieving out-of-fold AUC=0.403 (below chance) — a failed optimization, not a genuine level-favoring inductive bias; an untrained model's attributions carry no interpretable signal, so all headline statistics below are over **16 models, not 17**. (4) A cuDNN RNN backward-mode fix was applied to BiLSTM/BiGRU saliency computation (`torch.backends.cudnn.flags(enabled=False)` wrapped around the attribution call only — no effect on numerics, only avoids a fused-kernel backward-mode limitation in eval mode).

**Result: 16/16 models rank `diff`+`diff²` above `level`. Zero of 16 rank `level` as the single top feature** (binomial p=.0015). Full per-model table in **Table 2**; family-level aggregation in **Table 7**. Cross-model agreement: **Kendall's W = 0.609** (χ²=19.50, df=2, p<.001) — substantial agreement given each model was trained, attributed, and evaluated completely independently with no shared hyperparameters. Bootstrap stability (200 resamples on two representative, architecturally distinct tabular models): **100%** for both RandomForest and LogisticRegression. **Tabular (n=10) and sequence (n=6) representations agree on 13/13 families** — the strongest available evidence against "the engineered summary-feature design itself favored diff/diff²," since the sequence models never see those engineered features at all. The one notable exception is the **State-space (SSM)** family, which shows the narrowest margin (level=0.428) — attributed to the lightweight, non-CUDA-kernel Mamba-style substitute block used here, and flagged for confirmation under the official implementation.

---

## Figures

All figures below are generated by the scripts in `results/figures/` and are byte-for-byte what running each script produces from `results/*.csv` / `results/*.json`. No figure in this README was hand-edited after generation. Every figure is designed to be fully legible in black-and-white print: categories are encoded redundantly by **greyscale value *and* hatch pattern**, never by hue alone; where a figure previously relied on a colormap, it has been redrawn in pure black/white/hatch with the underlying category also spelled out as on-cell text, so the encoding is triply redundant (fill, hatch, text).

### Figure 1 — Channel scoping: MCC/Youden's J and the 9→5 funnel

![Figure 1: Channel scoping](results/figures/fig01_channel_scope.png)

*(a) MCC and Youden's J per channel at the locked hyperparameters `(mixture=False, forgetting=True)`. Blue/teal bars = channels retained in the final scope; grey bars = excluded, with the exclusion reason (structural / underpowered / chance-level) printed directly beneath each excluded channel's tick label. `CADC0884` has zero anomalous segments and cannot be scored. Note that `CADC0890` has by far the highest point-estimate MCC (0.826) of any channel yet is excluded — a deliberate conservative choice driven by its n=11/14 sample size, not by weak detection performance. (b) The 9→5 channel-scoping funnel: 9 raw channels → 6 channels survive the first-pass MCC/Youden screen → the final 5-channel scope is then confirmed identical across three independent re-derivations (full-data bootstrap CI stability, train-only reselection, test-only confirmation), annotated on each funnel stage.*
**Reproduces from:** `results/layer2/channel_scope.json`, `results/layer2/bootstrap_ci.csv` — **Script:** `results/figures/fig01_channel_scope.py`

### Figure 2 — 16-model cross-validation: feature attribution and agreement

![Figure 2: Model cross-validation](results/figures/fig02_model_cross_validation.png)

*(a) Normalized feature-importance share (level / diff / diff²) for each of 16 independently trained models, sorted by out-of-fold AUC (annotated at the right edge of each row). The dotted vertical line marks the chance level (1/3) each feature would receive under an uninformative split. `level` = dotted hatch, `diff` = diagonal hatch, `diff²` = solid black — every one of the 16 models allocates substantially more than 1/3 combined share to the diff+diff² (hatched+black) region, and the six sequence-representation models (top of the chart, highest AUCs) never having seen the engineered features at all still reach the same conclusion as the ten tabular models below them. (b) Agreement-statistics panel: Kendall's W and its χ² test, the binomial test on "0/16 models rank `level` top," 200-resample bootstrap stability for two architecturally distinct representative models, the 13/13 family-level agreement result, and the tabular-vs-sequence representation-agreement finding that rules out summary-statistic design as a confound.*
**Reproduces from:** `results/model_cross_validation/feature_attribution_16models.csv`, `results/model_cross_validation/agreement_statistics.json` — **Script:** `results/figures/fig02_model_cross_validation.py`

### Figure 3 — Quasi-experimental placebo-pool test and triple verification

![Figure 3: Quasi-experimental comparison](results/figures/fig03_quasi_experimental.png)

*(a) −log₁₀(p_bonferroni) for each of the 5 scoped channels' anomalous segments vs. same-channel placebo pool of normal segments (Mann–Whitney U, 15 tests total, Bonferroni-corrected; red dashed line = α=.05 threshold). `level` bars are exactly zero-height in every channel (p_bonf=1.000 uniformly) — the visual signature of a null result repeated identically five times; `diff` and `diff²` clear the significance threshold by 4 to 21 orders of magnitude in every one of the 5 scoped channels. (b) Full/train-only/test-only agreement matrix for all 15 (channel × feature) cells, rendered in black (`sig.`) / white (`n.s.`) / pale-yellow-hatched (`n/a`, meaning undecidable due to n<5) — 12 of the 13 decidable cells agree exactly across all three independent data slices; the single disagreement (CADC0888, `diff`, test-only) and the two undecidable cells (CADC0894, both features, test-only) are each annotated with their cause in the accompanying text (§ Layer 3b).*
**Reproduces from:** `results/layer3/placebo_comparison.csv`, `results/layer3/triple_verification_matrix.csv` — **Script:** `results/figures/fig03_quasi_experimental.py`

### Figure 4 — Final SCM skeleton and meta-analytic heterogeneity

![Figure 4: SCM and heterogeneity](results/figures/fig04_scm_and_heterogeneity.png)

*(a) The final structural skeleton ("Path A"): onset (black) drives a transient diff/diff² variance surge (dark grey) — the causal core; level shift (light grey) is excluded from the core (dashed grey arrow, annotated "weak / uncertain") because it is not distinguishable from normal-operation drift, citing the Fig. 3a result directly in a bordered annotation box. A second bordered box states the channel-type moderator finding: `float_noise_suspect` channels (872/873/874) show a strong diff/diff² effect, `quantized` channels (888/894) a systematically weaker one. (b) DerSimonian–Laird random-effects meta-analysis heterogeneity (*I²*) across the 5 scoped channels, for all 6 outcome measures (effect size |*d*| and significance fraction, × 3 features). Every bar clears the conventional "high heterogeneity" (*I²*=50%) threshold and all but one clears "very high" (*I²*=75%, dashed vertical line, labeled in its own row below the bars so it never collides with the panel title); each bar is annotated with its exact *I²* and Q-test p-value.*
**Reproduces from:** `results/layer3/heterogeneity_summary.csv` — **Script:** `results/figures/fig04_scm_and_heterogeneity.py`

### Figure 5 — Dataset overview and pooled temporal-precedence test

![Figure 5: Dataset and precedence](results/figures/fig05_dataset_and_precedence.png)

*(a) OPS-SAT-AD dataset composition: segment counts (bars, left axis) and anomaly prevalence (dashed black line with open-circle markers, right axis) for all 9 channels, with sensor type (Magn./Phot.) printed as a compact two-line tick sub-label beneath each channel code. Dark solid bars mark the final 5-channel analytical scope; hatched grey bars are excluded. `CADC0890`'s extreme 78.6% anomaly prevalence — the highest of any channel — is visible as the peak of the dashed line, directly over its (excluded, hatched) bar, visually reinforcing that its exclusion is a sample-size decision, not a "boring channel" decision. (b) The pooled temporal-precedence test as −log₁₀(p): `level`-vs-`diff` and `level`-vs-`diff²` both clear the α=.05 line (dashed) by 4–5 orders of magnitude, each bar annotated with the exact p-value and the percentage of paired segments where diff/diff² precedes level (84.3% / 82.9%); `diff`-vs-`diff²` (dotted hatch) sits far below the threshold, annotated "n.s. — no reliable ordering," consistent with the §3.4 tie-resolution finding that this pair genuinely has no consistent precedence in either regime.*
**Reproduces from:** `data/raw/` (segment metadata), `results/layer3/temporal_precedence.csv` — **Script:** `results/figures/fig05_dataset_and_precedence.py`

---

## Results tables

### Table 1 — Channel scoping decision table (underlies Fig. 1)

| Channel | Sensor | # segments | Anomaly % | Recall | FA rate | MCC | Youden's J | Decision |
|---|---|---|---|---|---|---|---|---|
| CADC0872 | Magnetometer #1 | 546 | 24.0 | 0.321 | 0.000 | 0.514 | 0.321 | **Included** |
| CADC0873 | Magnetometer #2 | 593 | 17.7 | 0.276 | 0.000 | 0.489 | 0.276 | **Included** |
| CADC0874 | Magnetometer #3 | 194 | 35.6 | 0.725 | 0.032 | 0.740 | 0.693 | **Included** |
| CADC0884 | Photodiode #1 | 158 | 0.0 | n/a | 0.241 | n/a | n/a | Excluded (structural: n_anom = 0) |
| CADC0886 | Photodiode #2 | 11 | 27.3 | 0.000 | 0.000 | 0.000 | 0.000 | Excluded (underpowered: n = 3/8) |
| CADC0888 | Photodiode #3 | 252 | 23.8 | 0.617 | 0.391 | 0.194 | 0.226 | **Included** |
| CADC0890 | Photodiode #4 | 14 | 78.6 | 0.909 | 0.000 | 0.826 | 0.909 | Excluded (underpowered: n = 11/14, despite highest MCC of any channel) |
| CADC0892 | Photodiode #5 | 211 | 16.1 | 0.971 | 0.994 | −0.090 | −0.024 | Excluded (chance-level: MCC < 0) |
| CADC0894 | Photodiode #6 | 144 | 14.6 | 1.000 | 0.797 | 0.189 | 0.203 | **Included** |

### Table 2 — Model cross-validation summary (underlies Fig. 2)

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
| *BOCPD (τ\* Youden, Layer-2 reference)* | *Reference (online, probabilistic)* | — | *n/a* | recall=0.685, FA=0.181 (not commensurable with batch-classifier AUC) | | |

**Agreement statistics**: Kendall's W = 0.609 (χ² = 19.50, df = 2, p < .001) · models ranking `level` top = 0/16 (binomial p = .0015) · bootstrap stability (200 resamples): RandomForest 100%, LogReg 100% favor diff/diff² · 13/13 inductive-bias families favor diff+diff² over level (narrowest margin: SSM family) · tabular (n=10) and sequence (n=6) representations agree.

### Table 3 — Quasi-experimental placebo-pool comparison (underlies Fig. 3a)

| Channel | level, p_bonf | diff, p_bonf | diff², p_bonf |
|---|---|---|---|
| CADC0872 | 1.000 | 7.94 × 10⁻²⁴ | 1.12 × 10⁻²² |
| CADC0873 | 1.000 | 2.40 × 10⁻¹⁷ | 3.09 × 10⁻¹⁶ |
| CADC0874 | 1.000 | 5.37 × 10⁻¹⁶ | 2.88 × 10⁻²² |
| CADC0888 | 1.000 | 6.92 × 10⁻⁶ | 3.24 × 10⁻¹³ |
| CADC0894 | 1.000 | 3.89 × 10⁻⁶ | 1.95 × 10⁻⁴ |

*Mann–Whitney U, 15 tests total, Bonferroni-corrected (α = .05 → α_bonf = .00333). Full n_anomaly / n_placebo / medians in `results/layer3/placebo_comparison.csv`.*

### Table 4 — Heterogeneity summary (underlies Fig. 4b)

| Outcome | I² | Q-test p |
|---|---|---|
| Effect size \|d\| (level) | 65.7% | 0.0202 |
| Effect size \|d\| (diff) | 97.2% | < 0.0001 |
| Effect size \|d\| (diff²) | 93.8% | 2.6 × 10⁻¹³ |
| frac_sig (level) | 82.8% | 1.1 × 10⁻⁴ |
| frac_sig (diff) | 93.8% | 3.2 × 10⁻¹³ |
| frac_sig (diff²) | 92.2% | 1.9 × 10⁻¹⁰ |

*DerSimonian–Laird random-effects meta-analysis, k = 5 channels.*

### Table 5 — Full/train/test triple-verification agreement (underlies Fig. 3b)

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
| CADC0888 | diff | Yes | Yes | **No** (p=0.051) | **No** (power loss, see § Layer 3b) |
| CADC0888 | diff² | Yes | Yes | Yes | Yes |
| CADC0894 | level | No | No | No | Yes |
| CADC0894 | diff | Yes | Yes | n/a (n=4<5) | n/a |
| CADC0894 | diff² | Yes | Yes | n/a (n=4<5) | n/a |

### Table 6 — Model-cross-validation agreement and stability statistics

| Statistic | Value |
|---|---|
| Kendall's coefficient of concordance (*W*) | **0.609** |
| χ² approximation (df = 2) | χ² = 19.50, *p* < .001 |
| Models ranking `level` as most important feature | 0 / 16 |
| Binomial test (H₀: chance rate = 1/3 favor diff/diff²) | *p* = .0015 |
| Bootstrap stability, RandomForest (200 resamples) | 100% favor diff+diff² > level |
| Bootstrap stability, LogisticRegression (200 resamples) | 100% favor diff+diff² > level |

### Table 7 — Family-level aggregation (13 inductive-bias families)

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

### Table 8 — 2×2 ablation grid (train macro-MCC; underlies Layer 2 §2.2)

| mixture | forgetting | train_macro_mcc | n_scoring_channels |
|---|---|---|---|
| False | True (**locked**) | 0.308 | 6 |
| True | True | 0.275 | 6 |
| False | False | 0.238 | 6 |
| True | False | 0.199 | 6 |

### Table 9 — Pooled temporal-precedence test (underlies Fig. 5b)

| Comparison | n pairs | p (Wilcoxon) | Interpretation |
|---|---|---|---|
| level vs. diff | 70 | 1.4 × 10⁻⁵ | diff precedes level in 84.3% of pairs |
| level vs. diff² | 70 | 3.2 × 10⁻⁵ | diff² precedes level in 82.9% of pairs |
| diff vs. diff² | 161 | 0.368 (n.s.) | no reliable ordering (resolved via cross-correlation lag) |

### Table 10 — Normal-segment control: precedence ordering reversal

| Comparison | Regime | n pairs | frac(first precedes second) |
|---|---|---|---|
| level vs. diff | anomalous | 70 | level 15.7% / diff **84.3%** |
| level vs. diff | normal | 224 | level **55.8%** / diff 44.2% |
| level vs. diff² | anomalous | 70 | level 17.1% / diff² **82.9%** |
| level vs. diff² | normal | 171 | level **74.3%** / diff² 25.7% |
| diff vs. diff² | anomalous | 161 | diff 7.5% / diff² **92.5%** |
| diff vs. diff² | normal | 211 | diff **64.5%** / diff² 35.5% |

### Table 11 — Layer 1 noise estimates (underlies signal-estimation stage)

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

# 2. Download the OPS-SAT-AD dataset (not redistributed in this repo)
bash data/download_opssat_ad.sh

# 3. Run the full Layer 1 → 2 → 3 pipeline
python -m layer1_signal_estimation.run
python -m layer2_anomaly_detection.run
python -m layer3_causal_analysis.run

# 4. Run the labeling-protocol audit + train/test triple re-verification (Layer 3b)
python -m layer3_causal_analysis.labeling_protocol_audit
python -m layer3_causal_analysis.train_test_triple_reverification

# 5. Run the independent 16-model cross-validation benchmark
python -m model_cross_validation.run_all

# 6. Regenerate every figure in this README from the results/ artifacts above
python results/figures/fig01_channel_scope.py
python results/figures/fig02_model_cross_validation.py
python results/figures/fig03_quasi_experimental.py
python results/figures/fig04_scm_and_heterogeneity.py
python results/figures/fig05_dataset_and_precedence.py

# ...or simply:
make all
```

`REPRODUCIBILITY_CHECKLIST.md` documents random seeds, package versions, and expected runtime (CPU-only: ~40 min for Layers 1–3; the 16-model benchmark, including the deep-sequence GPU-optional models, is the long pole at ~2–3 hours on CPU / ~25 min on a single GPU).

---

## Threats to validity and how each was addressed

| Threat | Mitigation |
|---|---|
| **Train/test leakage in channel selection** | The entire channel-scoping procedure (Layer 2) was independently re-run on the train-only split and re-confirmed on the test-only split (Fig. 1b); the 5-channel scope is identical in all three passes. |
| **Train/test leakage in the causal-signature (placebo-pool) result** | The entire §3.7 procedure was independently re-run on train-only and test-only slices (Layer 3b); 12/13 decidable comparisons agree exactly, the one disagreement is a diagnosed power artifact, and the two undecidable cells reflect a pre-existing, already-documented sample-size limitation. |
| **Diff/diff² result being a labeling artifact** | External audit of the OPS-SAT-AD labeling protocol (`docs/dev-log/step3b_labeling_protocol_revalidation.md`): no documented quantitative segment-cutting rule exists; boundaries were set by subjective manual expert judgment, which limits (without fully excluding) the plausibility of a systematic labeling-protocol confound. |
| **Diff/diff² result being an artifact of our detector's inductive bias** | Independent 16-model cross-validation across 13 inductive-bias families, 2 independent data representations (tabular / raw sequence), and 2 independent attribution methods (SHAP-permutation / integrated gradients) — all converge (Fig. 2); 100% bootstrap stability on two representative, architecturally distinct models. |
| **Level shift "looking" important only because it's not tested against the right null** | Quasi-experimental placebo-pool design: anomalous segments compared against *same-channel normal-operation segments*, not against a pooled cross-channel baseline (Fig. 3a). |
| **`level`'s high within-segment significance (§3.1) seemingly contradicting the precedence-test result (§3.2)** | Resolved via temporal-profile classification (transient vs. persistent), an early-window sensitivity check, and the normal-segment control — `level` is a sustained shift detectable at low sample sizes, while `diff`/`diff²` are brief transients diluted by a full-window test; the placebo-pool result (§3.7) shows `level`'s sustained shift is itself a non-specific confound. |
| **Small-sample channels inflating apparent effect sizes** | Structural/underpowered channels (`CADC0884`, `CADC0886`, `CADC0890`) explicitly excluded from the analytical scope rather than down-weighted (Fig. 1); `CADC0890`'s exclusion despite MCC = 0.826 (the highest of any channel) is a deliberate conservative choice, documented in Table 1. |
| **Between-channel heterogeneity being treated as noise** | Heterogeneity is quantified (I², Q-test), tested for robustness via a dual-criterion (I²-drop + relative-shift) leave-one-out sensitivity analysis, and shown to be *moderated* by a known covariate (channel noise/quantization regime) rather than dismissed (Fig. 4b, §3.6). |
| **Multiple-comparisons inflation** | Bonferroni correction applied to all 15 placebo-pool tests (Table 3) and their train/test replications (Table 5); pooled (not per-test) Wilcoxon test used for the temporal-precedence claim (Table 9); Bonferroni also applied to the temporal-precedence group-comparison and lag-resolution tests (§3.4). |

---

## Relationship to formal causal inference — scope statement

We use the term **"causal signature"** in this repository and the associated manuscript to describe a claim of the form *"X is the feature that changes at anomaly onset, and this is not explainable by normal-operation drift or by dataset/model artifacts."* This is established via **quasi-experimental design** (same-channel placebo-pool comparison), **temporal precedence testing with a normal-segment control**, **channel-stratified confirmation** (CMH), **train/test/bootstrap triple verification**, **convergent evidence from 16 architecturally independent classifiers**, and a **structural causal model (SCM) skeleton** built from the sensor's signal-processing structure (Path A in Fig. 4a) rather than from a data-driven causal discovery algorithm.

We do **not** claim formal statistical identification in the Pearl do-calculus sense, and we do not run a discovery algorithm (e.g., PC, FCI, LiNGAM) to *derive* the SCM edges — the skeleton in Fig. 4a is asserted from known instrument physics (a change-point in the underlying process necessarily first perturbs local variance/derivative statistics before a sustained mean shift is observable) and then **tested against data** for consistency, not derived from data. We flag this distinction explicitly so that reviewers and readers calibrate the strength of the causal claim correctly: it is a **rigorously tested, quasi-experimental causal-signature claim**, not a formally identified causal effect estimate.

---

## Limitations

- **Single benchmark dataset.** All results are on OPS-SAT-AD. We have not yet validated the diff/diff² signature on a second, independent spacecraft-telemetry anomaly benchmark; this is the most important direction for follow-up work and the most likely reviewer objection.
- **No comparison against published OPS-SAT-AD baselines.** This repository's contribution is a *causal-signature finding*, not a new detector claiming state-of-the-art detection performance; nonetheless, a table situating our detector's MCC/Youden's J against previously published OPS-SAT-AD results strengthens the paper and is planned for the camera-ready version.
- **Detection performance is modest in absolute terms** on some channels (e.g., CADC0888, CADC0894 sit well below CADC0874). The paper's contribution is about *what the signature is*, not about maximizing raw detection recall/precision.
- **5 of 9 channels are excluded from the causal analysis** for principled statistical-power reasons (Table 1); the causal-signature claim is scoped to these 5 channels and is not asserted for the excluded 4.
- **Labeling-protocol audit is inherently undecidable.** No public documentation specifies a quantitative segment-cutting rule for OPS-SAT-AD, so the possibility of a subtle, non-systematic labeling influence on onset-position skew cannot be fully excluded — only shown to be less plausible than under a fixed, rule-based cutting convention.
- **CADC0894's small anomalous-segment count** (n=21 full-data, n=4 in the test-only slice) limits statistical power in every test-split replication and is the source of the two undecidable cells in Table 5.
- **CADC0874's diff² effect size is likely under-estimated** by the full-window Welch design relative to a design tuned to its especially short-lived transient response (§3.3(a)); the reported meta-analytic estimate for this channel-feature combination should be treated as a conservative lower bound.
- **The state-space (Mamba-style) model is a lightweight, non-CUDA-kernel substitute** for the official `mamba_ssm` implementation; its narrower agreement margin relative to the other 12 inductive-bias families (Table 7) should not be over-interpreted without confirming the result under the official implementation.

---

## Suggested target journals

TODO — candidates under consideration include venues publishing spacecraft telemetry / anomaly-detection benchmark work and applied causal-inference methodology (to be finalized alongside the manuscript).

---

## Citation

TODO — `CITATION.cff` will be populated once the arXiv preprint is available.

---

## License

TODO (e.g., MIT for code; dataset governed by its own OPS-SAT-AD license — see `data/README.md`).
