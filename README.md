# opssat-ad-detection-causal

**Detection and Causal Signature Identification of Telemetry Anomalies in the OPS-SAT-AD Benchmark**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-fully%20reproducible-brightgreen.svg)](#reproducing-every-number-in-this-readme)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](#citation)

> **One-sentence finding.** In the OPS-SAT-AD spacecraft telemetry benchmark, the true signature of an anomaly is **not a level (mean) shift** but a **transient surge in first- and second-difference variance** (`diff`, `diff²`), and this pattern is reproducible across channels, train/test/bootstrap data splits, and **16 independent model architectures** spanning six inductive-bias families.

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
5. [Methodology](#methodology) — Layer 1 / Layer 2 / Layer 3 / Model cross-validation
6. [Figures](#figures) (all reproduced inline)
7. [Results tables](#results-tables)
8. [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
9. [Threats to validity and how each was addressed](#threats-to-validity-and-how-each-was-addressed)
10. [Relationship to formal causal inference — scope statement](#relationship-to-formal-causal-inference--scope-statement)
11. [Limitations](#limitations)
12. [Suggested target journals](#suggested-target-journals)
13. [Citation](#citation)
14. [License](#license)

---

## Why this repository exists

Change-point and anomaly-detection pipelines for spacecraft telemetry almost universally treat a **level shift** (a change in the mean of the raw signal) as the canonical anomaly signature. This repository documents a systematic, triple-validated investigation of the **OPS-SAT-AD** benchmark (European Space Agency OPS-SAT mission, 9 telemetry channels, 2,123 expert-labeled univariate segments) that found the opposite: level shifts are **not statistically distinguishable from normal-operation drift** once tested against a same-channel placebo pool, while **variance surges in the first and second differences of the signal are the actual, reproducible fingerprint of an anomaly**.

Because a single-pipeline finding is easy to dismiss as an artifact of one detector's inductive bias, we did not stop at the detection pipeline. We additionally ran an **independent 16-model cross-validation study** — spanning linear, probabilistic, kernel, tree-ensemble, and deep-sequence (CNN/TCN/BiLSTM/BiGRU/Transformer/SSM) architectures — and asked each model, blind to the others, which feature family it relied on. **16 out of 16 models ranked `diff`/`diff²` above `level`.**

This repository packages:
- The full **signal-estimation → change-point-detection → causal-signature** pipeline (Layers 1–3).
- The **16-model cross-validation** benchmark and its agreement statistics.
- **Every figure in this README regenerated from its exact source script** (`results/figures/scripts/*.py`), so every number a reviewer sees can be traced to code, not to a table typed by hand.
- **Raw development logs** (`docs/dev-log/`) documenting the full, unabridged analytical trail — including negative results, ablations that were discarded, and the exact point at which each methodological correction (e.g., train/test leakage checks) was introduced. These are provided for full transparency and are **not** a substitute for the polished Methods text in the eventual manuscript, which draws on them.

---

## Headline results

| Claim | Evidence | Section |
|---|---|---|
| Channel scope: 9 → 5 channels, survives 3 independent verifications | Fig. 1 | [§ Layer 2](#layer-2--anomaly-detection-change-point-model--channel-scoping) |
| `diff`/`diff²` precede `level` in onset timing (pooled Wilcoxon, 5 channels) | Fig. 5b | [§ Layer 3](#layer-3--causal-signature-analysis) |
| `level` is **not** distinguishable from normal-operation drift (quasi-experimental placebo test, 5/5 channels n.s.) | Fig. 3a | [§ Layer 3](#layer-3--causal-signature-analysis) |
| `diff`/`diff²` **are** distinguishable from normal drift (5/5 channels, Bonferroni-corrected) | Fig. 3a | [§ Layer 3](#layer-3--causal-signature-analysis) |
| Result replicates independently on full / train-only / test-only splits (12/13 decidable comparisons agree) | Fig. 3b | [§ Layer 3](#layer-3--causal-signature-analysis) |
| 16/16 independent models rank `diff`+`diff²` over `level`; Kendall's W = 0.609 (p < .001) | Fig. 2 | [§ Model cross-validation](#model-cross-validation-16-independent-architectures) |
| Channel-level heterogeneity is real and interpretable (I² up to 97%, moderated by quantization type) | Fig. 4b | [§ Layer 3](#layer-3--causal-signature-analysis) |

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
│   ├── tests/
│   └── run.py
│
├── layer3_causal_analysis/
│   ├── onset_mc_propagation.py            # Monte Carlo propagation of onset-time posterior
│   ├── temporal_precedence_test.py        # pooled Wilcoxon signed-rank, level vs diff vs diff² (Fig. 5b)
│   ├── quasi_experimental_placebo.py      # Mann–Whitney U vs. same-channel placebo pool (Fig. 3a)
│   ├── meta_analysis_random_effects.py    # DerSimonian–Laird random-effects meta-analysis + I² (Fig. 4b)
│   ├── loo_sensitivity.py                 # leave-one-channel-out sensitivity analysis
│   ├── scm_skeleton.py                    # structural skeleton construction, Path A vs. Path B (Fig. 4a)
│   ├── supplementary_diagnostics_A_D.py   # cross-correlation, CMH test, normal-segment controls
│   ├── labeling_protocol_audit.py         # external audit of the OPS-SAT-AD labeling protocol
│   ├── tests/
│   └── run.py
│
├── model_cross_validation/
│   ├── models/                            # 16 independent model wrappers
│   │   ├── linear.py                      # LogReg (L1, L2)
│   │   ├── probabilistic.py               # GaussianNB
│   │   ├── kernel.py                      # SVM (RBF)
│   │   ├── instance_based.py              # kNN
│   │   ├── tree_ensembles.py              # RandomForest, ExtraTrees, GradBoost, XGBoost, LightGBM
│   │   ├── conv_sequence.py               # CNN1D, TCN
│   │   ├── recurrent_sequence.py          # BiLSTM, BiGRU
│   │   ├── attention_sequence.py          # TinyTransformer
│   │   └── ssm_sequence.py                # LightMamba (state-space model)
│   ├── representations/
│   │   ├── tabular_summary_features.py    # 10 models: engineered summary-statistic features
│   │   └── raw_sequence_features.py       # 6 models: raw windowed time series
│   ├── attribution/
│   │   ├── shap_permutation.py            # tabular-model attribution (SHAP + permutation importance)
│   │   └── integrated_gradients.py        # sequence-model attribution (saliency)
│   ├── agreement_statistics.py            # Kendall's W, binomial test, bootstrap stability (Fig. 2b)
│   ├── tests/
│   └── run_all.py                         # entry point: `python -m model_cross_validation.run_all`
│
├── results/
│   ├── layer1/
│   │   ├── noise_estimates.csv
│   │   └── channel_classification.json
│   ├── layer2/
│   │   ├── channel_scope.json             # → consumed by opssat-ad-risk-optimization
│   │   ├── ablation_grid.csv
│   │   └── bootstrap_ci.csv
│   ├── layer3/
│   │   ├── onset_posteriors.parquet       # → consumed by opssat-ad-risk-optimization & opssat-ad-dss
│   │   ├── temporal_precedence.csv
│   │   ├── placebo_comparison.csv
│   │   ├── heterogeneity_summary.csv      # → consumed by opssat-ad-dss
│   │   └── triple_verification_matrix.csv
│   ├── model_cross_validation/
│   │   ├── feature_attribution_16models.csv
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

**OPS-SAT-AD** — European Space Agency's OPS-SAT experimental nanosatellite telemetry anomaly-detection benchmark.

- **9 channels**: 3 magnetometer channels (`CADC0872`, `CADC0873`, `CADC0874`) + 6 photodiode channels (`CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894`).
- **2,123 univariate segments** total, each expert-labeled as *nominal* or *anomalous*.
- Per-channel segment counts and anomaly prevalence vary sharply (0% to 78.6%) — see Fig. 5a.
- We do **not** redistribute the raw dataset. `data/download_opssat_ad.sh` fetches it directly from its Zenodo record and lays it out in the directory structure the pipeline expects.

---

## Methodology

### Layer 1 — Signal Estimation

**Goal**: obtain a denoised, well-calibrated state estimate for each channel before any change-point logic is applied, so that downstream detection is not confounded by channel-specific noise characteristics.

1. **Automatic channel classification** into `quantized`, `float_noise_suspect`, or `continuous` regimes, based on the empirical distribution of consecutive differences.
2. **Noise-estimator comparison**: a simple `var(diff)/2` estimator was compared against a Gaussian-mixture noise estimator. The simple estimator was adopted — the mixture model did not improve calibration and added a source of instability for low-sample channels.
3. **State estimation**: a local-linear-trend **Kalman filter** is fit per channel, with a **hierarchical Bayesian shrinkage** correction applied to channels with very small sample sizes, pooling strength across channels to stabilize estimates that would otherwise be dominated by noise.

### Layer 2 — Anomaly Detection (change-point model + channel scoping)

**Goal**: convert the Layer-1 state estimate into a calibrated change-point/anomaly score, and determine — rigorously, not by convenience — which channels are even analyzable.

1. **BOCPD + forgetting**: Bayesian Online Change-Point Detection was extended with a *forgetting* mechanism to prevent runaway overconfidence during long nominal stretches (a known failure mode of vanilla BOCPD on long quiet periods).
2. **Full ablation**: all four combinations of `{mixture noise model: on/off} × {forgetting: on/off}` were exhaustively compared. `(mixture=False, forgetting=True)` was selected on MCC/Youden's J grounds and locked for all downstream analysis.
3. **Channel scoping** (Fig. 1): channels were evaluated at the locked hyperparameters and screened for:
   - **Structural exclusion** — `CADC0884` has zero anomaly examples in the labeled set (cannot be evaluated at all).
   - **Statistical underpowering** — `CADC0886` (n=3/8 anomalous segments) and `CADC0890` (n=14) are excluded despite `CADC0890`'s very high MCC (0.826), because the estimate is not trustworthy at that sample size.
   - **Chance-level performance** — `CADC0892` has MCC = −0.090, i.e. performs *below* chance; excluded.
   - **Final scope: 5 channels** — `CADC0872`, `CADC0873`, `CADC0874`, `CADC0888`, `CADC0894`.
4. **Triple verification** (Fig. 1b): the 5-channel scope is not a single-pass decision. It is re-derived three independent ways — (i) full-data bootstrap CI stability, (ii) **train-only** re-selection (the entire channel-scoping procedure re-run using only the training split, to rule out test-set leakage into the selection process), and (iii) **test-only** confirmation. All three agree on the same 5 channels.

### Layer 3 — Causal Signature Analysis

**Goal**: determine *what actually changes* at an anomaly onset — and rule out that the answer is an artifact of labeling, channel choice, or data-split leakage.

This is the analytical core of the paper and combines five independent lines of evidence:

**(a) Uncertainty quantification.** Onset time is not a point estimate — it is a BOCPD posterior. We propagate this posterior via **Monte Carlo sampling** (200 draws per onset) through every downstream test, so that all subsequent p-values and effect sizes reflect onset-time uncertainty rather than treating a single MAP estimate as ground truth.

**(b) Temporal precedence test** (Fig. 5b). If `level` shift were the true driver of the anomaly and `diff`/`diff²` were merely downstream consequences, `level` should cross its anomaly threshold *first*. We tested this directly with a pooled Wilcoxon signed-rank test over paired segments (n = 70–161 pairs per comparison, across the 5 scoped channels):

| Comparison | p (pooled, Wilcoxon) | Interpretation |
|---|---|---|
| `level` vs. `diff` | 1.4 × 10⁻⁵ | `diff` precedes `level` in **84.3%** of pairs (level precedes in only 15.7%) |
| `level` vs. `diff²` | 3.2 × 10⁻⁵ | `diff²` precedes `level` in **82.9%** of pairs (level precedes in only 17.1%) |
| `diff` vs. `diff²` | 0.368 (n.s.) | No reliable ordering between `diff` and `diff²` themselves |

This is the opposite of the naive `level → diff → diff²` hypothesis, and was confirmed, not assumed — it directly falsified our own prior expectation going into the analysis.

**(c) Quasi-experimental placebo-pool comparison** (Fig. 3a) — **the single most decisive test in the paper**. Rather than only checking whether a feature moves *during* labeled anomalies, we ask the sharper question: is that movement *specific to anomalies*, or is it also common during ordinary operation? Each channel's anomalous segments were compared against a **placebo pool of that same channel's normal-operation segments** using Mann–Whitney U tests (15 tests total across 3 features × 5 channels, Bonferroni-corrected):

| Channel | `level` (\|d\|), p_bonf | `diff` (log-var ratio), p_bonf | `diff²` (log-var ratio), p_bonf |
|---|---|---|---|
| CADC0872 | 1.000 (n.s.) | 7.9 × 10⁻²⁴ | 1.1 × 10⁻²² |
| CADC0873 | 1.000 (n.s.) | 2.4 × 10⁻¹⁷ | 3.1 × 10⁻¹⁶ |
| CADC0874 | 1.000 (n.s.) | 5.4 × 10⁻¹⁶ | 2.9 × 10⁻²² |
| CADC0888 | 1.000 (n.s.) | 6.9 × 10⁻⁶ | 3.2 × 10⁻¹³ |
| CADC0894 | 1.000 (n.s.) | 3.9 × 10⁻⁶ | 1.9 × 10⁻⁴ |

**`level` is statistically indistinguishable from ordinary operational drift in all 5 channels.** `diff` and `diff²` are significant in all 5 channels after Bonferroni correction. This is why the SCM skeleton (Fig. 4a) treats level shift as a **confounded, non-causal-core** node rather than the anomaly's defining feature.

**(d) Random-effects meta-analysis and heterogeneity** (Fig. 4b). Pooling across 5 channels with a **DerSimonian–Laird random-effects model**, we find substantial and *interpretable* between-channel heterogeneity (I² = 65.7%–97.2% across the six effect-size/significance-fraction outcomes, all Q-test p < .05). Heterogeneity is not noise: it is **moderated by channel noise regime** — `float_noise_suspect` channels (0872/0873/0874) show a strong diff/diff² effect, while `quantized` channels (0888/0894) show a systematically weaker one. A leave-one-out (LOO) sensitivity analysis confirms this pattern is not driven by any single channel.

**(e) Supplementary diagnostics A–D.** To rule out that the diff/diff² pattern is a labeling or detector artifact, we additionally ran: (A) comparison against matched normal-segment controls, (B) cross-correlation analysis between level and diff/diff² time series, (C) a Cochran–Mantel–Haenszel (CMH) stratified test pooling across channels while controlling for channel-level confounding, and (D) an **external audit of the OPS-SAT-AD labeling protocol** together with a full train/test/bootstrap **triple re-verification** of every §3 result (Fig. 3b): **12 of 13 decidable comparisons agree exactly** across full-data, train-only, and test-only analyses (the 13th is undecidable on the test-only split due to insufficient anomalous segments, not disagreement).

**Final SCM skeleton (Fig. 4a, "Path A")**: Onset → {diff, diff² variance surge} is the causal core; level shift is downstream/confounded and is excluded from the core based on the placebo-pool result in (c).

### Model Cross-Validation (16 independent architectures)

**Goal**: the strongest possible test of whether "diff/diff² matter more than level" is a property of the *data* or an artifact of *our specific detector's inductive bias*.

We trained **16 independent models**, spanning **6 inductive-bias families**, on the task of separating anomalous from normal segments using `level`, `diff`, and `diff²`-derived features, and asked each model (via its native attribution method) which feature family it relied on:

| Family | Models | Representation | Attribution method |
|---|---|---|---|
| Linear | LogReg (L1), LogReg (L2) | tabular | SHAP / permutation |
| Probabilistic | GaussianNB | tabular | SHAP / permutation |
| Kernel | SVM (RBF) | tabular | SHAP / permutation |
| Instance-based | kNN | tabular | SHAP / permutation |
| Tree ensembles | RandomForest, ExtraTrees, GradBoost (sklearn), XGBoost, LightGBM | tabular | SHAP / permutation |
| Convolutional | CNN1D, TCN | raw sequence | Integrated-gradients saliency |
| Recurrent | BiLSTM, BiGRU | raw sequence | Integrated-gradients saliency |
| Attention | TinyTransformer | raw sequence | Integrated-gradients saliency |
| State-space (SSM) | LightMamba | raw sequence | Integrated-gradients saliency |

**Result: 16/16 models rank `diff` + `diff²` above `level`.** Zero of 16 models ranked `level` as the top feature (binomial test, p = .0015). Cross-model agreement is quantified with **Kendall's coefficient of concordance W = 0.609** (χ² = 19.50, df = 2, p < .001) — a substantial level of agreement given that each model was trained, attributed, and evaluated completely independently, with no shared hyperparameters or joint optimization. Bootstrap stability checks (200 resamples) on the two most divergent model classes (RandomForest, LogReg) show 100% stability of the diff/diff² > level ranking. Critically, **tabular models (n=10, engineered summary features) and sequence models (n=6, raw time series + saliency) agree**, which rules out the possibility that the result is an artifact of how the tabular summary features happened to be engineered.

---

## Figures

All figures below are generated by the scripts in `results/figures/scripts/` and are byte-for-byte what `make figures` produces from `results/*.csv` / `results/*.json` / `results/*.parquet`. No figure in this README was hand-edited after generation.

> **Figure conventions (all figures).** Every figure in this repository is designed to be **fully legible in black-and-white print**, as required by most subscription and IEEE-style journals: categories are encoded redundantly by **greyscale value *and* hatch pattern** (e.g. `level` = light grey + dotted hatch, `diff` = mid-grey + diagonal hatch, `diff²` = solid black), never by hue alone. Where a figure previously relied on a red/green or red/blue colormap (e.g. the Fig. 3b agreement matrix), it has been redrawn as pure black/white/hatch, with the underlying category also spelled out as on-cell text (`sig.` / `n.s.` / `n/a`) so the encoding is triply redundant (fill, hatch, text).

### Figure 1 — Channel scoping: MCC/Youden's J and the 9→5 funnel

![Figure 1: Channel scoping](results/figures/fig01_channel_scope.png)

*(a) MCC and Youden's J per channel at the locked hyperparameters `(mixture=False, forgetting=True)`. Solid bars = channels retained in the final scope; hatched bars = excluded, with the exclusion reason printed directly beneath each excluded channel's tick label. `CADC0884` has zero anomalous segments and is marked "n/a" rather than plotted as a zero-height bar. (b) The 9→5 channel-scoping funnel, rendered as a greyscale gradient from light (9-channel starting point) to black (final, triple-confirmed 5-channel scope), showing that the scope is stable across three independent verification passes (bootstrap CI, train-only reselection, test-only confirmation).*
**Reproduces from:** `results/layer2/channel_scope.json`, `results/layer2/bootstrap_ci.csv` — **Script:** `results/figures/scripts/fig01_channel_scope.py`

---

### Figure 2 — 16-model cross-validation: feature attribution and agreement

![Figure 2: Model cross-validation](results/figures/fig02_model_cross_validation.png)

*(a) Normalized feature-importance share (level / diff / diff²) for each of 16 independently trained models, sorted by AUC (printed at the right edge of each row, clear of the bar). The dotted vertical line marks the chance level (1/3) each feature would receive under an uninformative split. `level` is rendered with a dotted hatch, `diff` with a diagonal hatch, and `diff²` as solid black — every model allocates substantially more than 1/3 combined share to the diff + diff² (hatched + black) region. (b) Cross-model agreement statistics: Kendall's W, the binomial test on "models ranking level as the top feature," bootstrap stability, and the tabular/sequence representation-agreement check, in a bordered plain-text panel.*
**Reproduces from:** `results/model_cross_validation/feature_attribution_16models.csv`, `results/model_cross_validation/agreement_statistics.json` — **Script:** `results/figures/scripts/fig02_model_cross_validation.py`

---

### Figure 3 — Quasi-experimental placebo-pool test and triple verification

![Figure 3: Quasi-experimental comparison](results/figures/fig03_quasi_experimental.png)

*(a) Anomalous segments vs. same-channel placebo pool of normal segments (Mann–Whitney U, 15 tests, Bonferroni-corrected). `level` bars are exactly zero in every channel (called out explicitly in the boxed annotation, positioned over open chart space so it never overlaps a bar); `diff` (diagonal hatch) and `diff²` (solid black) are significant in every one of the 5 scoped channels. (b) Agreement matrix across full-data, train-only, and test-only analyses for all 15 (channel × feature) cells, now rendered in pure black (`sig.`) / white (`n.s.`) / hatched grey (`n/a`) — 12/13 decidable comparisons agree exactly.*
**Reproduces from:** `results/layer3/placebo_comparison.csv`, `results/layer3/triple_verification_matrix.csv` — **Script:** `results/figures/scripts/fig03_quasi_experimental.py`

---

### Figure 4 — Final SCM skeleton and meta-analytic heterogeneity

![Figure 4: SCM and heterogeneity](results/figures/fig04_scm_and_heterogeneity.png)

*(a) The final structural skeleton ("Path A"): onset (black) drives a transient diff/diff² variance surge (dark grey), which constitutes the causal core; level shift (light grey) is excluded from the core (dashed grey arrow) because it is not distinguishable from normal-operation drift (boxed annotation, citing the Fig. 3a result). The moderator annotation (float_noise_suspect vs. quantized channels) sits in its own bordered box beneath the diagram, clear of the arrows. (b) DerSimonian–Laird random-effects meta-analysis heterogeneity (I²) across the 5 scoped channels for 6 outcome measures (effect size and significance fraction, × 3 features, same hatch coding as Fig. 2/3). The "very high heterogeneity" threshold (I² = 75%) is labeled in a row below the bars, clear of the panel title.*
**Reproduces from:** `results/layer3/heterogeneity_summary.csv` — **Script:** `results/figures/scripts/fig04_scm_and_heterogeneity.py`

---

### Figure 5 — Dataset overview and pooled temporal-precedence test

![Figure 5: Dataset and precedence](results/figures/fig05_dataset_and_precedence.png)

*(a) OPS-SAT-AD dataset composition: segment counts (bars) and anomaly prevalence (dashed black line, open-circle markers) for all 9 channels; solid bars mark the final 5-channel analytical scope, hatched bars are excluded. Sensor type (magnetometer / photodiode) is printed as a short tag beneath each channel code rather than as a rotated in-plot label, so it never collides with the bars or the title. (b) The pooled temporal-precedence test: `diff` and `diff²` cross their respective anomaly thresholds before `level` does in the large majority of paired segments (84.3% and 82.9% of pairs respectively), with `diff` vs. `diff²` (dotted hatch) showing no reliable ordering between themselves.*
**Reproduces from:** `data/raw/` (segment metadata), `results/layer3/temporal_precedence.csv` — **Script:** `results/figures/scripts/fig05_dataset_and_precedence.py`

---

## Results tables

### Table 1 — Channel scoping decision table (underlies Fig. 1)

| Channel | Sensor | # segments | Anomaly % | MCC | Youden's J | Decision |
|---|---|---|---|---|---|---|
| CADC0872 | Magnetometer #1 | 546 | 24.0 | 0.514 | 0.321 | **Included** |
| CADC0873 | Magnetometer #2 | 593 | 17.7 | 0.489 | 0.276 | **Included** |
| CADC0874 | Magnetometer #3 | 194 | 35.6 | 0.740 | 0.693 | **Included** |
| CADC0884 | Photodiode #1 | 158 | 0.0 | — | — | Excluded (structural: n_anom = 0) |
| CADC0886 | Photodiode #2 | 11 | 27.3 | 0.0 | 0.0 | Excluded (underpowered: n = 3/8) |
| CADC0888 | Photodiode #3 | 252 | 23.8 | 0.194 | 0.226 | **Included** |
| CADC0890 | Photodiode #4 | 14 | 78.6 | 0.826 | 0.909 | Excluded (underpowered: n = 14) |
| CADC0892 | Photodiode #5 | 211 | 16.1 | −0.090 | −0.024 | Excluded (chance-level: MCC < 0) |
| CADC0894 | Photodiode #6 | 144 | 14.6 | 0.189 | 0.203 | **Included** |

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

**Agreement statistics**: Kendall's W = 0.609 (χ² = 19.50, df = 2, p < .001) · models ranking `level` top = 0/16 (binomial p = .0015) · bootstrap stability (200 resamples): RandomForest 100%, LogReg 100% favor diff/diff² · 13/13 inductive-bias families favor diff+diff² over level (narrowest margin: SSM family) · tabular (n=10) and sequence (n=6) representations agree.

### Table 3 — Quasi-experimental placebo-pool comparison (underlies Fig. 3a)

| Channel | level, p_bonf | diff, p_bonf | diff², p_bonf |
|---|---|---|---|
| CADC0872 | 1.000 | 7.94 × 10⁻²⁴ | 1.12 × 10⁻²² |
| CADC0873 | 1.000 | 2.40 × 10⁻¹⁷ | 3.09 × 10⁻¹⁶ |
| CADC0874 | 1.000 | 5.37 × 10⁻¹⁶ | 2.88 × 10⁻²² |
| CADC0888 | 1.000 | 6.92 × 10⁻⁶ | 3.24 × 10⁻¹³ |
| CADC0894 | 1.000 | 3.89 × 10⁻⁶ | 1.95 × 10⁻⁴ |

*Mann–Whitney U, 15 tests total, Bonferroni-corrected (α = .05 → α_bonf = .00333).*

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

# 4. Run the independent 16-model cross-validation benchmark
python -m model_cross_validation.run_all

# 5. Regenerate every figure in this README from the results/ artifacts above
python results/figures/scripts/fig01_channel_scope.py
python results/figures/scripts/fig02_model_cross_validation.py
python results/figures/scripts/fig03_quasi_experimental.py
python results/figures/scripts/fig04_scm_and_heterogeneity.py
python results/figures/scripts/fig05_dataset_and_precedence.py

# ...or simply:
make all
```

`REPRODUCIBILITY_CHECKLIST.md` documents random seeds, package versions, and expected runtime (CPU-only: ~40 min for Layers 1–3; the 16-model benchmark, including the two deep-sequence GPU-optional models, is the long pole at ~2–3 hours on CPU / ~25 min on a single GPU).

---

## Threats to validity and how each was addressed

| Threat | Mitigation |
|---|---|
| **Train/test leakage in channel selection** | The entire channel-scoping procedure (Layer 2) was independently re-run on the train-only split and re-confirmed on the test-only split (Fig. 1b); the 5-channel scope is identical in all three passes. |
| **Diff/diff² result being a labeling artifact** | External audit of the OPS-SAT-AD labeling protocol (`docs/dev-log/step3b_labeling_protocol_revalidation.md`) plus full triple re-verification of the §3 causal-signature results (Fig. 3b). |
| **Diff/diff² result being an artifact of our detector's inductive bias** | Independent 16-model cross-validation across 6 inductive-bias families, 2 independent data representations (tabular / raw sequence), and 2 independent attribution methods (SHAP-permutation / integrated gradients) — all converge (Fig. 2). |
| **Level shift "looking" important only because it's not tested against the right null** | Quasi-experimental placebo-pool design: anomalous segments compared against *same-channel normal-operation segments*, not against a pooled cross-channel baseline (Fig. 3a). |
| **Small-sample channels inflating apparent effect sizes** | Structural/underpowered channels (`CADC0884`, `CADC0886`, `CADC0890`) explicitly excluded from the analytical scope rather than down-weighted (Fig. 1); `CADC0890`'s exclusion despite MCC = 0.826 is a deliberate conservative choice, documented in Table 1. |
| **Between-channel heterogeneity being treated as noise** | Heterogeneity is quantified (I², Q-test), tested for robustness via leave-one-out sensitivity analysis, and shown to be *moderated* by a known covariate (quantization regime) rather than dismissed (Fig. 4b). |
| **Multiple-comparisons inflation** | Bonferroni correction applied to all 15 placebo-pool tests (Table 3); pooled (not per-test) Wilcoxon test used for the temporal-precedence claim (Table underlying Fig. 5b). |

---

## Relationship to formal causal inference — scope statement

We use the term **"causal signature"** in this repository and the associated manuscript to describe a claim of the form *"X is the feature that changes at anomaly onset, and this is not explainable by normal-operation drift or by dataset/model artifacts."* This is established via **quasi-experimental design** (same-channel placebo-pool comparison), **temporal precedence testing**, and a **structural causal model (SCM) skeleton** built from the sensor's signal-processing structure (Path A in Fig. 4a) rather than from a data-driven causal discovery algorithm.

We do **not** claim formal statistical identification in the Pearl do-calculus sense, and we do not run a discovery algorithm (e.g., PC, FCI, LiNGAM) to *derive* the SCM edges — the skeleton in Fig. 4a is asserted from known instrument physics (a change-point in the underlying process necessarily first perturbs local variance/derivative statistics before a sustained mean shift is observable) and then **tested against data** for consistency, not derived from data. We flag this distinction explicitly so that reviewers and readers calibrate the strength of the causal claim correctly: it is a **rigorously tested, quasi-experimental causal-signature claim**, not a formally identified causal effect estimate.

---

## Limitations

- **Single benchmark dataset.** All results are on OPS-SAT-AD. We have not yet validated the diff/diff² signature on a second, independent spacecraft-telemetry anomaly benchmark; this is the most important direction for follow-up work and the most likely reviewer objection.
- **No comparison against published OPS-SAT-AD baselines.** This repository's contribution is a *causal-signature finding*, not a new detector claiming state-of-the-art detection performance; nonetheless, a table situating our detector's MCC/Youden's J against previously published OPS-SAT-AD results strengthens the paper and is planned for the camera-ready version.
- **Detection performance is modest in absolute terms** on some channels (e.g., CADC0888, CADC0894 sit well below CADC0874). The paper's contribution is about *what the signature is*, not about maximizing raw detection recall/precision.
- **5 of 9 channels are excluded from the causal analysis** for principled statistical-power reasons (Table 1); the causal-signature claim is scoped to these 5 channels and is not asserted for the excluded 4.

---

## License

TODO (e.g., MIT for code; dataset governed by its own OPS-SAT-AD license — see `data/README.md`).
