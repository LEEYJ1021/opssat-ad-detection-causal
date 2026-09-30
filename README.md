# opssat-ad-onset-signatures

**Dependence-Aware Evaluation of Onset Features for Alarm Design: Cross-Dataset Evidence from Spacecraft and Industrial Telemetry (OPS-SAT-AD, SMAP/MSL, SMD)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducible](https://img.shields.io/badge/results-fully%20reproducible-brightgreen.svg)](#reproducing-every-number-in-this-readme)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](#citation)

> **Status legend.** ✅ = computed and reproduced from the released scripts. Analyses that were **not performed in this release** are listed once, with the consequence for each claim, in §11.8; they are not marked inline as if they were results.

> **Summary.** Alarm thresholds for spacecraft telemetry are conventionally set on signal *level*, on the premise that an anomaly begins as a mean shift. We compare *level* with onset-relative change features (`diff`, `diff²`) on labeled anomaly onsets in three benchmarks: OPS-SAT-AD (9 ESA telemetry channels, 5 in scope), SMAP/MSL (82 NASA channels) and SMD (1,064 industrial server channels), each against same-channel placebo pivots. Because onset labels are nested in segments, channels or machines, every headline statistic is an onset-vs-placebo AUROC with a **cluster-bootstrap interval at the unit that repeats** (segment / channel / machine), and the analysis was passed through a fixed audit sequence: an independent replica of the data pipeline, resampling at the higher level (small-K), false-alarm-rate (FAR) matching over an extended grid, a detector-matched placebo for OPS-SAT-AD, a tie-handling audit of the precedence criterion, and a synthetic existence-proof.
>
> **What survived the audit.** (i) *Direction.* Δ(diff − level) AUROC = **+0.462 [0.409, 0.508]** (OPS-SAT-AD, position-matched placebo), **+0.122 [0.045, 0.196]** (SMAP/MSL), **+0.042 [0.016, 0.066]** (SMD). SMAP/MSL and SMD have ground-truth onsets, so this direction does not depend on an onset estimator. Because `level` falls below 0.5 in OPS-SAT-AD, the direction-agnostic contrast (level scored as max(AUC, 1 − AUC) = 0.562) is also reported: **+0.338**. (ii) *No incremental value of level.* Adding `level` to `diff`/`diff²` in an out-of-fold logistic model changes AUROC by −0.0007, −0.0068 and −0.0009. (iii) *Small-K robustness.* The two OPS-SAT-AD contrasts keep direction and size under two-level, wild-cluster and exact sign-flip resampling at the channel level (e.g. [0.282, 0.593] two-level), although with K = 5 no exact test can reject at 0.05 (minimum p = 0.0625), so they are read as direction and magnitude. (iv) *Operating value.* At a matched 5% FAR in SMD, a `level OR diff OR diff²` rule raises coverage from 15.9% to 17.2%, **+1.2 pp [0.4, 2.0]**. Over an extended grid of six FAR targets (0.5–20%) the fusion gain is positive in SMD at every target (+0.25 to +1.84 pp) with intervals excluding zero at 0.5%, 1%, 5% and 20% (Table R3b). In the 56-window SMAP/MSL sample the sign of the fusion gain is not stable across targets, so that sample does not rank fusion against single-derivative rules.
>
> **What did not survive, or was weakened.** (i) The pooled OPS-SAT-AD `level` reversal (AUROC 0.438) is significant only under segment-level resampling (p = 0.004); under channel-level schemes every interval includes 0.5 (Table R7b), so it is *inconclusive*, and it is driven by one channel (CADC0873). (ii) SMAP/MSL `level` is "weak", not "reliably detectable": its channel-cluster interval [0.497, 0.608] includes 0.5. (iii) **Part of the external `diff` advantage is definitional.** Replacing `level`'s pooled-SD denominator by the pre-onset SD (`level_prez`) raises `level` AUROC by +0.080 (SMAP/MSL) and +0.035 (SMD), after which `diff − level_prez` is +0.060 (p = 0.22) and +0.010 (p = 0.50): not distinguishable from zero (Table R4). (iv) **The OPS-SAT-AD derivative effect is much smaller against a detector-matched placebo.** With the stage-2 detector run on the 996 normal segments and its detected resets used as placebo pivots, pooled AUROC falls from 0.900 to **0.569 [0.518, 0.626]** (`diff`) and from 0.917 to **0.720 [0.675, 0.769]** (`diff²`); in CADC0888 and CADC0894 `diff` falls below 0.5 (0.136, 0.314). Pooled `level` is 0.382 [0.327, 0.441], so its direction-agnostic value (0.618) exceeds the detector-matched `diff` value: **the `diff > level` ordering in OPS-SAT-AD is not robust to detector matching, whereas `diff² > level` is** (Table R12). By the pre-specified rule (§11.8) the pooled `diff`/`diff²` intervals still exclude 0.5, so OPS-SAT-AD is retained as a case study, but its magnitude is detector-conditioned. (v) **The operator-baseline decomposition was an artifact of tie handling.** Simultaneous crossings (`t_level = t_diff`) make up 33–66% of external pairs and had been counted as `diff`-first. With ties excluded, baselines are 36.9% / 34.8% / 32.1% (all below 50%) instead of 44.2% / 56.5% / 72.7%, and the anomalous-minus-normal increment is +34.9 [18.6, 49.2] (OPS-SAT-AD), +6.1 [−14.3, 26.0] (SMAP/MSL) and −1.5 [−5.5, 3.5] (SMD): only OPS-SAT-AD excludes zero (Table 13). The decomposition is demoted to a descriptive control. (vi) The "0/16 models rank `level` first" result holds under the corrected (within-training-fold early stopping) protocol; under the earlier test-fold protocol one sequence model (CNN1D) ranks `level` first (Fig. 2, §11.7).
>
> **Methodological finding that generalizes.** The correct resampling unit depends on the estimand. In a synthetic two-level design (K = 5), a segment-level interval covers the size-weighted AUROC at 96.2% but the channel-mean AUROC at only 79.2%; channel-level procedures cover both at ≥ 97% but are conservative (Table R8). When cluster size is correlated with effect size, pooled AUROC is biased relative to the equal-weight AUROC (−0.049 at ρ = −0.8, ≈ 0 at ρ = 0; Tables R9, R10). In OPS-SAT-AD the observed retention–effect correlation (ρ = −0.60, p = 0.285, n = 5) points the same way but cannot confirm the mechanism. A second, related lesson concerns first-crossing criteria on integer-sample data: ties must be reported and their handling varied, because here they changed the sign of a headline contrast.
>
> **Not performed in this release.** A method-matched (SHAP) re-run of the ten tabular models and Kendall's *W* for the re-run arm (*W* is reported for the original arm only), tie sensitivity of the precedence criterion for OPS-SAT-AD (crossing-time inputs not used here), and channel-level resampling of the detector-matched intervals; see §11.8. We interpret the findings as an **associational, direction-level result with a modest, quantified incremental value**, not as evidence of a universally superior detector, a causal mechanism, or a general instrument-physics effect.

This repository is the code, data-provenance and figure-reproduction companion to **Paper 1** of a two-paper series built primarily on OPS-SAT-AD, with independent generalization testing on SMAP/MSL and SMD. It is **self-contained and independently evaluable** and does not depend on any other repository.

| This series | Repository | Status |
|---|---|---|
| **Paper 1 (this repo)** — Cross-dataset signature analysis | `opssat-ad-onset-signatures` | ✅ independent, no upstream deps |
| Paper 2 — Risk-aware alarm design (CVaR / conformal) | [`opssat-ad-risk-optimization`](https://github.com/USERNAME/opssat-ad-risk-optimization) | depends on this repo's `results/` artifacts |
| DSS prototype (not a paper) | [`opssat-ad-dss`](https://github.com/USERNAME/opssat-ad-dss) | depends on both papers' artifacts |

---

## Table of Contents

1. [Contribution at a glance](#contribution-at-a-glance)
2. [Scope and submission package](#scope-and-submission-package)
3. [Evidence ladder and claim scope](#evidence-ladder-and-claim-scope)
4. [Storyline of the investigation](#storyline-of-the-investigation)
5. [Onset provenance of each result](#onset-provenance-of-each-result)
6. [Headline results](#headline-results)
7. [Repository structure](#repository-structure)
8. [Datasets](#datasets)
9. [Methodology (OPS-SAT-AD primary pipeline)](#methodology-ops-sat-ad-primary-pipeline)
10. [External Validation / Generalization Study (SMAP/MSL, SMD)](#external-validation--generalization-study-smapmsl-smd)
11. [Robustness Re-Analysis](#robustness-re-analysis-cluster-robust-far-matched-and-sensitivity-checks)
12. [Analytical baseline: why differencing may respond faster](#analytical-baseline-why-differencing-may-respond-faster)
13. [Practical implications for alarm design](#practical-implications-for-alarm-design)
14. [Scope of the inferential claims](#scope-of-the-inferential-claims)
15. [Figures](#figures)
16. [Results tables](#results-tables)
17. [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
18. [Threats to validity](#threats-to-validity)
19. [Boundary conditions and scope of validity](#boundary-conditions-and-scope-of-validity)
20. [Research roadmap enabled by this release](#research-roadmap-enabled-by-this-release)
21. [Data and code availability](#data-and-code-availability)
22. [License](#license)

---

## Contribution at a glance

**The question.** Change-point and anomaly-detection pipelines for spacecraft and industrial telemetry treat a level shift as the default anomaly signature. Which feature family should an alarm be built on, and how much of an apparent advantage survives when the dependence structure of the labels, the choice of placebo, and the handling of ties are respected?

**The answer this study provides.** Across three benchmarks that differ in platform, sensor physics, labeling protocol and onset provenance, `diff`/`diff²` features separate labeled onsets from placebo pivots better than `level` on cluster-robust AUROC, with an advantage that is large in OPS-SAT-AD (position-matched placebo), moderate in SMAP/MSL and small in SMD. Applying the same audit sequence to every headline statistic also shows which readings do *not* hold up (pooled `level` reversal, the strength of the external `diff` advantage under an alternative `level` definition, the size of the OPS-SAT-AD `diff` effect under a detector-matched placebo, and the operator-baseline reading of the precedence criterion), and yields a reusable procedure for reporting feature-discrimination claims under nested labels.

**Seven contributions.**

1. **A cross-dataset directional result with an estimator-free arm.** `diff`/`diff²` exceed `level` in onset-vs-placebo AUROC in 3/3 datasets; in SMAP/MSL and SMD the onsets are ground-truth labels, so no onset estimator is involved (Table 12, Table R2, Fig. 6a).
2. **Cluster-robust quantification with a small-K check and a direction-agnostic variant.** Δ(diff − level) = +0.462 / +0.122 / +0.042 (segment / channel / machine clusters; Table R2). For OPS-SAT-AD (K = 5) the same contrasts are re-estimated under two-level, wild-cluster (Webb, Rademacher) and exact sign-flip resampling with unchanged direction and size (Tables R7, R7c). The direction-agnostic contrast is +0.338 rather than +0.462.
3. **Documented cases where the naive and the dependence-aware reading differ.** Pooled `level` reversal in OPS-SAT-AD: p = 0.004 at segment level, interval including 0.5 at channel level (Table R7b, Fig. 3a). SMAP/MSL `level`: pooled p_fdr = 0.031 versus a channel-cluster interval including 0.5 (Table R2). The two cases where both readings agree (SMD `level`; all `diff`/`diff²` contrasts under position-matched placebo) are equally reported.
4. **A detector-matched placebo for OPS-SAT-AD.** Running the stage-2 detector on the normal segments removes the asymmetry between detector-selected onsets and position-matched pivots. The pooled derivative AUROCs remain above 0.5 (`diff` 0.569, `diff²` 0.720) but are much smaller than under position-matched pivots, and `diff` reverses in two channels (Table R12, Fig. 3c).
5. **A FAR-matched measurement of the alarm-design payoff over an extended grid.** At 5% FAR, the `level|diff|diff²` fusion rule adds +1.2 pp [0.4, 2.0] coverage over `level` in SMD; the fusion threshold is calibrated so that the *joint* placebo FAR equals the target. Over six FAR targets the SMD gain is positive throughout (Table R3b, Fig. 6c). In the 56-window SMAP/MSL sample the gain changes sign across targets.
6. **An estimand-dependent resampling result, a pooling-bias identity, and a tie-handling audit.** Coverage of segment-, two-level and wild-cluster intervals against the size-weighted and channel-mean estimands (Table R8, Fig. 6b); the exact pair-decomposition identity (Table R9); and a demonstration that counting simultaneous crossings as `diff`-first produced a spurious operator-baseline gradient (Table 13, Fig. 7).
7. **An independent replica audit, leakage/duplication checks and disclosure of demoted and unperformed items.** From-scratch replica of the canonical-onset and placebo-pool pipeline (4,996/4,996 rows, max relative error 1.8×10⁻¹⁵); de-duplicated placebo pool; multi-seed grouped CV; row-level leakage positive control; a protocol comparison (test-fold vs within-fold early stopping) for sequence models (§11.2, §11.7). An early SMAP/MSL scheme that contaminated the pre-onset baseline was caught, discarded and documented (§10, `docs/dev-log/`).

**What this changes for practice.**

- **Add a derivative-variance channel to the alarm feature set, and expect a modest gain.** At a matched false-alarm rate the realistic gain is on the order of one percentage point of coverage in SMD (§11.4). In the 56-window SMAP/MSL sample the data do not separate fusion from single derivative rules.
- **Report cluster-robust intervals and the direction-agnostic AUROC before claiming a feature reversal or advantage**, and state which estimand (pooled vs. equal-weight) and which placebo (position-matched vs. detector-matched) the number refers to (§11.3, §11.9).
- **Do not read fixed-threshold precedence as early-warning evidence.** Report ties, use a normal-segment control and a FAR-matched comparison (§11.4, §11.6).
- **Prioritize by instrument type only provisionally.** Float-noise magnetometer channels show the strongest position-matched derivative signature, but this pattern is aliased with detector-selection retention (28–100%) and reverses for `diff` in the two photodiode channels under detector matching (§11.2, §11.9).

---

## Scope and submission package

**Scope fit.** The paper addresses a general methodological question in anomaly and onset detection: which signal statistic should carry the primary decision signal when an alarm is intended to identify the onset of an anomalous episode, and how should such a claim be evaluated when labels are nested in segments, channels or machines? The contribution is not tied to a particular spacecraft, sensor or detector. The analysis combines dependence-aware cluster-bootstrap AUROC with FAR-matched coverage to quantify statistical discrimination and operational relevance. The companion paper (`opssat-ad-risk-optimization`) addresses the subsequent problem of selecting alarm thresholds under explicit risk criteria (CVaR, conformal calibration); the present repository supplies the feature-choice inputs (`channel_scope.json`, `onset_posteriors.parquet`) and their justification.

**What is, and is not, claimed.** The paper does not claim a detector that outperforms published benchmarks (§19), nor a causal mechanism (§11.6, §14). It claims (a) a cross-dataset directional result, with estimator-free support in SMAP/MSL and SMD; (b) dependence-aware and FAR-matched quantification of the corresponding increment; and (c) an explicit map of where the evidence is robust, conditional or absent (§11.8).

**Reproducibility package.** The repository contains the analysis code, data-provenance scripts, the exact source scripts for the figures, development logs and the robustness re-analysis of §11. `docs/METHODS_SUPPLEMENT.md` is the extended methodological record; `docs/REPRODUCIBILITY_CHECKLIST.md` documents seeds, package versions and expected runtimes (§17). Every figure script asserts the sample sizes of its input CSVs and writes a `figure_manifest.json`. `results/README.md` contains the sample-size accounting table (`partE_n_accounting.csv`) referenced by every table and figure caption. §15 states, for each figure, whether it was rendered from analysis outputs or from inputs extracted from README tables.

---

## Evidence ladder and claim scope

The claims are graded by how far the evidence travels. Tiers are ordered by weight; Tier 3 and Tier 4 are secondary.

| Tier | Claim | Where it is supported | Status and boundary |
|---|---|---|---|
| **1 — Direction, cluster-robust** | `diff`/`diff²` separate labeled onsets from same-channel placebo pivots more strongly than `level` on a cluster-bootstrap AUROC at the segment (OPS-SAT-AD), channel (SMAP/MSL) or machine (SMD) level. | Table 12; **Table R2**; **Tables R7, R7c** (small-K); **Table R12** (detector-matched); Fig. 6a, Fig. 3c | ✅ Same direction in 3/3 datasets; two of three onset sources are estimator-free. With K = 5 the OPS-SAT-AD contrasts are direction and magnitude, not formal tests. **Qualifications:** (a) in the external datasets `diff − level_prez` is not distinguishable from zero (Table R4); (b) in OPS-SAT-AD, against a detector-matched placebo `diff²` remains well above `level` but `diff` (0.569) does not exceed the direction-agnostic `level` (0.618). |
| **2 — Operational, FAR-matched** | Adding derivative features raises coverage at a matched false-alarm rate. | **Tables R3, R3b, §11.4**, Fig. 6c | ✅ Fusion rule in SMD: +1.2 pp at 5% FAR, +0.8 pp at 1% FAR, positive at all six targets, interval excluding zero at four of six. Single-feature gains and the SMAP/MSL estimate (n = 56) are not stable. Not evaluated for OPS-SAT-AD. |
| **3 — Baseline-calibrated increment (descriptive, secondary)** | With simultaneous crossings excluded, the anomalous-minus-normal increment in `diff`-first ordering is +34.9 pp (OPS-SAT-AD), +6.1 pp (SMAP/MSL) and −1.5 pp (SMD); baselines are below 50% in all three. | Table 13, Fig. 7 (appendix-level) | **Descriptive only.** Only the OPS-SAT-AD interval excludes zero. The earlier baseline gradient (44.2 / 56.5 / 72.7%) is a tie-handling artifact. The criterion is not FAR-matched and does not distinguish a mean shift from a variance surge (Table R5). |
| **4 — OPS-SAT-AD instrument-level readings** | Pooled `level` is inconclusive (AUROC 0.438; interval includes 0.5 under channel-level resampling); CADC0873 shows a within-channel reversal; derivative effects are stronger in magnetometer than photodiode channels. | Layer 3; **Tables R1, R4, R7b, R7c, R12** | Conditional on the BOCPD onset estimator and on the placebo used; the pooled reversal is not robust; the CADC0873 reversal rests on segment clusters within one channel and is unresolved; sensor family, channel type and detector retention are aliased. Preliminary. |
| **Scope by design** | A causal mechanism linking onsets to derivative-statistic surges; superiority of a `diff`-based detector over published detectors; do-calculus identification. | — | Outside this paper. |

**Scope of detection results.** Layer 2 detection metrics (Table 1) serve channel scoping and lock the BOCPD hyperparameters. Separating *which feature carries onset information* (this paper) from *how to set the threshold under a risk criterion* (Paper 2) is a deliberate modular design.

---

## Storyline of the investigation

The investigation began with **OPS-SAT-AD** (ESA OPS-SAT mission, 9 telemetry channels, 2,123 expert-labeled univariate segments) and one question: is a level shift the operative anomaly signature, or do variance surges in the first and second differences (`diff`, `diff²`) carry more anomaly-relevant information?

The early working hypothesis, stated before any hypothesis testing, was the conventional `level → diff → diff²` ordering. The pooled Wilcoxon precedence test of §3.2 found the opposite pattern within OPS-SAT-AD. Every subsequent analysis was designed to stress-test that finding, first inside OPS-SAT-AD and then outside it. The stress test proceeded along seven lines, each aimed at a different class of confound:

1. **Is `level`'s lack of excess just a weaker null?** → same-channel placebo-pool comparison (§3.7). *Result: `level` shows no detectable excess over placebo in 5/5 channels; `diff`/`diff²` are significant in 5/5 (raw and de-duplicated pools).*
2. **Is the pattern a side effect of manual segment cutting?** → labeling-protocol audit (§3b). *Result: no quantitative cutting rule is documented; the question is open, and SMAP/MSL and SMD, with different protocols, bound its consequence for the directional claim.*
3. **Is the result an artifact of using the full dataset?** → train-only and test-only reproduction (§3b). *Result: 15/15 train-only and 12/13 decidable test-only calls match.*
4. **Is the attribution ranking specific to one learner?** → 16-model cross-validation (§9). *Result: 0/16 models rank `level` first in the original arm; 0/6 sequence models under the corrected re-run protocol. This controls the classifier side only.*
5. **Does the result depend on the onset estimator or on the placebo?** → SMAP/MSL and SMD with ground-truth onsets (§10) and a detector-matched placebo for OPS-SAT-AD (§11.9). *Result: the direction is confirmed with no estimator in the loop; in OPS-SAT-AD the derivative effect shrinks substantially under detector matching (`diff` 0.900 → 0.569, `diff²` 0.917 → 0.720).*
6. **Is the result specific to OPS-SAT-AD's instrument physics?** → the same external replication. *Result: the direction travels; the strict-specificity and reversal readings do not (Tables 12, 13).*
7. **Do the significance statements, the precedence decomposition and the 16-model benchmark survive independence, deduplication, small-K, FAR-matching, alternative-statistic and tie-handling checks?** → §11. *Result: the direction survives; the pooled `level` reversal does not; the size of the external `diff` advantage depends on the `level` definition; the precedence decomposition was produced by tie handling and is demoted to a descriptive control.*

Lines 1–3 and 6 converge within OPS-SAT-AD. Lines 5–6 carry the directional claim to independent datasets. Line 7 is a self-audit conducted after the original analysis, under the same disclosure standard as the SMAP/MSL contamination episode of §10.

This repository packages:

- The **signal-estimation → onset-estimation → signature-analysis** pipeline for OPS-SAT-AD (Layers 1–3).
- The **labeling-protocol audit** and **train/test/bootstrap triple re-verification** (Layer 3b).
- The **16-model cross-validation** benchmark and its agreement statistics.
- An **external-validation pipeline** for SMAP/MSL and SMD.
- A **robustness re-analysis** (§11): cluster-bootstrap re-estimation, small-K schemes, estimand-dependent coverage, FAR-matched comparison on an extended grid, detector-matched placebo, tie-sensitive precedence analysis, duplicate-pool and leakage audit, `level`-statistic sensitivity, and a synthetic regime map.
- **Figures regenerated from their source scripts** (`results/figures/*.py`), with the provenance of each stated in §15.
- **Raw development logs** (`docs/dev-log/`), including discarded ablations and corrections.

---

## Onset provenance of each result

| Result | Onset source | Depends on BOCPD onset estimate? |
|---|---|---|
| OPS-SAT-AD Layer 3 (§3.1–3.7), triple verification, 16-model cross-validation | BOCPD on Kalman-filter innovations; canonical onset = median of 200 Monte Carlo draws | **Yes** |
| OPS-SAT-AD detector-matched placebo (Table R12) | Same detector applied to 996 normal segments; detected resets are placebo pivots | **Yes** (both arms) |
| SMAP/MSL Stage 1–2 | Ground-truth onset indices (`labeled_anomalies.csv`) | **No** |
| SMD Stage 1–2 | Ground-truth timestep-level 0/1 labels | **No** |

**Design of the OPS-SAT-AD onset front end.** A local-linear-trend Kalman filter absorbs a persistent mean shift into its state within a few samples and leaves a short-lived transient in the innovations. BOCPD on those innovations is therefore a transient-sensitive front end. It yields a scoreable window for 178 of 386 anomalous segments (46.1%): 176 high-reliability, 1 medium, 1 low (177/178 = 99.4% high-or-medium). The scoreable subset is a quality-gated selection. Placebo pivots in §3.7 are position-matched resamples on normal segments while anomalous onsets are detector-selected; this asymmetry was the main open threat to the OPS-SAT-AD readings and is addressed by the detector-matched analysis of §11.9.

**How the results are weighted.** The onset-independent evidence (SMAP/MSL, SMD) carries the directional claim (Tier 1). OPS-SAT-AD-specific readings (Tier 4) are conditional on the estimator. OPS-SAT-AD's large position-matched effect (AUROC 0.90) is treated as case-study evidence whose magnitude is detector-conditioned (detector-matched AUROC 0.57 for `diff`, 0.72 for `diff²`).

---

## Headline results

| Claim | OPS-SAT-AD evidence | Outside OPS-SAT-AD (SMAP/MSL, SMD) |
|---|---|---|
| Channel scope: 9 → 5 channels, identical across 3 independent derivations | Fig. 1, Table 1 | N/A |
| `diff`/`diff²` separate onsets from placebo more strongly than `level` (**direction, cluster-robust**) | Fig. 6a, Table R2: **Δ +0.462 [0.409, 0.508]** (position-matched); direction-agnostic **+0.338**; channel-level schemes [0.282, 0.593] (two-level), [0.350, 0.574] (exact); sign stable in every leave-one-channel-out fit (R7, R7c); K = 5, so direction and magnitude, not a formal test | **3/3 datasets**: +0.122 [0.045, 0.196] (SMAP/MSL), +0.042 [0.016, 0.066] (SMD); with `level_prez` the external contrasts are +0.060 (p = 0.22) and +0.010 (p = 0.50) (R4) |
| OPS-SAT-AD under a detector-matched placebo | Pooled AUROC `diff` 0.569 [0.518, 0.626], `diff²` 0.720 [0.675, 0.769], `level` 0.382 [0.327, 0.441]; `diff` below 0.5 in CADC0888 (0.136) and CADC0894 (0.314); direction-agnostic `level` 0.618 > `diff` 0.569 (R12, Fig. 3c) | — |
| `level` incremental value once `diff`/`diff²` are present | ΔAUROC −0.0007 (negligible) | −0.0068 (n.s.), −0.0009 (n.s.) |
| `level` versus placebo | Pooled AUROC 0.438 [0.401, 0.477] at segment level; **intervals include 0.5 under channel-level schemes (R7b, Fig. 3a)** → inconclusive; CADC0873 within-channel AUROC 0.289 (two-sided p_bonf = 0.002, segment-level, not re-tested at channel level) | SMAP/MSL: weak (channel-cluster AUROC 0.554 [0.497, 0.608] includes 0.5; pooled p_fdr = 0.031). SMD: small but reliable (0.534 [0.521, 0.546]) |
| Adding derivative features at a matched false-alarm rate | Not evaluated (no raw-telemetry FAR pipeline) | SMD fusion **+1.2 pp [0.4, 2.0]** at 5% FAR, +0.8 pp [0.3, 1.4] at 1% FAR, positive at all six targets; SMAP/MSL +3.6 pp [0.0, 11.3] at 5% (n = 56), sign not stable across targets (R3, R3b, Fig. 6c) |
| Precedence decomposition (descriptive; simultaneous crossings excluded) | +34.9 pp [18.6, 49.2] (n = 39/198) | SMAP/MSL +6.1 [−14.3, 26.0] (n = 22/5,074); SMD −1.5 [−5.5, 3.5] (n = 1,219/2,770). Baselines all below 50%. Not FAR-matched; does not separate mean shift from variance surge (R5) |
| Estimand-dependent resampling | — | Simulation: segment-level interval covers channel-mean AUROC at 79.2%, size-weighted at 96.2% (R8, Fig. 6b) |
| Magnetometer vs photodiode derivative effects | Fig. 3a; aliased with sensor family and with retention 28–100% (R1); reverses for `diff` in the photodiode channels under detector matching (R12) | 1/82 float-noise channels in SMAP/MSL; no significant effect in SMD's 17 |
| Placebo-pool result replicates across full / train / test | 12/13 decidable agree (Fig. 3b); replica max error 1.8×10⁻¹⁵ (R1) | — |
| 0/16 models rank `level` first | Original arm 0/16 (Fig. 2, Table 6). Sequence re-run, corrected protocol: 0/6; test-fold protocol: 1/6 (CNN1D) (R6) | — |
| No documented quantitative segment-cutting rule | — | N/A |

**Reading this table.** Rows 2–3 are the load-bearing results, with the qualifications stated in those rows. Row 5 documents the case where naive and dependence-aware readings of `level` differ. Row 6 gives the operating size. The precedence row is descriptive and secondary.

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

> **Note on `docs/dev-log/`.** These files are the conversational research logs kept during development, included **verbatim** for transparency (file names retain the original "causal" wording). `step_external_validation.md` also records: (a) the SMAP/MSL sliding-onset scheme that contaminated the pre-onset baseline for windows longer than the baseline (diagnosed by a jointly-valid-candidate check and a window-length stratification, then discarded); (b) the 150→5,000-sample search-window sensitivity sweep (n = 40→41; `diff`-first 67.5%→65.9%, original counting); and (c) the consistency-gate results of the re-analysis.

---

## Datasets

### OPS-SAT-AD (primary)

European Space Agency's OPS-SAT experimental nanosatellite telemetry anomaly-detection benchmark (Ruszczak, Kotowski, Evans & Nalepa, *Scientific Data* 12:710, 2025).

- **9 channels**: 3 magnetometer channels (`CADC0872`, `CADC0873`, `CADC0874`) + 6 photodiode channels (`CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894`).
- **2,123 univariate segments**, each expert-labeled *nominal* or *anomalous*. Labels are at segment level; onset position within a segment is estimated via BOCPD (§ Layer 2).
- Per-channel segment counts and anomaly prevalence vary sharply (0% to 78.6%); see Fig. 1, Fig. 5a and Table 1. The five in-scope channels hold 386 anomalous segments (131 + 105 + 69 + 60 + 21).
- An official train/test partition (`train ∈ {0,1}`, ≈75%/25%) is used for train-fit/test-evaluate verification.
- Segment boundaries were determined by manual expert annotation using ESA's OXI tool; no quantitative cutting rule is documented (§ Layer 3b).
- Not redistributed; `data/download_opssat_ad.sh` fetches it from its Zenodo record.

### SMAP/MSL (external validation)

NASA SMAP and MSL spacecraft telemetry, released with the Telemanom benchmark (Hundman et al., KDD 2018).

- **82 channels** (55 SMAP + 27 MSL), each a long continuous series with **ground-truth onset/offset indices** per anomaly window (`labeled_anomalies.csv`).
- Each `.npy` file has multiple columns; only the first (telemetry) column is used; the rest are one-hot command context and are excluded from all `diff`/`diff²` computation.
- Anomaly window lengths are highly variable (median 120, mean 616, max 4,217 samples).
- A triviality filter motivated by Wu & Keogh (2021) excludes extreme, visually obvious point outliers (17 of 105 windows; 88 non-trivial windows remain).
- Not redistributed; `data/download_smap_msl.sh` fetches the public release.

### SMD (Server Machine Dataset, external validation)

Industrial server telemetry from the OmniAnomaly benchmark (Su et al., KDD 2019).

- **28 machines × 38 dimensions = 1,064 channels**, each with timestep-level (0/1) ground-truth labels; 1,038 channels enter the channel-level counts of Stage 1.
- Included as a domain-generalization check; SMD is *supporting*, not primary, evidence.
- Anomaly events (327) are nested within 28 machines and replicated across dimensions, which is why §11.3 uses the **machine** as the cluster unit.
- Not redistributed; `data/download_smd.sh` fetches the public release.

---

## Methodology (OPS-SAT-AD primary pipeline)

### Layer 1 — Signal Estimation

**Goal**: obtain a denoised, well-calibrated state estimate for each channel before any change-point logic is applied.

**1.1 Channel typing.** Each channel is classified from the distribution of first differences (Δ) within nominal segments only. Thresholds were set from this dataset's diagnostics and **re-fitted, not reused,** on SMAP/MSL and SMD:

- **`quantized`** if the fraction of exact-zero Δ is ≥ 0.10 (step = 25th percentile of non-zero |Δ|).
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

Channel type maps one-to-one onto sensor family in this dataset (three magnetometers are `float_noise_suspect`; the photodiodes are `quantized` except CADC0890, which is `continuous` and out of scope). §11.2 shows both are also aliased with detector-selection retention.

**1.2 Noise estimation.** `r = Var(Δ)/2`, `q = Var(Δ²)/6` from pooled nominal-segment differences. A Gaussian-mixture alternative did not improve train macro-MCC (Table 8), so the simple estimator was retained.

**1.3 Quantization floor.** For `quantized` channels, `step²/12` is added to the observation-noise variance.

**1.4 Hierarchical Bayesian shrinkage.** Efron–Morris-type empirical-Bayes shrinkage stabilizes channels with very few nominal points (CADC0886: n=8; CADC0890: n=3). Shrunk estimates are in Table 11.

**1.5 State estimation.** A local-linear-trend Kalman filter is fit per channel using the shrunk `r`, `q`. Standardized innovations `z_t = (y_t − ŷ_t)/√S_t` are the sole input to Layer 2.

---

### Layer 2 — Onset Estimation and Channel Scoping

**2.1 Change-point model.** BOCPD (Adams & MacKay 2007) on `z_t`, Normal-Inverse-Gamma conjugate predictive model, hazard rate `1/250`, with a forgetting extension (`κ_max = 40`).

**2.2 Full 2×2 ablation.**

| Combination | Train macro-MCC (6 scoring channels) |
|---|---|
| mixture=False, forgetting=True (**locked**) | **0.308** |
| mixture=True, forgetting=True | 0.275 |
| mixture=False, forgetting=False | 0.238 |
| mixture=True, forgetting=False | 0.199 |

Train-only reselection reproduces the locked combination.

**2.3 Channel scoping.** MCC and Youden's *J* at locked hyperparameters, with exclusion criteria applied in order: structural (zero anomalous segments), underpowered (< 5 anomalous or < 5 normal), chance-level (MCC ≤ 0). See **Table 1** and Fig. 1. The rule is conservative by construction: CADC0890 has the highest point-estimate MCC (0.83) yet is excluded for having only 3 nominal segments.

**2.4 Triple verification.** The 5-channel scope was re-derived via bootstrap CI, train-fit/test-eval and test-only bootstrap CI; all three agree (Fig. 1b).

**2.5 Problem-channel diagnosis (CADC0890/0892/0894).** Hypothesis B (BOCPD forgetting deficiency) was rejected; Hypothesis A (noise-model deficiency) is supported for CADC0890/CADC0894 under the mixture estimator; CADC0892 is explained by neither (steady-state kurtosis 63.32 drives spurious resets). This localizes the detector-tuning difficulty to channel-specific noise modeling, separately from the feature-choice question.

---

### Layer 3 — Signature Analysis

**Goal**: determine what changes at an OPS-SAT-AD anomaly onset and whether the answer is an artifact of labeling, channel choice, placebo choice or leakage. All Layer 3 analyses operate on the 5 scoped channels and on **BOCPD-estimated onsets**.

**3.1 Onset-uncertainty propagation (Monte Carlo).** 200 onset candidates per segment (seed = 42); pre/post comparisons for `level` (Cohen's *d*) and for `diff`, `diff²` (log-variance ratio). 178/386 anomalous segments (46.1%) yield a scoreable window; 176/178 (98.9%) are high-reliability and 177/178 (99.4%) high-or-medium. **Sample accounting:** 178 scoreable → 177 high-or-medium → **168 complete-case** events; the nine dropped between the last two have a non-positive pre- or post-window variance in the `diff` windows (CADC0873: 1, CADC0888: 3, CADC0894: 5), so `diff`/`diff²` are undefined for them (`partE_n_accounting.csv`). The CADC0894 `diff` results therefore rest on n = 16 events.

Within-segment, `level` is significant in 54–92% of segments per channel and `diff`/`diff²` in 2–76%; read alone this favors `level`; §3.3 reconciles it with the precedence test.

**3.2 Temporal precedence test.** The canonical onset (median of 200 MC draws) defines the pre/post split; the first post-onset |z| > 3 crossing time is recorded per series. *These counts and p-values follow the original pairing rule; see the tie-handling note in §3.3 and §11.6.*

| Comparison | n pairs | *p* (pooled Wilcoxon) | Result |
|---|---|---|---|
| `level` vs. `diff` | 70 | 1.4×10⁻⁵ | `diff` first in **84.3%** of pairs |
| `level` vs. `diff²` | 70 | 3.2×10⁻⁵ | `diff²` first in **82.9%** of pairs |
| `diff` vs. `diff²` | 161 | 0.368 (n.s.) | treated as one derivative family (see reporting note under Table 10) |

**3.3 Reconciling §3.1 and §3.2.**
- **(a) Temporal-profile classification**: `diff` is transient in 98% of segments, `diff²` in 94%, `level` a near-even mix (50%/45%). A first-crossing test rewards transient, onset-localized surges.
- **(b) Early-window sensitivity**: `level`'s significance rate drops 0.864→0.633 in the early window, a larger relative loss than `diff`/`diff²`.
- **(c) Normal-segment control.** Repeating the test on normal segments with resampled pivots (mean position ratio ≈ 0.569), `level` precedes `diff` in 55.8% of normal pairs (vs 15.7% anomalous) and `diff²` in 74.3% (vs 17.1%) under the original pairing rule. The canonical-pair re-estimation of Table 13 (39/198 pairs, simultaneous crossings excluded) gives an anomalous-minus-normal increment of **+34.9 pp [18.6, 49.2]**. Both are **descriptive**: the criterion is not FAR-matched, cannot separate a mean shift from a variance surge (§11.6), and its OPS-SAT-AD pair counts under the two rules (70/224 vs 39/198) have not been reconciled (§11.8, item 6).

**3.4 Formal group comparison and diff-vs-diff² tie resolution.** Two-proportion *z*/Fisher exact tests and a channel-stratified CMH test confirm the orderings (CMH p ≤ 4.33×10⁻¹⁵). For `diff` vs `diff²` the cross-correlation lag in anomalous segments differs from zero (p = 0.00094) but anomalous-vs-normal lag distributions do not differ (p = 0.906); the two are therefore treated as one derivative-variance family.

**3.5 Random-effects meta-analysis (DerSimonian–Laird, k = 5).** *I²* ranges 65.7%–97.2% (Table 4, Fig. 4). With k = 5, *I²* and Q are descriptive; they motivate reporting channel-level results next to pooled ones.

**3.6 Leave-one-out (LOO) decomposition.** Removing CADC0872 eliminates all `level`-|*d*| heterogeneity; removing CADC0874 eliminates all `diff²`-|*d*| heterogeneity; removing CADC0894 nearly halves the pooled `diff`/`diff²` significance fraction.

**3.7 Quasi-experimental placebo-pool comparison.** Each anomalous segment's canonical-onset effect is compared, by one-sided Mann–Whitney *U*, against a placebo pool of resampled pivots on normal segments (15 tests, Bonferroni).

*How to read the statistics.*
- Each feature is tested on its **own natural statistic** (|*d*| for `level`; log-variance ratio for `diff`/`diff²`). The p-values are distinguishability measures; the common-scale comparison is in §11.3 (Table R2).
- `level` p_bonf = 1.000 is the Bonferroni cap (raw p × 15 ≥ 1), not an estimate of equality; the correct reading is *no detectable excess over placebo*, and no equivalence test was run. The null is not uniform: CADC0873 shows a within-channel reversal (two-sided p_bonf = 0.002, segment clusters), whereas the pooled reversal is not robust to channel-level resampling (Table R7b, Fig. 3a).
- **Table 3 was computed on the raw placebo pool (4,980 rows, 20.7% duplicates).** The de-duplicated recomputation (3,951 rows) changes the p-values by at most a factor of about three and no significance call (Table 3, right-hand columns).
- **The placebo pivots are position-matched resamples on normal segments.** A detector-matched placebo is analyzed in §11.9 (Table R12).

*Result.* `level` shows no detectable excess over placebo in 5/5 channels; `diff`/`diff²` are significant in 5/5 (p_bonf < 2×10⁻⁴) against the position-matched pool. `diff`/`diff²` strength is higher in the three magnetometer channels (0.78–0.89 significant-segment fraction) than in the two photodiode channels (0.12–0.42); type, sensor family and retention are aliased.

> **Independent verification (§11.2).** A from-scratch replica of the canonical-onset extraction, effect computation and placebo-pool assembly, built from raw segment time series, matches the saved analysis dataset row for row: **4,996/4,996 rows, maximum relative error 1.8×10⁻¹⁵.** This rules out a class of end-to-end pipeline bugs (onset-index mismatch, feature-definition drift, placebo-pool corruption) as an explanation for any result in this section.

**Interpretive consequence (OPS-SAT-AD-scoped).** The within-segment `level` significance of §3.1 is not reproduced against placebo, while the `diff`/`diff²` surge is against a position-matched placebo and, more weakly, against a detector-matched one (§11.9). The cause of `level`'s weak showing is **unresolved**: the pooled-SD denominator of |*d*| is *not* an explanation in OPS-SAT-AD (Table R4), and the segment-level reversal is not robust at channel level (Table R7b). These readings are conditional on the onset estimator.

**Note on the former structural working model.** Earlier versions drew an onset → variance surge → level-shift diagram. It has been removed from the figure and from the argument: it was asserted from signal-processing structure rather than identified, and the precedence criterion on which its ordering rested cannot distinguish a mean shift from a variance surge (§11.6). `scm_skeleton.py` is retained for provenance only.

---

### Layer 3b — Labeling-Protocol Audit & Train/Test Triple Verification

**3b.1 Labeling-protocol audit.** The normal-segment control found anomalous-segment onsets skewed toward the back half of their segment (mean position ratio ≈ 0.569). No quantitative segment-cutting rule was found in the primary publication, its SoftwareX companion or either preprint; segments were cut manually via ESA's OXI tool. The origin of the skew remains an open provenance question. Its consequence is bounded because the directional claim is confirmed on SMAP/MSL and SMD, whose labels follow different protocols.

**3b.2 Train-only and test-only reproduction of §3.7.**

| Slice | Coverage | Result |
|---|---|---|
| Train-only | 178,504/240,979 rows (74.1%); 126 anomalous segments | 15/15 (channel×feature) significance calls match full-data |
| Test-only | 62,475/240,979 rows (25.9%); 51 anomalous segments | 12/13 decidable calls match; 1 borderline (CADC0888 `diff`, p = 0.051, monotone power loss); 2 not decidable (CADC0894, n = 4 < 5) |

*Segment accounting.* Train (126) + test (51) = 177 = high-or-medium segments; the full-data analysis adds one low-tier segment (178). The 177 → 168 complete-case step is explained in §3.1.

---

### Model Cross-Validation (16 architecturally diverse models)

**Goal**: test whether the feature-importance ranking (`diff`/`diff²` above `level`) depends on a particular learner's inductive bias.

**What this controls.** All models solve the same binary problem, whose labels come from BOCPD-estimated canonical onsets. The check controls the **classifier side** (architecture, representation, attribution method); onset-estimation dependence is addressed by the external replication. The models share data and labels, so agreement statistics are descriptive.

**Task and labels.** Canonical-onset anomalous segments vs. resampled placebo pivots: 4,996 tabular rows (168 positive / 4,828 negative; raw placebo pool) and 2,733 sequence-window rows (2,505 after de-duplication). Out-of-fold AUCs only order the models; the quantity of interest is the attribution share.

**Two data representations.**
- *Tabular* (10 models): the three engineered features of §3.7. Attribution: SHAP / permutation importance (method recorded per model).
- *Raw sequence* (6 models): per-window z-normalized 3-channel series; out-of-fold AUC 0.969–0.994 (original arm). Attribution: Integrated Gradients.

Shares are normalized within each model and compared by rank and by the 1/3 uniform reference, not by magnitude across methods.

**Model zoo — 16 models, 9 coarse families (13 finer groupings):**

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

A 17th model (MLP_shallow) was removed by the fit-quality screen (AUC = 0.403, failed optimization). Headline statistics are over **16 models**.

**Result (original arm).**
- **0/16 models rank `level` first** (`diff` first in 5, `diff²` in 11; binomial reference (2/3)¹⁶, p = .0015, descriptive). Tabular 0/10, sequence 0/6.
- **15/16 give `level` less than 1/3**; the exception is LightMamba (0.428).
- Kendall's *W* = 0.609 (χ² = 19.50, df = 2, p = 5.8×10⁻⁵), original arm.
- Bootstrap stability (200 resamples): 100% of resamples favor `diff`+`diff²` over `level` for RandomForest and LogisticRegression.

Because the design has three features, "`diff`+`diff²` > `level`" is expected 2:1 under a uniform-share null; the informative quantities are the top-rank count and the size of the `level` share.

> **Leakage, duplication and re-run audit (§11.7).** The tabular dataset was re-derived from raw time series (exact match) and re-run with a de-duplicated placebo pool, multi-seed stratified-group CV, and a row-level CV as a leakage positive control (AUC moves by at most 0.017; positive control |Δ| ≤ 0.009, non-systematic). The six sequence models were re-run under two protocols: *as run* (early stopping on the test fold) and *corrected* (early stopping within the training fold, 60 epochs). Under the corrected protocol **0/6 sequence models rank `level` first** and `level` has a share below 1/3 in 6/6 (`diff` first in 4, `diff²` in 2). Under the as-run protocol CNN1D ranks `level` first (0.371 vs 0.362 for `diff²`) and two models (CNN1D, LightMamba) exceed 1/3, so the sequence-side counts depend on the training protocol (Fig. 2). Kendall's *W* is computed for the original arm only; a method-matched (SHAP) re-run of the tabular models and a *W* for the re-run arm were **not performed in this release** (§11.8).

**Scope note.** Run on OPS-SAT-AD only; a reduced replication on SMAP/MSL or SMD is in the roadmap.

---

## External Validation / Generalization Study (SMAP/MSL, SMD)

### Motivation and positioning

The external datasets provide the **onset-independent** evidence: onsets are ground-truth labels, and placebo pivots are compared against human-labeled onsets rather than detector-selected ones. We re-ran the placebo-pool comparison (§3.7) and the temporal-precedence test with normal-segment control (§3.2–3.3c) on SMAP/MSL and SMD, with the directional hypothesis (`diff`/`diff²` > `level`) fixed in advance. The 16-model benchmark and the labeling-protocol audit remain OPS-SAT-AD-internal.

### Stage 0 — Adaptations required by the label formats

Ground-truth onset locations remove the need for Monte Carlo onset propagation. Five adaptations are documented in `docs/dev-log/step_external_validation.md`:

- **SMAP/MSL command-column exclusion.** `diff`/`diff²` use only the telemetry column.
- **Channel re-typing.** Typing thresholds re-fitted per dataset.
- **Multiple-comparisons correction.** Bonferroni is inappropriate for 82 and 1,064 channels; Benjamini–Hochberg FDR is used, with a **confirmatory** family (dataset-pooled, 3 tests) and an **exploratory** family (channel-type-stratified) corrected separately because the latter is nested in the former.
- **Triviality filtering.** Windows whose values exceed 20 nominal SDs are excluded from the primary test (`is_trivial_anomaly`).
- **Sample-size-matched design.** SMAP/MSL has a median of 1 window per channel (max 3); per-channel tests are infeasible (no channel reaches n ≥ 5), so dataset-pooled and type-stratified tests are primary, with per-channel counts for SMD.

### Stage 1 — Placebo-pool comparison (external replication of §3.7; Fig. 6)

**Table 12 — Placebo-pool comparison across datasets** (pooled p-values are nominal one-sided tests for excess over placebo: windows are nested in channels and machines; the cluster-robust version is Table R2)

| Dataset | Scope | `level` | `diff` | `diff²` |
|---|---|---|---|---|
| **OPS-SAT-AD** | channel-level (n = 5) | 0/5 significant (p_bonf capped at 1.000) | 5/5 (all p < 2×10⁻⁴) | 5/5 (all p < 2×10⁻⁴) |
| **SMAP/MSL** | dataset-pooled (88 windows) | p_fdr = 0.031 | p_fdr = 4.7×10⁻⁹ | p_fdr = 1.3×10⁻⁶ |
| **SMD** | dataset-pooled (11,493 windows ≈ 327 events × ≤ 38 dimensions) | p_fdr = 8×10⁻³⁸ | p_fdr = 3.7×10⁻¹⁶⁷ | p_fdr = 1.4×10⁻¹⁶⁶ |

**Direction.** `diff`/`diff²` clear placebo by larger margins than `level` in all three datasets. **The pooled p-values treat non-independent windows as exchangeable and are descriptive of direction only**; Table R2 and Fig. 6a give the calibrated comparison. Outside OPS-SAT-AD, `level` carries a small component above placebo in the pooled test; in SMAP/MSL, however, the channel-cluster interval for `level` AUROC includes 0.5, so the component is best described as weak there and small-but-reliable in SMD.

**Channel-level breadth (SMD, descriptive; Fig. 6b).** 97/1,038 channels (9.3%; Wilson 95% 7.7–11.3%) are significant for `level`, 142/1,038 (13.7%; 11.7–15.9%) for `diff`, and 145/1,038 (14.0%; 12.0–16.2%) for `diff²`: the derivative features flag about 1.5× as many channels.

**Channel-type-stratified results (exploratory family).** In SMAP/MSL, `continuous` channels show `level` n.s. (p_fdr = 0.35) with `diff`/`diff²` significant, the OPS-SAT-AD pattern, while `quantized` channels show `level` weakly significant. `float_noise_suspect` cannot be tested in SMAP/MSL (1/82 channels, 3 windows) and shows no significant effect for any feature in SMD (162 windows across 17 channels; `diff`/`diff²` medians are negative there). The moderator hypothesis is untestable externally.

### Stage 2 — Temporal precedence and normal-segment control (external replication of §3.2–3.3c; Fig. 7)

**Methodological self-audit 1 (onset sliding).** An early attempt to raise power by sliding onset candidates within each SMAP/MSL window contaminated the pre-onset baseline whenever the window (median 120) exceeded the baseline length (20): shifted candidates placed part of the anomaly inside the baseline. A jointly-valid-candidate check and a window-length stratification diagnosed this. The reported results use the **ground-truth onset only** with an **adaptive, non-sliding post-onset search window**, whose upper bound was shown by a 150→5,000-sample sweep not to induce censoring bias. Of 88 non-trivial windows, only 40 have all three features crossing within the window.

**Methodological self-audit 2 (ties).** Crossing times are integer sample indices, so `t_level = t_diff` occurs often: in **45% / 33%** of SMAP/MSL anomalous / normal pairs and **66% / 61%** of SMD pairs (`stage2_pairs.csv`). The first version of Table 13 counted a tie as "`diff` precedes `level`". That version is reproduced exactly from the pair file (SMAP/MSL 67.5% vs 56.5%; SMD 76.6% vs 73.5%) and is reported below as a sensitivity row, but it is not the primary analysis: a tie carries no information about ordering. The primary analysis excludes ties; a second sensitivity scores ties as 0.5. Intervals are cluster bootstraps (SMAP/MSL: channel, K = 63; SMD: machine, K = 28; OPS-SAT-AD: canonical pairs treated as independent, K = 237).

**Table 13 — Precedence ordering and normal-control baseline (descriptive; primary analysis, simultaneous crossings excluded)**

| Dataset | Regime | n pairs | `diff` precedes `level` (95% cluster CI) |
|---|---|---|---|
| **OPS-SAT-AD** | anomalous | 39 | 71.8% [56.4, 84.6] |
| **OPS-SAT-AD** | normal control | 198 | 36.9% [30.3, 43.4] |
| **SMAP/MSL** | anomalous | 22 | 40.9% [19.0, 62.5] |
| **SMAP/MSL** | normal control | 5,074 | 34.8% [28.7, 41.7] |
| **SMD** | anomalous | 1,219 | 30.6% [26.4, 36.3] |
| **SMD** | normal control | 2,770 | 32.1% [29.9, 34.2] |

**Increment (anomalous − normal), with tie-handling sensitivity** (percentage points; 95% cluster-bootstrap CI)

| Dataset | Primary: ties excluded | Sensitivity: ties = 0.5 | Sensitivity: ties → `diff`-first (original) | Reading |
|---|---|---|---|---|
| OPS-SAT-AD | **+34.9 [18.6, 49.2]** (p_boot = 0.002) | not computed | +40.1 [29.4, 50.8] (nominal, original pairing rule, 70/224) | Baseline < 50%; increment excludes zero |
| SMAP/MSL | **+6.1 [−14.3, 26.0]** (p_boot = 0.64) | +5.1 [−6.0, 16.3] | +11.0 [−4.0, 23.9] | Interval spans zero under every rule |
| SMD | **−1.5 [−5.5, 3.5]** (p_boot = 0.49) | +0.4 [−1.2, 2.1] | +3.0 [−0.9, 6.4] | Interval spans zero under every rule; sign depends on tie handling |

*Notes.* (i) Primary rows are from `partD_table13_cluster_bootstrap.csv`; sensitivity rows for the external datasets were recomputed from `stage2_pairs.csv` (1,000 cluster resamples, seed 42; the primary rows reproduce to within the resampling noise of the interval endpoints). (ii) The OPS-SAT-AD sensitivity rows other than the original nominal value need the OPS crossing-time files and are not part of this release. (iii) The previously reported SMD normal-control count (7,304 pairs, 72.7%) is not reproduced by `stage2_pairs.csv`, which has 7,105 normal pairs (73.5% under the original counting rule); the anomalous count (3,612) and the SMAP/MSL counts (40 / 7,596) match. (iv) The OPS-SAT-AD pair counts of this table (39/198) differ from those of Tables 9–10 and Fig. 5b (70/224, original pairing rule); the difference has not been reconciled (§11.8, item 6).

**Interpretation.**
- With ties excluded, `diff` precedes `level` in fewer than half of non-tied pairs at placebo pivots in **all three datasets** (32–37%). The earlier statement that the baseline rises from 44% to 73% across datasets, and that the normal-control reversal is specific to OPS-SAT-AD, was produced by the tie convention and is withdrawn.
- Only the OPS-SAT-AD increment excludes zero. In the external datasets the increment is not distinguishable from zero and, in SMD, changes sign with the tie convention.
- **This criterion has limited diagnostic power.** It uses a fixed |z| > 3 threshold whose placebo false-alarm rate is 46.4% (`level`) and 37.8% (`diff`) in SMAP/MSL and 43.7% / 27.2% in SMD; it conditions on pairs where features cross; and in synthetic AR(1) series with φ ≥ 0.8 a **pure level shift** already yields `diff`-first in 74–92% of pairs (Table R5). The decomposition is therefore a calibration control, not evidence for a variance-surge mechanism.

### Summary of what generalizes and what is regime-specific

| | OPS-SAT-AD | SMAP/MSL | SMD |
|---|---|---|---|
| `diff`/`diff²` more placebo-separating than `level` (cluster-robust direction) | ✓ against position-matched placebo (also at channel level, K = 5; R7). Against detector-matched placebo: `diff²` ✓, `diff` not above direction-agnostic `level` (R12) | ✓ | ✓ |
| Same contrast against `level_prez` (pre-SD statistic) | +0.447 (p ≤ 0.002) | +0.060 (n.s.) | +0.010 (n.s.) |
| `level` shows positive discrimination | no; pooled reversal inconclusive; CADC0873 reversed | weak (interval includes 0.5) | small, reliable |
| Anomalous-vs-normal `diff`-first increment (descriptive; ties excluded) | +34.9 pp [18.6, 49.2] | +6.1 pp, spans zero | −1.5 pp, spans zero |
| Baseline below 50% in normal control (ties excluded) | ✓ (36.9%) | ✓ (34.8%) | ✓ (32.1%) |
| Magnetometer > photodiode pattern | ✓ position-matched (aliased with retention); `diff` reverses in photodiode channels under detector matching | not testable | no effect in 17 channels |
| FAR-matched coverage gain from fusion | not evaluated | +3.6 pp [0.0, 11.3] at 5% (n = 56); sign unstable across targets | ✓ +1.2 pp at 5% FAR; positive at all six targets |

Per-dataset artifacts are in `results/external_validation/`.

---

## Robustness Re-Analysis: Cluster-Robust, FAR-Matched, and Sensitivity Checks

### 11.1 Motivation and scope

The analyses above establish direction from per-feature rank tests and pooled p-values. Four questions remain before that direction can support an alarm-design recommendation: (i) do the statements survive a resampling unit that respects the dependence structure (segments within channels; windows within machines)? (ii) how large is the FAR-matched benefit, and is it specific to the pre-specified operating points? (iii) are the two anomalies around `level` (the reversal, most visibly CADC0873, and the weak external `level`) artifacts of the statistic used for `level`? (iv) does the OPS-SAT-AD effect depend on the asymmetry between detector-selected onsets and position-matched placebo pivots, and does the precedence criterion depend on its tie convention? A fifth, methodological question is which estimand each interval addresses (§11.3).

This section is built, where possible, directly from raw time series rather than from aggregated CSVs, so that it does not inherit earlier errors. Every number traces to the scripts in §17 and to artifacts in `results/**` and `results_care_v8/` (index in `results/README.md`).

### 11.2 Data-consistency gate for OPS-SAT-AD

**Table R1 — OPS-SAT-AD consistency gate** ✅

| Check | Result |
|---|---|
| Replica vs. saved 16-model dataset (4,996 rows) | Exact row-for-row match; max relative error 1.8×10⁻¹⁵ |
| Replica vs. saved canonical effect values (168 events) | Max relative error 7.8×10⁻¹⁵ |
| Replica placebo pool (raw / de-duplicated) | 4,980 / 3,951 rows; matches reported counts |
| Replica's Table 3 re-test | Same significance pattern in all 15 channel×feature cells |
| Attrition funnel | 386 → 178 scoreable (46.1%) → 177 high-or-medium (99.4%) → 168 complete-case (nine `diff`-undefined segments: CADC0873 1, CADC0888 3, CADC0894 5) |
| Channel-level retention (scoreable / anomalous) | CADC0872 32.1% (42/131), CADC0873 27.6% (29/105), CADC0874 72.5% (50/69), CADC0888 60.0% (36/60), CADC0894 100.0% (21/21) |
| Pooled AUROC (complete-case n = 3,991) | level 0.438, diff 0.900, diff² 0.917 |
| Stratified-by-channel AUROC (Simpson check) | level 0.434, diff 0.908, diff² 0.937; no material distortion |
| MC-median vs canonical onset (level AUROC, per channel) | 0.463 vs 0.468, 0.276 vs 0.289, 0.468 vs 0.470, 0.424 vs 0.454, 0.558 vs 0.574 (raw-pool denominators; the complete-case canonical values used in Fig. 3a are 0.468, 0.289, 0.470, 0.476, 0.593): the Monte Carlo median is not the source of the `level` reversal |
| Table 3 placebo pool | computed on the raw pool (per-channel n_placebo 1,495 / 1,485 / 625 / 760 / 615 = raw sizes); de-duplicated recomputation in Table 3 |

*Row-count note.* **3,951** = de-duplicated placebo-pool rows (4,980 raw, 20.7% duplicated). **3,991** = 168 anomalous + 3,823 de-duplicated complete-case placebo rows. **4,996** = 168 + 4,828 raw complete-case placebo rows (16-model tabular data). **2,733 / 2,505** = sequence windows before / after de-duplication.

**Reading.** The exact match rules out silent pipeline bugs. The three magnetometer channels, which carry the strongest derivative signature, have the lowest retention (27.6–32.1% vs 60–100%), so the instrument-type pattern is aliased with detector selectivity.

### 11.3 Cluster-robust AUROC: the load-bearing statistic

Each dataset's onset-vs-placebo separation is re-estimated as an AUROC with a cluster bootstrap (1,000 resamples) at the unit that repeats: **segment** (OPS-SAT-AD), **channel** (SMAP/MSL), **machine** (SMD; channel-level reported as a secondary check).

**Table R2 — Cluster-robust AUROC and paired contrasts (95% cluster-bootstrap CI; OPS-SAT-AD against the position-matched placebo)** ✅

| Dataset | Cluster unit (K) | AUROC `level` | AUROC `diff` | AUROC `diff²` | Δ(diff − level) | Δ(diff² − level) |
|---|---|---|---|---|---|---|
| OPS-SAT-AD | segment (K = 1,163; 168 anomalous events) | 0.438 [0.401, 0.477] | 0.900 [0.875, 0.924] | 0.917 [0.890, 0.943] | **+0.462 [0.409, 0.508]** | **+0.479 [0.427, 0.527]** |
| SMAP/MSL | channel (K = 81; 88 windows) | 0.554 [0.497, 0.608] | 0.677 [0.600, 0.746] | 0.642 [0.568, 0.713] | **+0.122 [0.045, 0.196]** | **+0.088 [0.008, 0.164]** |
| SMD | machine (K = 28; 327 events) | 0.534 [0.521, 0.546] | 0.576 [0.553, 0.599] | 0.576 [0.554, 0.597] | **+0.042 [0.016, 0.066]** | **+0.042 [0.017, 0.066]** |
| SMD (reference) | channel (K = 1,064; 327 events) | 0.534 [0.523, 0.546] | 0.576 [0.566, 0.585] | 0.576 [0.566, 0.585] | +0.042 [0.028, 0.054] | +0.042 [0.028, 0.055] |

**Direction-agnostic variant** (`level` scored as max(AUC, 1 − AUC); `diff`/`diff²` are above 0.5 in every row and unchanged):

| Dataset | `level`, direction-agnostic | Δ(diff − level) | Δ(diff² − level) |
|---|---|---|---|
| OPS-SAT-AD | 0.562 [0.523, 0.599] | **+0.338** | **+0.355** |
| SMAP/MSL | 0.554 (level ≥ 0.5, unchanged) | +0.122 | +0.088 |
| SMD | 0.534 (unchanged) | +0.042 | +0.042 |

*The direction-agnostic OPS-SAT-AD contrast removes the part of +0.462 that arises because `level` lies below 0.5. The `level` interval is the mirror of the segment-level interval; intervals for the direction-agnostic contrasts are in `cluster_robust_contrasts.csv`. The SMAP/MSL AUROCs are on 88 windows (R2 sample); FAR-matched analyses use 56 and Table R4 uses 68 channels (see `partE_n_accounting.csv`). Under the detector-matched placebo the OPS-SAT-AD values are those of Table R12.*

*SMD's 327 events are counted once per machine; the 11,493 window-level pairs of Table 12 are these events replicated across up to 38 dimensions, so the machine (K = 28) is the primary cluster. The OPS-SAT-AD row resamples segments (K = 1,163) and does not reflect between-channel heterogeneity (I² = 66–97%, Table 4).*

**Does `level` add anything once `diff`/`diff²` are in the model?** Out-of-fold, group-held-out logistic model, `[level, diff, diff²]` vs `[diff, diff²]`:

| Dataset | Δ AUROC (3-feature − 2-feature) | 95% CI | p (cluster bootstrap) |
|---|---|---|---|
| OPS-SAT-AD (segment) | −0.0007 | [−0.0014, −0.0001] | 0.03 |
| SMAP/MSL (channel) | −0.0068 | [−0.0381, 0.0241] | 0.70 |
| SMD (machine) | −0.0009 | [−0.0143, 0.0129] | 0.93 |

`level` contributes no positive incremental discrimination in any dataset. The OPS-SAT-AD change (|Δ| < 0.001) is negligible in size.

#### Small-K check (OPS-SAT-AD, K = 5 channels)

Segment-level resampling does not reflect between-channel heterogeneity. Table R7 re-estimates the two contrasts with resampling at channel level. With K = 5 no exact test can reject at α = 0.05: the sign-flip enumeration has 2⁵ = 32 patterns, so the smallest attainable two-sided p is 2/32 = 0.0625 (0.0606 with the +1 correction). The intervals are read as direction and magnitude.

**Table R7 — Δ AUROC under alternative resampling schemes (OPS-SAT-AD, position-matched placebo)** ✅

| Scheme | Unit (K) | Δ(diff − level) [95% CI] | Δ(diff² − level) [95% CI] |
|---|---|---|---|
| Point estimate | — | +0.462 | +0.479 |
| Single-level bootstrap (R2) | segment (1,163) | [0.409, 0.508] | [0.427, 0.527] |
| Two-level bootstrap | channel > segment (5) | [0.282, 0.593] | [0.327, 0.600] |
| Wild cluster, Webb 6-point | channel (5) | [0.342, 0.582] | [0.379, 0.578] |
| Wild cluster, Rademacher | channel (5) | [0.334, 0.590] | [0.380, 0.578] |
| Exact sign-flip enumeration (2⁵ = 32) | channel (5) | [0.350, 0.574] | [0.384, 0.574] |

*The two-level bootstrap is conservative (simulated coverage ≥ 99%, Table R8). From the intervals above, single-level widths are 0.099 (diff) and 0.100 (diff²) and two-level widths are 0.311 and 0.273 (2.7–3.1×).*

**Table R7b — Pooled `level` AUROC (0.438) under the same schemes: a case where the naive and the dependence-aware reading differ** ✅

| Scheme | 95% CI | p vs 0.5 | Excludes 0.5 |
|---|---|---|---|
| Single-level (segment) | [0.401, 0.477] | 0.004 | yes |
| Two-level (channel > segment) | [0.350, 0.543] | 0.244 | no |
| Wild cluster, Webb | [0.364, 0.510] | 0.151 | no |
| Exact enumeration | [0.370, 0.506] | 0.182 | no |

> The pooled `level` reversal is significant only when segments are treated as the independent unit. With channel-level resampling it is not distinguishable from chance. This does not re-test the within-channel CADC0873 result (AUROC 0.289), which rests on segment clusters inside one channel (Fig. 3a). Read as inconclusive, not as evidence of no discrimination.

**Table R7c — Leave-one-channel-out (OPS-SAT-AD)** ✅

| Channel dropped | AUROC `level` | Δ(diff − level) | Δ(diff² − level) |
|---|---|---|---|
| none (full) | 0.438 | 0.462 | 0.479 |
| CADC0872 | 0.426 | 0.448 | 0.476 |
| CADC0873 | 0.486 | 0.390 | 0.417 |
| CADC0874 | 0.437 | 0.475 | 0.489 |
| CADC0888 | 0.433 | 0.495 | 0.494 |
| CADC0894 | 0.415 | 0.490 | 0.513 |

> Both contrasts keep their sign in every fit (LOO range 0.390–0.495 and 0.417–0.513). Pooled `level` moves toward 0.5 only when CADC0873 is removed.

#### Supporting simulations: which estimand does each interval cover, and how does pooling bias the estimate?

**Table R8 — Coverage of 95% intervals, synthetic two-level design (K = 5, 500 replications, MC SE ≈ 0.010)** ✅

| Procedure | Coverage, size-weighted target | Coverage, channel-mean target | Mean width |
|---|---|---|---|
| Single-level (inner) | 96.2% | **79.2%** | 0.066 |
| Two-level | 99.6% | 99.4% | 0.188 |
| Wild cluster (outer) | 98.2% | 97.4% | 0.163 |

*Reading:* the single-level interval is near nominal for the pooled (size-weighted) estimand and under-covers the channel-mean estimand; channel-level procedures cover both but are conservative. **The appropriate resampling unit is determined by the estimand**, which is why Tables R2 and R7 are both reported (Fig. 6b, left).

**Table R9 — Pooled vs. cluster-equal-weight AUROC (simulation, 300 replications per row)** ✅

| Size–effect correlation ρ | Mean (pooled − equal-weight) | SD |
|---|---|---|
| −0.8 | −0.0488 | 0.031 |
| −0.4 | −0.0245 | 0.032 |
| 0.0 | +0.0003 | 0.030 |
| +0.4 | +0.0224 | 0.031 |
| +0.8 | +0.0469 | 0.029 |

*The pair-decomposition identity — pooled AUROC = [Σₖ n₁ₖn₀ₖ·AUCₖ + cross-cluster pair terms] / (N₁N₀), with the within-cluster weighted mean minus the equal-weight mean equal to a weighted covariance — holds to 0 (pooled) and 2.8×10⁻¹⁷ (weighted-mean vs covariance form) on test data, and is enforced by `layer3_signature_analysis/tests/test_pooled_identity.py`. The cross-cluster term is verified only approximately (SD of AUC across target clusters < 0.02). The sign of the bias in the simulation matches the identity in 5/5 conditions. This is an identity plus simulation, not a general proof of bias, and it does not say which estimand is correct.*

**Table R10 — Endogenous cluster size (retention) and pooled-AUROC bias** ✅ simulation; observed correlation descriptive

| Mechanism (K = 8, 500 reps) | Pooled bias (mean, SD) | Equal-weight bias (mean, SD) |
|---|---|---|
| Size correlated with effect (ρ = −0.8) | −0.0487 (0.0300) | −0.0004 (0.0208) |
| Size independent of effect | +0.0003 (0.0266) | −0.0001 (0.0222) |

*Observed OPS-SAT-AD (K = 5): retention vs. `diff` AUROC gives Spearman ρ = −0.60 (p = 0.285); the retention-weighted mean `diff` AUROC is 0.864 vs 0.885 equal-weight (difference 0.021). The reported p = 0.285 is the t-approximation (exact two-sided permutation p = 0.35); either way, with n = 5 the correlation is compatible with the simulated mechanism but does not confirm it; the simulation shows what bias would follow if the mechanism operated.*

**Table R11 — K-sensitivity of interval widths (SMD, machine subsets; descriptive)** ✅

| K (machines) | Reps | Width single | Width hierarchical | Ratio hier/single | Ratio wild/single |
|---|---|---|---|---|---|
| 5 | 20 | 0.082 | 0.148 | 1.79 | 1.22 |
| 12 | 20 | 0.067 | 0.113 | 1.68 | 1.10 |
| 20 | 20 | 0.065 | 0.102 | 1.57 | 0.90 |
| 28 | 1 | 0.064 | 0.090 | 1.41 | 0.80 |

*(K = 8, 16, 24 in `P3_K_sensitivity_summary.csv`.) The hierarchical/single ratio decreases with K (log–log slope −0.11, p = 0.004). Subsets overlap and K = 28 is a single full-data replicate, so this is a descriptive trend, not a convergence result; the hierarchical interval remains ≈1.4× the single-level width at K = 28. Naive (single/hierarchical) resampling used a cluster-proportional subsample of at most 20,000 rows; wild and exact procedures used all rows.*
