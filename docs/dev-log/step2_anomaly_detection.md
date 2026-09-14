# Step 2 — Anomaly Detection (Change-Point Model + Channel Scoping)

**Pipeline stage:** Layer 2 (`layer2_anomaly_detection/`)
**Purpose:** Convert the Layer 1 state estimate into a calibrated change-point/anomaly score, and rigorously determine which channels are analyzable at all.
**Input:** Standardized innovations `z_t` from the Layer 1 Kalman filter (see `step1_signal_estimation.md`).
**Status:** Locked. Consumed by Layer 3 (`step3_causal_analysis.md`).

---

## 1. Change-point model

Bayesian Online Change-Point Detection (BOCPD; Adams & MacKay, 2007) is run on `z_t` using a Normal-Inverse-Gamma conjugate predictive model:

```
mu0 = 0, kappa0 = 1, alpha0 = 1, beta0 = 1
hazard = 1 / 250        (constant across all channels)
```

**Forgetting extension.** Once a run's effective sample size `kappa` exceeds `kappa_max = 40`, the conjugate sufficient statistics (`kappa, alpha, beta`) for that run-length are rescaled proportionally toward the cap. This addresses a failure mode of vanilla BOCPD: without forgetting, long nominal stretches make the posterior over-confident, suppressing sensitivity to genuine later changepoints.

**Onset rule.** Onset is declared at the first time step `t` where the MAP run-length collapses from > 5 to ≤ 2. Onset index = `t − run_length(t)`. The posterior over the top-30 most probable run-lengths at confirmation is retained (`posterior_over_onset`) for Monte Carlo uncertainty propagation in Layer 3.

## 2. 2×2 ablation (mixture-noise × forgetting)

All four combinations of `{mixture noise estimator: on/off} × {forgetting: on/off}` were run to completion. The two factors act through **disjoint mechanisms on disjoint channel subsets**:

| Channel type | Dominant lever | Evidence |
|---|---|---|
| `float_noise_suspect` (872/873/874) | **forgetting** | Disabling forgetting nearly halves CADC0874 recall (0.725 → 0.319); smaller consistent drops on 872/873. Mixture-vs-naïve makes little difference. |
| quantized/continuous (888/890/892/894) | **noise estimator** | Mixture estimator inverts CADC0890's MCC sign (0.826 → −0.213) and degrades CADC0894. Forgetting has little effect on 888/890/892, modest effect on 894. |

Per-channel MCC across all four combinations (channel-type in parentheses):

| Channel | (mix=F, forget=T) — **locked** | (mix=T, forget=T) | (mix=T, forget=F) | (mix=F, forget=F) | Type |
|---|---|---|---|---|---|
| CADC0872 | **0.514** | 0.507 | 0.396 | 0.371 | float_noise_suspect |
| CADC0873 | 0.489 (tie) | 0.489 | 0.385 | 0.385 | float_noise_suspect |
| CADC0874 | 0.740 (tie) | 0.740 | 0.521 | 0.521 | float_noise_suspect |
| CADC0888 | 0.194 (tie) | 0.107 | 0.107 | 0.194 | quantized |
| CADC0890 | 0.826 (tie) | −0.213 | −0.213 | 0.826 | continuous |
| CADC0892 | −0.090 (best, still <0) | −0.158 | −0.223 | −0.090 | quantized |
| CADC0894 | 0.189 | 0.086 | 0.096 | **0.203** | quantized |

**Locked configuration:** `mixture = False`, `forgetting = True`. It is at or within ≈ 0.014 (≈ 7%) of the best combination for every channel except CADC0894 (where the margin to the single best is 0.014), and either substantially underperforms or inverts sign for CADC0872 and CADC0890 under alternative combinations. Treated as Pareto-near-optimal, not uniquely optimal for every channel — reported explicitly rather than concealed.

**In-sample-tuning check.** The 4-combination comparison above was initially scored on the same full dataset later used for reporting — a mild in-sample-tuning risk given only 9 channels. Re-run with the combination **selected on the train split only** (macro-averaged MCC over channels with ≥ 5 anomalous and ≥ 5 normal segments), then evaluated on the held-out test split:

| Combination | Train macro-MCC |
|---|---|
| mixture=False, forgetting=True | **0.308** |
| mixture=True, forgetting=True | 0.275 |
| mixture=False, forgetting=False | 0.238 |
| mixture=True, forgetting=False | 0.199 |

Train-only selection reproduces the same locked combination — supportive, though with only 9 channels not conclusive, evidence that the original choice was not scored-on-the-reporting-set overfitting.

## 3. Channel scoping (MCC / Youden's J)

Raw false-alarm rate alone is inadequate (ignores recall). MCC (0 under chance regardless of class balance; Chicco & Jurman, *BMC Genomics* 21:6, 2020) and Youden's *J* are used instead, computed at the locked hyperparameters.

**Decision table (9 → 5 channels):**

| Channel | Sensor | n_segments | Anomaly % | Recall | FA rate | MCC | Youden's J | Disposition |
|---|---|---|---|---|---|---|---|---|
| CADC0872 | Magnetometer 1 | 546 | 24.0 | 0.321 | 0.000 | 0.514 | 0.321 | **Included** |
| CADC0873 | Magnetometer 2 | 593 | 17.7 | 0.276 | 0.000 | 0.489 | 0.276 | **Included** |
| CADC0874 | Magnetometer 3 | 194 | 35.6 | 0.725 | 0.032 | 0.740 | 0.693 | **Included** |
| CADC0884 | Photodiode 1 | 158 | 0.0 | n/a | 0.241 | n/a | n/a | Excluded — structural (0 anomalous segments) |
| CADC0886 | Photodiode 2 | 11 | 27.3 | 0.000 | 0.000 | 0.000 | 0.000 | Excluded — underpowered (n = 3/8) |
| CADC0888 | Photodiode 3 | 252 | 23.8 | 0.617 | 0.391 | 0.194 | 0.226 | **Included** |
| CADC0890 | Photodiode 4 | 14 | 78.6 | 0.909 | 0.000 | 0.826 | 0.909 | Excluded — underpowered (n = 11/14), despite high point estimate |
| CADC0892 | Photodiode 5 | 211 | 16.1 | 0.971 | 0.994 | **−0.090** | −0.024 | Excluded — below chance |
| CADC0894 | Photodiode 6 | 144 | 14.6 | 1.000 | 0.797 | 0.189 | 0.203 | **Included** |

**Corrected framing note.** An earlier informal proposal to exclude both CADC0892 and CADC0894 on raw false-alarm-rate grounds (both > 0.7) was superseded once MCC/Youden's J were computed: CADC0894's very high recall (1.000) offsets its high false-alarm rate (positive, non-trivial MCC = 0.189, comparable to CADC0888's 0.194); CADC0892's high recall (0.971) is fully offset by its 0.994 false-alarm rate (negative MCC — it fires on almost everything). **Only CADC0892 is excluded on these grounds; CADC0894 is retained.**

## 4. Triple verification

The 5-channel scope (872/873/874/888/894) was independently re-derived three ways:

1. **Bootstrap CI (2,000 resamples, full data).** MCC 95% CIs:

   | Channel | MCC median | 95% CI | CI excludes 0? |
   |---|---|---|---|
   | CADC0872 | 0.514 | [0.444, 0.577] | Yes |
   | CADC0873 | 0.489 | [0.403, 0.564] | Yes |
   | CADC0874 | 0.744 | [0.635, 0.832] | Yes |
   | CADC0888 | 0.194 | [0.074, 0.311] | Yes |
   | CADC0890 | 0.826 | [0.603, 1.000] | Yes (but n=14, see below) |
   | CADC0892 | −0.090 | [−0.274, 0.053] | **No — reinforces exclusion** |
   | CADC0894 | 0.189 | [0.146, 0.230] | Yes |

2. **Train-fit, test-evaluate.** Parameters re-estimated on train split only, evaluated on held-out test split:

   | Channel | Recall (full) | Recall (test-only) | FA (full) | FA (test-only) |
   |---|---|---|---|---|
   | CADC0872 | 0.321 | 0.375 | 0.000 | 0.000 |
   | CADC0873 | 0.276 | 0.226 | 0.000 | 0.000 |
   | CADC0874 | 0.725 | 0.826 | 0.032 | 0.000 |
   | CADC0888 | 0.617 | 0.750 | 0.391 | 0.346 |
   | CADC0890 | 0.909 | 1.000 | 0.000 | n/a (0 normal segments in test split) |
   | CADC0892 | 0.971 | 1.000 | 0.994 | 0.978 |
   | CADC0894 | 1.000 | 1.000 | 0.797 | 0.786 |

3. **Test-only bootstrap MCC CI.** All five included channels' test-only 95% CIs exclude 0 (e.g., CADC0888 [0.092, 0.528], CADC0874 [0.715, 0.962]); CADC0892's test-only estimate (MCC = 0.054, CI [0.000, 0.096]) remains near-zero. Full-data MCC medians fall inside the corresponding test-only CI for every included channel, confirming the two slices are not mutually contradictory. CADC0884 and CADC0890 are un-evaluable on the test slice (0 of the relevant class present) — this **reinforces**, and does not newly decide, their existing exclusion.

**Final locked scope:** CADC0872, CADC0873, CADC0874, CADC0888, CADC0894 (5 channels). CADC0884 (structural), CADC0886 (underpowered), CADC0890 (underpowered, reinforced by test-split un-evaluability), CADC0892 (below chance, reinforced by test-split near-zero MCC) are excluded from Layer 2 reporting and from all of Layer 3.

## 5. Problem-channel diagnosis (CADC0890 / 0892 / 0894)

Two hypotheses were tested for why these three channels still show high false-alarm rates under the locked configuration:

- **Hypothesis A (noise-model deficiency):** steady-state standardized-innovation SD ≫ 1 or fraction |z| > 3 elevated.
- **Hypothesis B (BOCPD forgetting mechanism):** steady-state SD ≈ 1 despite high false alarms → problem is the onset-confirmation rule, not the noise model.

| Channel | Steady-state SD | Steady-state kurtosis | Fraction \|z\|>3 |
|---|---|---|---|
| CADC0890 | 0.945 | 0.37 | 0.000 |
| CADC0892 | 0.626 | 63.32 | 0.008 |
| CADC0894 | 0.978 | 34.27 | 0.026 |

**Result:** Hypothesis B is rejected — forgetting on/off produces negligible change in steady-state SD or false-alarm rate for these three channels. Hypothesis A is supported for CADC0890/CADC0894 specifically under the *mixture* estimator (steady-state SD inflates to 2.25/2.88 under mixture=True vs. 0.945/0.978 under the locked mixture=False), i.e., resolved by the Layer 1 noise-estimator choice, not by re-tuning BOCPD.

**CADC0892 is explained by neither hypothesis.** Its steady-state SD is close to 1 under either noise estimator, yet its false-alarm rate remains ≈ 0.994–1.000 regardless. Direct inspection of the run-length trajectory (`map_run_length`) on CADC0892's *normal* segments shows **99.4% of them (176/177) trigger at least one spurious reset** (run-length collapsing from > 5 to ≤ 2), averaging 1.93 resets per segment. This is consistent with the channel's extreme steady-state kurtosis (63.32): CADC0892 is "usually quiet, occasionally spikes," a pattern the shared onset-confirmation rule (and the common `hazard_lambda = 250` across channels) is not well matched to.

**CADC0894 shows the same qualitative failure mode at a markedly lower rate:** 79.7% of normal segments (98/123) show ≥ 1 reset, averaging 1.24 resets per segment (steady-state kurtosis 34.27, lower than 892's). This is a genuine but weaker version of the same instability — **not** evidence that 892 and 894 share a single root cause; treating them identically in narrative would overstate the similarity (see the `frac_z_gt3` canonicalization note below).

**Canonicalization note.** Two different computational conventions for "fraction of extreme z-values" were used across diagnostic passes (burn-in-excluded pooled steady-state vs. per-segment-averaged including burn-in), giving different absolute numbers for 892 vs. 894 (0.008 vs. 0.026 in the canonical burn-in-excluded convention; a different pair of values under the per-segment convention). Both conventions agree on **direction** (894 > 892 in extreme-value fraction under either convention) but should not be quoted interchangeably; the canonical convention (burn-in excluded, pooled) is the one reported in the table above.

## 6. Outputs consumed downstream

- `channel_scope.csv`, `channel_scope_final_locked.csv` — final 5-channel scope with exclusion reasons.
- `detector_calibration_v2_final_locked.csv` — per-channel recall/FA at locked hyperparameters.
- `channel_mcc_bootstrap.csv`, `channel_mcc_bootstrap_test_only.csv` — bootstrap CIs (full-data and test-only).
- `onset_bayesian_results.csv`, `onset_posterior_distributions.csv` — per-segment onset index/confidence and full run-length posterior, consumed directly by Layer 3 §Monte Carlo uncertainty propagation.
