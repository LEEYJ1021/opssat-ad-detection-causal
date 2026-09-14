# Step 3 — Causal Signature Analysis

**Pipeline stage:** Layer 3 (`layer3_causal_analysis/`)
**Purpose:** Determine what actually changes at anomaly onset, and rule out that the answer is an artifact of labeling, channel choice, or data-split leakage.
**Input:** Locked Layer 1–2 outputs for the 5 scoped channels (CADC0872/0873/0874/0888/0894).
**Status:** Locked. See `step3b_labeling_protocol_revalidation.md` for the labeling audit and `step_ai_model_cross_validation.md` for the independent model-class robustness check.

---

## 3.1 Onset-uncertainty propagation (Monte Carlo)

BOCPD's onset is a **posterior** over run-lengths, not a point estimate. For each detected anomalous segment across the 5 scoped channels, 200 onset candidates are drawn in proportion to this posterior (fixed seed = 42). For each draw, pre- vs. post-onset comparisons are computed for:

- `level` — raw signal mean, Welch *t*-test + Cohen's *d* (unpooled SD).
- `diff` — first difference, Welch *t*-test + Cohen's *d* on **squared** values (variance-increase proxy).
- `diff²` — second difference, same treatment as `diff`.

**Coverage.** Of 386 anomalous segments in the 5 scoped channels, 178 (46.1%) yield a detected onset and a scoreable pre/post window (per-channel: 872 n=42, 873 n=29, 874 n=50, 888 n=36/37, 894 n=21). The remaining 53.9% are excluded — mostly "onset not detected" (203/208), consistent with the channels' known recall (`step2_anomaly_detection.md` §5).

**Two corrections applied before channel-level aggregation:**

1. **Sign cancellation.** Roughly half of anomalous segments shift `level` upward and half downward (plausible mixture of, e.g., "stuck-high" vs. "stuck-low" sensor behavior); naïve signed averaging washes out the magnitude. |*d*| is reported separately from the segment-level sign distribution.
2. **Draw-count reliability.** Segments are tiered by usable-draw count: high (≥ 100), medium (30–99), low (< 30). 176/178 (99.4%) detected segments fall in the high tier (only 1 low, 1 medium); channel-level summaries use draw-count-weighted high+medium segments only.

**Channel-level summary (weighted, |*d*| for level; log-variance-ratio for diff/diff²):**

| Channel | n reliable segs | level \|d\| | level frac_sig | diff \|d\| | diff frac_sig | diff² \|d\| | diff² frac_sig |
|---|---|---|---|---|---|---|---|
| CADC0872 | 41 | 1.659 | 0.86 | 0.273 (signed, uncorrected) / 0.251 (\|d\|) | 0.024 | 0.322–0.350 | 0.086 |
| CADC0873 | 29 | 0.982 | 0.86 | 0.266 | 0.034 | 0.342 | 0.172 |
| CADC0874 | 50 | 1.161 | 0.92 | 0.151 | 0.120 | 0.190 | 0.280 |
| CADC0888 | 36 | 0.897–0.924 | 0.54 | 0.428 | 0.203 | 0.334–0.360 | 0.030 |
| CADC0894 | 21 | 1.086–1.093 | 0.90 | 0.298–0.300 | 0.762 | 0.362–0.374 | 0.714 |

**Level direction (sign) distribution, corrected:** roughly balanced for four of five channels (CADC0872: 39% positive/61% negative; CADC0873: 31%/69%; CADC0874: 50%/50%; CADC0888: 44%/56%) — i.e., **effect size is large but direction is not consistent**, confirming that a signed pooled estimate for `level` would be misleading. CADC0894 is the exception (76% positive/24% negative).

**Direction consistency for `diff`/`diff²` (within-segment, across the 200 draws):** overwhelmingly one-sided — `diff` is positive (variance increase) in 88–100% of segments across all five channels; `diff²` similarly 89–97%. This directional consistency, combined with the large but sign-inconsistent `level` effect, foreshadows the pattern later confirmed decisively in §3.6.

## 3.2 Temporal precedence test

**Rationale.** If `level` shift were the primary driver and `diff`/`diff²` were downstream consequences, `level` should cross a fixed threshold (|standardized deviation from pre-onset baseline| > 3) *first*. Each segment's "canonical" onset (median of its 200 Monte Carlo draws) defines the pre/post split for each of the three series independently; the first post-onset time step exceeding the threshold, in seconds, is recorded (`diff`/`diff²` index offsets of 1/2 samples applied to align time axes).

**Coverage:** of 177 reliable segments, a threshold crossing is found for `level` in 70 (40%), `diff` in 162 (92%), `diff²` in 166 (94%).

**Pooled Wilcoxon signed-rank test (5-channel pooled, paired first-crossing times):**

| Comparison | n pairs | median Δt (s) | *p* (Wilcoxon) | Fraction where first term precedes second |
|---|---|---|---|---|
| level vs. diff | 70 | 0.0 | 1.4 × 10⁻⁵ | diff precedes level in 84.3% (level precedes in 15.7%) |
| level vs. diff² | 70 | 0.0 | 3.2 × 10⁻⁵ | diff² precedes level in 82.9% (level precedes in 17.1%) |
| diff vs. diff² | 161 | 0.0 | 0.368 (n.s.) | no reliable ordering (later resolved as a **sampling-resolution tie problem**, §3.4) |

**This directly falsifies the working hypothesis `level → diff → diff²`** — the opposite ordering was observed.

## 3.3 Reconciling §3.1 and §3.2 (why "level is more significant" and "diff is faster" are not contradictory)

§3.1 found `level` significant (within-segment pre/post) in 86–92% of segments for three channels, while `diff` was significant in only 2–12% — apparently at odds with §3.2's finding that `diff` crosses threshold *first*. Three diagnostics resolve this:

**(a) Temporal-profile classification** (post-onset |z| trajectory, 60-sample lookahead; **transient** = decays to ≤ 50% of peak within window, **persistent** = does not):

| Variable | Transient | Persistent | Undetermined |
|---|---|---|---|
| level | 88 (50%) | 80 (45%) | 9 |
| diff | 161 (98%) | 1 | 15 |
| diff² | 158 (94%) | 2 | 17 |

Per-channel `diff²` transient fraction: CADC0872 0.976, CADC0873 0.966, CADC0874 0.980, CADC0888 0.750, CADC0894 0.667. `diff`/`diff²` are overwhelmingly short-lived spikes; `level` is a near-even mix of transient and sustained change.

**(b) Early-window sensitivity check.** Restricting the post-onset Welch-test window to the first 10 points (vs. the full post-onset segment):

| Variable | frac_sig (early, 10 pt) | frac_sig (full window) | gain (early − full) |
|---|---|---|---|
| level | 0.633 | 0.864 | −0.232 |
| diff | 0.090 | 0.175 | −0.085 |
| diff² | 0.130 | 0.220 | −0.090 |

The full-window design dilutes, rather than misses, `diff`/`diff²`'s effect: neither variable's early-window rate exceeds its full-window rate, but the persistently-elevated `level` variable's rate falls by far more when the window shrinks — consistent with `level` being detectable from a smaller sample of a sustained shift, while `diff`/`diff²`'s brief spikes require averaging over more of the post-window to accumulate significance under a mean-based test not designed for transient signals.

**(c) Normal-segment control (decisive).** The §3.2 precedence procedure was repeated on **normal segments**, using pivot points resampled from the empirical distribution of anomalous-segment onset-position ratios (mean ratio ≈ 0.569; §`step3b`). Result — the **opposite** ordering:

| Comparison | Normal segments (n pairs) | frac(first precedes second) | Anomalous segments (n pairs) | frac(first precedes second) |
|---|---|---|---|---|
| level vs. diff | 224 | level 55.8% / diff 44.2% | 70 | level 15.7% / **diff 84.3%** |
| level vs. diff² | 171 | level **74.3%** / diff² 25.7% | 70 | level 17.1% / **diff² 82.9%** |
| diff vs. diff² | 211 | diff **64.5%** / diff² 35.5% | 161 | diff 7.5% / **diff² 92.5%** (p=0.368 in anomalous; see §3.4) |

The precedence ordering **reverses direction between normal and anomalous segments for all three pairs**. This is the decisive evidence against the "diff precedes level merely because differencing is inherently more sensitive" artifact hypothesis: the same differencing operation applied to normal data gives the opposite ordering.

## 3.4 Formal group comparison and diff-vs-diff² tie resolution

**Two-proportion z-test / Fisher's exact test (pooled, anomalous vs. normal):**

| Comparison | Anomalous frac (n) | Normal frac (n) | Odds ratio | *p* (Fisher) | *p*₍Bonf, 3 tests₎ |
|---|---|---|---|---|---|
| level precedes diff | 0.157 (70) | 0.558 (224) | 0.148 | 1.56 × 10⁻⁹ | 4.68 × 10⁻⁹ |
| level precedes diff² | 0.171 (70) | 0.743 (171) | 0.072 | 2.83 × 10⁻¹⁶ | 8.48 × 10⁻¹⁶ |
| diff precedes diff² | 0.075 (161) | 0.645 (211) | 0.044 | 1.06 × 10⁻³¹ | 3.18 × 10⁻³¹ |

**Channel-stratified Cochran–Mantel–Haenszel test** (controls for channel-specific base rates):

| Comparison | n strata | χ² | *p* | OR (Mantel–Haenszel) | *p*₍Bonf, 3 tests₎ |
|---|---|---|---|---|---|
| level precedes diff | 5 | 44.03 | 3.24 × 10⁻¹¹ | 0.095 | 9.72 × 10⁻¹¹ |
| level precedes diff² | 5 | 63.70 | 1.44 × 10⁻¹⁵ | 0.067 | 4.33 × 10⁻¹⁵ |
| diff precedes diff² | 5 | 114.57 | ≈ 0 | 0.058 | ≈ 0 |

All three orderings remain highly significant after stratifying by channel, ruling out channel-specific base-rate confounding as the explanation.

**diff-vs-diff² tie resolution.** The §3.2 non-significant diff-vs-diff² result (p = 0.368, `median_diff_sec = 0.0`) reflects extensive tied first-crossing times at 1–5 s sampling resolution. Re-estimating relative timing via the **lag of maximum cross-correlation** between the two series' post-onset |z| trajectories (continuous-valued, tie-free, ±15-sample search):

| Test | n | median lag | frac diff leads (lag>0) | frac diff² leads (lag<0) | *p* |
|---|---|---|---|---|---|
| Anomalous: lag vs. 0 (Wilcoxon) | 168 | 0.0 | 0.405 | 0.101 | 0.00094 (0.0028 Bonf.) |
| Normal: lag vs. 0 (Wilcoxon) | 702 | 0.0 | 0.454 | 0.312 | 0.033 (0.098 Bonf.) |
| Anomalous vs. Normal (Mann–Whitney U) | 168 vs. 702 | 0.00 vs. 0.00 | — | — | 0.906 |

**Conclusion:** `diff` and `diff²` genuinely have no reliable temporal ordering relative to each other, in either regime — the group-comparison test confirms the lag *distributions* themselves do not differ between anomalous and normal segments (p = 0.91), unlike every level-vs-{diff,diff²} comparison above.

## 3.5 Random-effects meta-analysis (DerSimonian–Laird, *k* = 5 channels)

Channel-level |Cohen's *d*| and significance-fraction (arcsine-square-root transformed, back-transformed for reporting) are pooled with a random-effects model.

**Effect size (|d|) pooled results:**

| Feature | *k* | pooled \|d\| | 95% CI | *p*₍Bonf, 6 tests₎ | *I²* |
|---|---|---|---|---|---|
| level | 5 | 1.128 | [0.925, 1.331] | ≈ 0 | 65.7% (high) |
| diff | 5 | 0.283 | [0.158, 0.408] | ≈ 0 | 97.2% (very high) |
| diff² | 5 | 0.317 | [0.220, 0.415] | ≈ 0 | 93.8% (very high) |

**Significance-fraction pooled results:**

| Feature | *k* | pooled proportion | 95% CI | *p*₍Bonf₎ | *I²* |
|---|---|---|---|---|---|
| level | 5 | 0.928 | [0.805, 0.993] | ≈ 0 | 82.8% (very high) |
| diff | 5 | 0.100 | [0.007, 0.278] | 0.022 | 93.8% (very high) |
| diff² | 5 | 0.133 | [0.013, 0.348] | 0.014 | 92.2% (very high) |

*I²* ≥ 65% throughout argues against reporting a single pooled "channel-agnostic" effect size without qualification; heterogeneity is decomposed by leave-one-out analysis next.

## 3.6 Leave-one-out (LOO) decomposition of heterogeneity

Standard *I²*-drop (≥ 30 pp) criterion alone misses cases where the pooled point estimate shifts substantially without a correspondingly large *I²* change; a **relative-shift criterion** (≥ 30% shift in pooled estimate) was added and flags additional cases.

**Key LOO findings (channel removed → resulting change):**

| Outcome | Influential channel | Effect of removal |
|---|---|---|
| level \|d\| | **CADC0872** | *I²* 65.7% → 0.0% (removal eliminates heterogeneity entirely) |
| level frac_sig | **CADC0888** | *I²* 82.8% → 0.0% |
| diff² \|d\| | **CADC0874** | *I²* 93.8% → 0.0%; correlates with CADC0874 having the *lowest* diff² \|d\| (0.190) and *highest* diff² transient_ratio (0.980) of the five channels — Spearman *r* = −1.00 (*n* = 5, exploratory) between transient_ratio and \|d\|, consistent with §3.3(b)'s dilution mechanism most severely affecting the most transient channel |
| diff/diff² frac_sig | **CADC0894** (relative-shift only; standard *I²*-drop misses this) | removing CADC0894 nearly halves pooled frac_sig (diff: 0.0996 → 0.0168, −64%; diff²: 0.1325 → 0.0415, −69%), but on top of an already-present, orderly hierarchy among the other four channels (diff frac_sig: 872=0.001 < 873=0.003 < 874=0.035 < 888=0.104) — CADC0894 is an extreme point on a continuum, not a categorically separate population |

**Note on CADC0872's diagnosed level |d| outlier (channel-level, not segment-level analysis):** two artifact hypotheses were tested and rejected:

- *Onset-sampling dispersion* (wide BOCPD posterior → noisy pre/post split → inflated *d*): CADC0872's onset-sampling SD is unremarkable (rank 3/5 among the five channels) and the cross-channel correlation with |*d*| runs the *wrong* direction (Spearman *r* = −0.5, *n* = 5).
- *Denominator shrinkage* (artificially small pre-onset SD inflating *d*): restricted to the three comparable `float_noise_suspect` channels, pre-onset SDs are essentially identical (1.4–1.6 × 10⁻⁵) while CADC0872's absolute mean shift is ≈ 60% larger than its siblings — the numerator, not the denominator, differs.
- *Extreme-segment contribution*: removing the top 5 most extreme segments (of 41) drops CADC0872's mean level |d| from 1.659 to 1.287 (−22%) — extreme segments contribute materially but the channel remains elevated relative to the other four (0.92–1.16) even after their removal.
- *Visual confirmation*: the 8 most extreme segments show genuine, sustained step-like level transitions coincident with a confidently localized BOCPD onset (typically 180–200/200 Monte Carlo draws concentrated at a single candidate), with a caveat that several pre-onset windows show a mild trend or brief plateau rather than a flat baseline — likely inflating, but not manufacturing, the effect size.

## 3.7 Quasi-experimental placebo-pool comparison (decisive test)

**Design.** All prior comparisons (§3.1–3.6) are *within-segment* (pre vs. post of the same anomalous segment) and cannot distinguish "the anomaly changed this" from "this channel drifts like this anyway." Each anomalous segment's observed effect at its canonical onset is compared, via one-sided Mann–Whitney *U*, against a **placebo pool**: for each of up to 300 normal segments per channel, up to 5 pivot points are drawn from the empirical distribution of anomalous-segment onset-position ratios in that same channel, and the identical pre/post effect (`level` |*d*|; `diff`/`diff²` log-variance-ratio `log(Var(post)/Var(pre))`) is computed at each pivot.

**Result (Mann–Whitney U, one-sided "greater", Bonferroni across 5 channels × 3 features = 15 tests):**

| Channel | Feature | n_anomaly | n_placebo | median (anomaly) | median (placebo) | *p*₍Bonf₎ |
|---|---|---|---|---|---|---|
| CADC0872 | level | 41 | 1495 | 1.451 | 2.020 | 1.000 |
| CADC0872 | diff | 41 | 1495 | 2.928 | −0.801 | 7.88 × 10⁻²⁴ |
| CADC0872 | diff² | 41 | 1495 | 2.962 | 0.097 | 1.11 × 10⁻²² |
| CADC0873 | level | 29 | 1485 | 0.769 | 2.194 | 1.000 |
| CADC0873 | diff | 28 | 1485 | 4.329 | −1.162 | 2.41 × 10⁻¹⁷ |
| CADC0873 | diff² | 28 | 1485 | 4.324 | 0.040 | 3.06 × 10⁻¹⁶ |
| CADC0874 | level | 50 | 625 | 1.036 | 1.067 | 1.000 |
| CADC0874 | diff | 50 | 625 | 2.398 | −0.011 | 5.40 × 10⁻¹⁶ |
| CADC0874 | diff² | 50 | 625 | 2.421 | −0.003 | 2.89 × 10⁻²² |
| CADC0888 | level | 36 | 760 | 0.985 | 1.164 | 1.000 |
| CADC0888 | diff | 33 | 654 | 2.386 | 0.193 | 6.93 × 10⁻⁶ |
| CADC0888 | diff² | 33 | 659 | 3.383 | −0.435 | 3.26 × 10⁻¹³ |
| CADC0894 | level | 21 | 615 | 0.889 | 0.914 | 1.000 |
| CADC0894 | diff | 16 | 569 | 2.471 | 0.269 | 3.90 × 10⁻⁶ |
| CADC0894 | diff² | 16 | 571 | 2.197 | 0.244 | 1.94 × 10⁻⁴ |

**level is statistically indistinguishable from ordinary channel drift in all 5 channels (p_Bonf = 1.000 uniformly).** `diff` and `diff²` are significant in all 5 channels after Bonferroni correction. Channel-level significant-segment fractions (per §"channel_summary" output) show `diff`/`diff²` strength is markedly higher in the three `float_noise_suspect` magnetometer channels (0.78–0.89) than in the two quantized photodiode channels (0.12–0.42), consistent with the meta-analytic heterogeneity in §3.5–3.6.

**Interpretive consequence.** This single result reframes the causal narrative established through §3.1–3.6: the `level` "signal" documented as highly significant within-segment (§3.1: frac_sig 0.86–0.92) is a real but **anomaly-non-specific** channel-drift pattern — a confounder — while the `diff`/`diff²` variance surge is reproduced nowhere in the placebo pool and is the genuine anomaly-specific signature. `level` should not be treated as a causal-chain endpoint; it should be treated as a confounded, non-core node.

## 3.8 Robustness of the placebo-pool result (train/test triple verification)

The full-data comparison in §3.7 was independently re-run on (i) the train split only and (ii) the test split only, using the dataset's official train/test partition.

| Channel | Feature | sig (full) | sig (train) | sig (test) | *p*₍Bonf₎ full / train / test | 3-way agree? |
|---|---|---|---|---|---|---|
| CADC0872 | level | No | No | No | 1.000 / 1.000 / 1.000 | Yes |
| CADC0872 | diff | Yes | Yes | Yes | 7.9e-24 / 1.2e-16 / 9.6e-8 | Yes |
| CADC0872 | diff² | Yes | Yes | Yes | 1.1e-22 / 3.7e-17 / 1.2e-6 | Yes |
| CADC0873 | level | No | No | No | 1.000 / 1.000 / 1.000 | Yes |
| CADC0873 | diff | Yes | Yes | Yes | 2.4e-17 / 1.1e-12 / 3.5e-5 | Yes |
| CADC0873 | diff² | Yes | Yes | Yes | 3.1e-16 / 7.2e-12 / 3.5e-5 | Yes |
| CADC0874 | level | No | No | No | 1.000 / 1.000 / 1.000 | Yes |
| CADC0874 | diff | Yes | Yes | Yes | 5.4e-16 / 1.2e-9 / 1.1e-8 | Yes |
| CADC0874 | diff² | Yes | Yes | Yes | 2.9e-22 / 9.3e-14 / 2.6e-9 | Yes |
| CADC0888 | level | No | No | No | 1.000 / 1.000 / 1.000 | Yes |
| CADC0888 | diff | Yes | Yes | **No** | 6.9e-6 / 4.5e-4 / **0.051** | **No** (monotone power loss, see below) |
| CADC0888 | diff² | Yes | Yes | Yes | 3.3e-13 / 1.6e-9 / 1.5e-3 | Yes |
| CADC0894 | level | No | No | No | 1.000 / 1.000 / 1.000 | Yes |
| CADC0894 | diff | Yes | Yes | n/a (n=4<5) | 3.9e-6 / 1.4e-4 / — | n/a |
| CADC0894 | diff² | Yes | Yes | n/a (n=4<5) | 1.9e-4 / 1.2e-3 / — | n/a |

**Summary:** 12 of 13 decidable (channel × feature) comparisons agree exactly across all three data slices; 2 cells (CADC0894 diff/diff²) are undecidable on the test split (n = 4 anomalous segments, below the n ≥ 5 threshold) — a known limitation of this channel already flagged at Layer 2 scoping, not a new finding. The single disagreement (CADC0888, diff) shows *p* rising **monotonically** with shrinking sample (6.9×10⁻⁶ → 4.5×10⁻⁴ → 0.051, crossing the Bonferroni threshold only in the smallest slice), while the same channel's diff² remains significant in the identical test slice (p = 1.5×10⁻³) — read as a power artifact of the smallest slice, not an effect reversal.

## 3.9 Summary of the confirmed causal skeleton (Path A)

```
Onset (BOCPD confirmed changepoint)
      │
      ▼
Diff/Diff² variance surge (transient, onset-localized; channel-moderated strength:
                            float_noise_suspect(872/873/874) strong > quantized(888/894) weaker)
      │
      ▼  (weak / confounded — not a causal-chain endpoint)
Level shift (persistent in ~50% of segments, but statistically indistinguishable
             from ordinary channel drift; excluded from the causal core)
```

This skeleton (§"Path A" in the README) is asserted from known instrument physics — a changepoint in the underlying process first perturbs local variance/derivative statistics before a sustained mean shift is reliably observable — and is *tested against data for consistency* (§3.2–3.8), not derived from data via a causal-discovery algorithm. See `README.md §Relationship to formal causal inference` for the full scope statement.

## 3.10 Outputs consumed downstream

- `stage2_mc_draws_raw.csv`, `stage2_mc_segment_summary_corrected.csv` — §3.1 Monte Carlo draws and corrected segment summaries (tiered by reliability).
- `stage2_precedence_crossing_times.csv`, `stage2_precedence_test_pooled.csv`, `stage2_precedence_test_by_channel.csv` — §3.2 precedence test.
- `stage2_diagA_diff_temporal_profile.csv`, `stage2_diagB_early_window_mc.csv`, `stage2_diagC_normal_control_crossing_times.csv`, `stage2_diagD_*` — §3.3–3.4 supplementary diagnostics.
- `stage2_meta_effect_size_*.csv`, `stage2_meta_frac_significant_*.csv`, `stage2_meta_loo_*.csv` — §3.5–3.6 meta-analysis and LOO.
- `stage3_qexp_placebo_pool.csv`, `stage3_qexp_mannwhitney*.csv`, `stage3_triple_verification_summary.csv` — §3.7–3.8 quasi-experimental test and its train/test replication.
