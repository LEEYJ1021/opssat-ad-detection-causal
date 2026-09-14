# Step 3b — Labeling Protocol Audit & Train/Test Triple Verification of Layer 3

**Pipeline stage:** Layer 3 supplement (`layer3_causal_analysis/labeling_protocol_audit.py`, `layer3_causal_analysis/supplementary_diagnostics_A_D.py`)
**Purpose:** Close the two outstanding threats-to-validity items flagged at the end of `step3_causal_analysis.md`: (i) whether the onset-position skew used to build the placebo pool is a physical pattern or a labeling artifact, and (ii) whether the §3.7 quasi-experimental placebo-pool result — the single most decisive test in the paper — survives the same train/test separation rigor that Layer 2 applied to detection performance.
**Input:** Locked Layer 1–3 outputs for the 5 scoped channels (CADC0872/0873/0874/0888/0894); `segments.csv`'s official `train` column.
**Status:** Locked. Both items closed. Consumed by `README.md` §"Threats to validity" and Fig. 3b.

---

## Motivation

Two items were left open at the close of `step3_causal_analysis.md`:

- **B-5 (labeling-protocol audit).** §3's control analysis found anomalous-segment onsets concentrated at ≈56.9% of segment length (skewed toward the back half). It is unresolved whether this reflects (1) a genuine physical tendency for anomalies to occur later in a monitored window, or (2) an artifact of how ESA annotators cut segment boundaries (e.g., a convention of leaving generous margin before a flagged onset). This is a documentation-research question, not a code task.
- **B-6 / B-7 (train/test re-verification of §3).** Layer 2's detection performance was rigorously separated into train-fit/test-evaluate passes (`step2_anomaly_detection.md` §4). Layer 3's entire causal-signature pipeline (Monte Carlo propagation, temporal precedence, meta-analysis, and the §3.7 placebo-pool test) was run on the **full dataset without a train/test split**. This leaves open the concern that the diff/diff² finding is specific to having pooled all the data, rather than a pattern that would already be visible from the training data alone. This requires an actual code re-run: the placebo-pool comparison, re-executed independently on train-only and test-only segments.

---

## Part A — B-5: Labeling protocol audit

**Method.** The primary OPS-SAT-AD publication (Ruszczak, Kotowski, Evans & Nalepa, *Scientific Data* 12:710, 2025), its SoftwareX companion describing the OXI annotation tool, and the associated arXiv/PMC full texts were reviewed for any documented, quantitative rule governing segment-boundary placement (e.g., a fixed margin of N seconds before an operator-flagged onset).

**Findings.**

- Segments were cut entirely **manually**. ESA spacecraft operations engineers first proposed candidate telemetry windows judged subjectively "interesting" for anomaly detection; domain experts then used OXI, a web-based visualization/annotation tool, to collaboratively extract and label the normal/anomalous segments.
- No document — main text, SoftwareX paper, or either preprint version — specifies a quantitative segmentation rule (e.g., "always include 30 s before onset"). The only description given anywhere is "reviewed and manually curated by domain experts."

**Interpretation.** This is effectively an **"undecidable" finding**, but its implication is informative rather than null: because boundary placement was subjective, case-by-case human judgment rather than the output of a fixed rule, the mean onset-position skew (0.569) is more plausibly the aggregate of idiosyncratic individual choices than the systematic product of a single quantitative convention. Had a fixed rule existed, that rule itself could have imprinted a structure onto the data that later masqueraded as the diff/diff² precedence signature; a purely subjective process makes that specific confound less likely (though it cannot be fully excluded).

**Recommended manuscript language:** *"Segment boundaries were determined by manual ESA-expert annotation using the OXI tool; no quantitative cutting rule is documented in the public materials. The possibility that onset-position skew reflects a systematic labeling-protocol artifact cannot be fully excluded, but the absence of a fixed, rule-based cutting convention limits the plausibility of a systematic bias of this kind."*

---

## Part B — B-6 / B-7: Train-only and test-only reproduction of the §3.7 placebo-pool test

### Method

The exact §3.7 procedure (`quasi_experimental_placebo.py`) — canonical-onset effect computation (`level` |Cohen's *d*|; `diff`/`diff²` log-variance-ratio `log(Var(post)/Var(pre))`), placebo-pool construction (up to 5 pivots per normal segment, drawn from the channel's empirical onset-position-ratio distribution, up to 300 normal segments/channel), and one-sided Mann–Whitney *U* tests (Bonferroni-corrected across 5 channels × 3 features = 15 tests) — was re-run twice, independently, using only `train == 1` segments (B-6) and only `train == 0` segments (B-7), using the dataset's official train/test partition. All helper functions (`cohens_d_abs`, `log_var_ratio`, `compute_effects`) are byte-identical to the full-data §3.7 implementation.

### B-6 — Train-only results

Filtering to `train == 1` retains 178,504 of 240,979 raw observation rows (74.1%) and 126 of 177 reliable Monte Carlo-tiered anomalous segments (channel breakdown: 872=29, 873=22, 874=31, 888=28, 894=16).

| Channel | Feature | n_anomaly | n_placebo | median (anomaly) | median (placebo) | p (Bonferroni) |
|---|---|---|---|---|---|---|
| CADC0872 | level | 29 | 1500 | 1.444 | 2.004 | 1.000 |
| CADC0872 | diff | 29 | 1500 | 2.997 | −0.723 | 1.22 × 10⁻¹⁶ |
| CADC0872 | diff² | 29 | 1500 | 3.012 | 0.069 | 3.73 × 10⁻¹⁷ |
| CADC0873 | level | 22 | 1490 | 0.670 | 2.102 | 1.000 |
| CADC0873 | diff | 21 | 1490 | 3.499 | −1.125 | 1.12 × 10⁻¹² |
| CADC0873 | diff² | 21 | 1490 | 3.590 | 0.040 | 7.22 × 10⁻¹² |
| CADC0874 | level | 31 | 480 | 0.889 | 1.127 | 1.000 |
| CADC0874 | diff | 31 | 480 | 2.585 | 0.148 | 1.17 × 10⁻⁹ |
| CADC0874 | diff² | 31 | 480 | 2.471 | 0.021 | 9.32 × 10⁻¹⁴ |
| CADC0888 | level | 28 | 550 | 1.099 | 1.212 | 1.000 |
| CADC0888 | diff | 26 | 456 | 2.565 | −0.082 | 4.49 × 10⁻⁴ |
| CADC0888 | diff² | 26 | 464 | 3.840 | −0.378 | 1.55 × 10⁻⁹ |
| CADC0894 | level | 16 | 475 | 0.999 | 0.955 | 1.000 |
| CADC0894 | diff | 12 | 440 | 2.581 | 0.302 | 1.45 × 10⁻⁴ |
| CADC0894 | diff² | 12 | 442 | 2.490 | 0.348 | 1.22 × 10⁻³ |

**Result: 15/15 (channel × feature) significance calls match the full-data §3.7 result exactly.** `level` remains uniformly non-significant (p_bonf = 1.000, all 5 channels); `diff`/`diff²` remain significant in all 5 channels. p-values are uniformly larger under train-only (smaller sample), but no call flips.

### B-7 — Test-only results

Filtering to `train == 0` retains 62,475 rows (25.9%) and only 51 reliable anomalous segments (872=12, 873=7, 874=19, 888=8, 894=5) — the thin test slice already flagged at Layer 2 scoping.

| Channel | Feature | n_anomaly | n_placebo | median (anomaly) | median (placebo) | p (Bonferroni) |
|---|---|---|---|---|---|---|
| CADC0872 | level | 12 | 495 | 1.620 | 2.232 | 1.000 |
| CADC0872 | diff | 12 | 495 | 2.890 | −1.074 | 9.58 × 10⁻⁸ |
| CADC0872 | diff² | 12 | 495 | 2.610 | 0.137 | 1.23 × 10⁻⁶ |
| CADC0873 | level | 7 | 610 | 0.870 | 2.257 | 1.000 |
| CADC0873 | diff | 7 | 610 | 4.651 | −0.957 | 3.46 × 10⁻⁵ |
| CADC0873 | diff² | 7 | 610 | 4.638 | 0.030 | 3.46 × 10⁻⁵ |
| CADC0874 | level | 19 | 145 | 1.130 | 1.020 | 1.000 |
| CADC0874 | diff | 19 | 145 | 1.960 | −0.404 | 1.05 × 10⁻⁸ |
| CADC0874 | diff² | 19 | 145 | 1.808 | −0.064 | 2.62 × 10⁻⁹ |
| CADC0888 | level | 8 | 210 | 0.657 | 0.991 | 1.000 |
| CADC0888 | diff | 7 | 194 | 2.245 | 0.823 | **0.051** |
| CADC0888 | diff² | 7 | 194 | 2.938 | −0.421 | 1.51 × 10⁻³ |
| CADC0894 | level | 5 | 140 | 0.672 | 0.778 | 1.000 |
| CADC0894 | diff | 4 | 129 | — | — | n/a (n=4 < 5) |
| CADC0894 | diff² | 4 | 129 | — | — | n/a (n=4 < 5) |

### Three-way comparison (full / train / test)

| Result category | n (of 15) | Detail |
|---|---|---|
| Fully agree across all 3 slices | 12 | `level` non-significant in all 5 channels; `diff`/`diff²` significant in 872, 873, 874 (both features), and 888 diff² |
| Disagree | 1 | CADC0888 `diff`: significant in full (6.9×10⁻⁶) and train-only (4.5×10⁻⁴), crosses the Bonferroni threshold in test-only (p = 0.051) |
| Undecidable (test n < 5) | 2 | CADC0894 `diff`/`diff²` — test slice yields only 4 usable anomalous segments |

**Reading the single disagreement.** The CADC0888 `diff` p-value rises **monotonically** with shrinking sample size (6.9×10⁻⁶ → 4.5×10⁻⁴ → 0.051) rather than reversing sign or direction, and the same channel's `diff²` in the identical test slice remains strongly significant (p = 1.5×10⁻³). This is read as a power artifact of the smallest slice, not an effect reversal — consistent with the CADC0888/diff pattern already seen in the original §3.8 triple-verification table in `step3_causal_analysis.md`.

**Reading the two undecidable cells.** CADC0894 is already documented (Layer 2 scoping, `step2_anomaly_detection.md` §3–4) as the thinnest-margin included channel; the same limitation resurfaces here as a sample-size floor on the test slice, not a new finding.

### Summary statement

> Layer 3's headline result — that `level` is a non-specific channel-drift confounder while `diff`/`diff²` are the anomaly-specific signature — reproduces independently when the entire §3.7 placebo-pool procedure is re-run using only training-split segments and, separately, only test-split segments. 12 of 13 decidable (channel × feature) comparisons agree exactly across all three data slices; the sole disagreement (CADC0888 `diff`) is explained by a monotone loss of statistical power in the smallest slice, not an effect reversal; the 2 undecidable cells (CADC0894 `diff`/`diff²` on the test slice) reflect a pre-existing, already-documented sample-size limitation of that channel, not a new one. This closes the same train/test-leakage concern for the causal-signature analysis that `step2_anomaly_detection.md` §4 closed for detection performance.

---

## Outputs consumed downstream

- `docs/dev-log/step3b_labeling_protocol_revalidation.md` — this document (labeling audit + train/test triple verification).
- `stage3_qexp_mannwhitney_train_only.csv` — B-6 train-only Mann–Whitney results (15 rows).
- `stage3_qexp_mannwhitney_test_only.csv` — B-7 test-only Mann–Whitney results (15 rows, 2 marked "n<5").
- `stage3_train_vs_full_comparison.csv` — train-vs-full 2-way agreement table.
- `stage3_triple_verification_summary.csv` — full/train/test 3-way agreement table (`sig_full`, `sig_train`, `sig_test`, `all_three_agree`) — **this is the table underlying Fig. 3b** in `README.md`.
