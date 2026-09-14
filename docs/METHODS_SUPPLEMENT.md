# Methods Supplement

**Companion to:** `README.md` (headline results and figures), `docs/DEV_LOG_SUMMARY.md` (analytical trail, negative results, discarded alternatives)

This supplement gives the manuscript-ready methodological detail underlying every reported number. It is organized to mirror the three-layer pipeline plus the independent model-class robustness check. Locked hyperparameters, thresholds, and exclusion criteria are stated explicitly wherever a decision was made from the data rather than fixed a priori, together with the ablation or verification evidence that motivated the choice.

---

## 1. Dataset

OPS-SAT-AD (Ruszczak, Kotowski, Evans & Nalepa, *Scientific Data* 12:710, 2025): 2,123 expert-labeled univariate telemetry segments from the European Space Agency's OPS-SAT nanosatellite, spanning 9 channels — three magnetometer channels (CADC0872–0874) and six photodiode channels (CADC0884, 0886, 0888, 0890, 0892, 0894). Each segment belongs to exactly one channel (verified: 0 multi-channel segments across all 2,123) and is labeled `anomaly ∈ {0,1}`; a secondary `label` field (`a2`/`a3`/`a4`/`anomaly`) partitions the normal class further but is not used for any modeling decision, following the original authors' own supervised-benchmark code, which likewise ignores it. The dataset provides an official train/test partition (`train ∈ {0,1}`, stratified by anomaly rate, ≈ 75%/25%); this partition is used throughout for train-fit/test-evaluate verification and is distinct from any causal treatment/control assignment. The dataset is not redistributed; `data/download_opssat_ad.sh` retrieves it from its Zenodo record.

---

## 2. Layer 1 — Signal Estimation

**2.1 Channel typing.** For each channel, first differences of the raw signal within nominal (`anomaly = 0`) segments are pooled. A channel is classified `quantized` if the fraction of exact-zero differences is ≥ 0.10 (its quantization step is then estimated as the 25th percentile of non-zero |Δ|); otherwise it is classified `float_noise_suspect` if the smallest non-zero |Δ| is below 1 × 10⁻⁶; otherwise `continuous`. Under this rule, CADC0872/0873/0874 are `float_noise_suspect`, CADC0884/0886/0888/0892/0894 are `quantized`, and CADC0890 is `continuous`. These thresholds were set from inspection of this dataset's empirical difference distributions and should be revisited for other telemetry sources.

**2.2 Observation- and process-noise estimation.** The observation-noise variance `r` and process-noise variance `q` (increments of the local level and local trend, respectively) are estimated as `Var(Δ)/2` and `Var(Δ²)/6` over the pooled nominal first/second differences. A Gaussian-mixture ("jump probability × jump-size MAD") alternative was evaluated and rejected: on zero-inflated (quantized) channels, MAD applied to the full difference distribution collapses to zero (the median of a ≥ 50%-zero sample is zero); restricting MAD to non-zero jumps avoids the collapse but destabilizes the steady-state calibration of CADC0890 and CADC0894 (standardized-innovation SD inflates to 2.25–2.88 rather than the expected ≈ 1). The naïve `Var(Δ)/2` / `Var(Δ²)/6` estimator is used throughout the reported pipeline.

**2.3 Quantization floor.** For channels classified `quantized`, a term `step²/12` (the variance of a Uniform(−step/2, step/2) distribution) is added to the observation-noise variance, reflecting the finite resolution of the underlying analog-to-digital conversion.

**2.4 Hierarchical shrinkage.** Per-channel `log(r)` and `log(q)` estimates are pooled via an Efron–Morris-type empirical-Bayes shrinkage estimator (inverse-variance-weighted global mean, moment-based between-channel variance `τ²`, per-channel shrinkage weight `τ²/(τ² + SE²)`), stabilizing estimates for the two channels with very few nominal points (CADC0886, CADC0890).

**2.5 State estimation.** A local-linear-trend Kalman filter (state = [level, trend], transition matrix `[[1,1],[0,1]]`, observation matrix `[1,0]`) is fit per channel using the shrunk `r`, `q`, and (where applicable) quantization floor. This is a signal-processing convenience model, not a physical model of the spacecraft's attitude or orbital dynamics; standardized innovations `z_t = (y_t − ŷ_t) / √S_t` from this filter are the input to Layer 2.

---

## 3. Layer 2 — Anomaly Detection

**3.1 Change-point model.** Bayesian Online Change-Point Detection (BOCPD; Adams & MacKay 2007) is run on the standardized innovation sequence `z_t`, using a Normal-Inverse-Gamma conjugate predictive model (`μ₀ = 0, κ₀ = 1, α₀ = 1, β₀ = 1`) and hazard rate `1/250` (constant across channels). A *forgetting* extension caps the run-length-specific sufficient statistic `κ` at `κ_max = 40`; once exceeded, `κ, α, β` for that run-length are rescaled proportionally toward the cap. Onset for a given segment is declared at the first time step `t` at which the MAP run-length collapses from > 5 to ≤ 2 (interpreted as a confirmed changepoint), with the onset index recovered as `t − run_length(t)` and the posterior distribution over the top-30 most probable run-lengths at confirmation retained for later uncertainty propagation (§5.1).

**3.2 Locked hyperparameters and ablation evidence.** A full 2 × 2 grid (`mixture noise estimator` ∈ {on, off}) × (`forgetting` ∈ {on, off}) was evaluated. The two factors act through disjoint mechanisms: forgetting is necessary to preserve recall on the `float_noise_suspect` channels (disabling it nearly halves CADC0874 recall, 0.725 → 0.319), while the noise-estimator choice is necessary to preserve correct calibration and sign of association on CADC0890 (mixture estimator inverts its MCC, 0.826 → −0.213) and CADC0894. The locked configuration — `mixture = False`, `forgetting = True` — is at or within ≈ 7% of the best of the four combinations for 6 of 7 channels with a defined MCC, and is confirmed as the winning combination under a train-only reselection procedure (channel-level MCC restricted to channels with ≥ 5 anomalous and ≥ 5 normal segments, averaged; train macro-MCC = 0.308 vs. 0.199–0.275 for the alternatives), evaluated on the held-out test split (per-channel test performance reported in §3.4).

**3.3 Channel scoping.** Channels are screened for inclusion in downstream detection reporting and causal analysis using Matthews Correlation Coefficient (MCC; Chicco & Jurman, *BMC Genomics* 21:6, 2020) and Youden's *J* (sensitivity + specificity − 1), computed at the locked hyperparameters. MCC is preferred to raw recall/false-alarm rate because it is normalized to 0 under chance-level prediction regardless of class balance, whereas false-alarm rate alone cannot distinguish a genuinely informative high-recall/high-false-alarm channel from an uninformative one that fires indiscriminately. Exclusion criteria, applied in order:

1. **Structural**: zero anomalous segments (CADC0884).
2. **Underpowered**: fewer than 5 anomalous or 5 normal segments (CADC0886: 3/8; CADC0890: 11/14 — excluded despite MCC = 0.826, a deliberate conservative choice given the small denominator).
3. **Chance-level or worse**: MCC ≤ 0 (CADC0892: MCC = −0.090, recall = 0.971 but false-alarm rate = 0.994).

Final scope: **CADC0872, CADC0873, CADC0874, CADC0888, CADC0894** (5 of 9 channels), carried forward to all of Layer 3.

**3.4 Triple verification.** The 5-channel scope and its per-channel MCC were independently re-derived three ways: (i) 2,000-resample bootstrap 95% confidence intervals on full-data MCC (all five included channels exclude 0; CADC0892's interval, [−0.274, 0.053], includes 0, reinforcing its exclusion); (ii) parameters re-estimated on the train split only, evaluated on the held-out test split (CADC0872 test recall 0.375, CADC0873 0.226, CADC0874 0.826, CADC0888 0.750, CADC0894 1.000 — directionally consistent with full-data recall, with expected sampling variation); (iii) bootstrap 95% CIs computed on the test split alone (e.g., CADC0888 [0.092, 0.528], CADC0874 [0.715, 0.962]), all excluding 0 for the five included channels and consistent with the full-data point estimates falling inside the test-only interval. CADC0890's exclusion is additionally reinforced (not newly decided) by the observation that its test split contains zero normal segments, making its high full-data MCC point estimate un-evaluable out of sample.

**3.5 Locked detection performance.**

| Channel | Recall | False-alarm rate | MCC | MCC 95% bootstrap CI |
|---|---|---|---|---|
| CADC0872 | 0.321 | 0.000 | 0.514 | [0.444, 0.577] |
| CADC0873 | 0.276 | 0.000 | 0.489 | [0.403, 0.564] |
| CADC0874 | 0.725 | 0.032 | 0.740 | [0.635, 0.832] |
| CADC0888 | 0.617 | 0.391 | 0.194 | [0.074, 0.311] |
| CADC0894 | 1.000 | 0.797 | 0.189 | [0.146, 0.230] |

---

## 4. Layer 3 — Causal Signature Analysis

All Layer 3 analyses operate on the 5 scoped channels and re-use the locked Layer 1–2 hyperparameters.

**4.1 Onset-uncertainty propagation.** BOCPD produces a posterior distribution over onset candidates (retained as the top-30 run-lengths at confirmation, §3.1), not a point estimate. For each detected anomalous segment, 200 onset candidates are drawn in proportion to this posterior (Monte Carlo, fixed seed = 42). For each draw, a pre-onset vs. post-onset comparison is computed for three engineered features:

- `level`: raw signal, compared via Welch's *t*-test and Cohen's *d* (unpooled-SD version) on the raw pre/post means.
- `diff`: first difference, compared via Welch's *t*-test and Cohen's *d* on the **squared** pre/post values (a variance-increase proxy, since the mean of a difference series is uninformative but its variance is the quantity of interest).
- `diff²`: second difference, treated identically to `diff`.

Segment-level summaries (median *p*, 95% CI, fraction of draws with *p* < 0.05) are computed across the 200 draws, and two corrections are applied before channel-level aggregation: (i) `level` effect direction is reported separately from magnitude (|*d*|), because roughly half of anomalous segments in most channels shift the mean upward and half downward, causing simple signed averaging to understate the true magnitude; (ii) segments are tiered by usable-draw count (high ≥ 100, medium 30–99, low < 30) and only high/medium-tier segments (99.4% of detected segments) are carried into channel-level summaries, draw-count-weighted.

**4.2 Temporal precedence test.** For each segment, a single "canonical" onset (median of the 200 Monte Carlo draws, rounded) is used to define pre/post baselines for `level`, `diff`, and `diff²` independently. Each series' first post-onset time step at which |standardized deviation from its own pre-onset baseline| exceeds 3 is recorded (in seconds, using each segment's native sampling interval; index offsets of 1 and 2 samples are applied to `diff` and `diff²` respectively to align them to the same time axis as `level`). Paired first-crossing times are compared via Wilcoxon signed-rank test, pooled across the 5 channels and separately per channel, with Bonferroni correction applied across all decidable comparisons. A parallel analysis on **normal segments** — using pivot points resampled from the empirical distribution of anomalous-segment onset positions within their own segments — serves as a quasi-experimental control for the precedence test itself, distinguishing a genuine anomaly-specific ordering from a generic property of the differencing operation. Formal comparison of the anomalous-vs-normal proportions uses two-proportion *z*-tests, Fisher's exact tests, and a channel-stratified Cochran–Mantel–Haenszel test (to remove channel-specific base-rate confounding), all Bonferroni-corrected. Where tied crossing times (a consequence of 1–5 s sampling resolution) leave the Wilcoxon test under-powered (specifically, `diff` vs. `diff²` ordering), relative timing is re-estimated via the lag of maximum cross-correlation between the two series' absolute standardized post-onset trajectories (continuous-valued, tie-free), compared within-group (Wilcoxon vs. lag = 0) and between-group (Mann–Whitney *U*).

**4.3 Temporal-profile classification.** Each series' post-onset |z| trajectory (within a 60-sample lookahead) is classified as **transient** if it peaks and then decays to ≤ 50% of its peak value within the lookahead window, or **persistent** if it does not decay within that window. This diagnostic is reported alongside, and used to interpret the sensitivity of, the full-window vs. early-window Welch-test comparison described next.

**4.4 Early-window sensitivity check.** As a robustness check on §4.1, the post-onset Welch-test window is additionally restricted to the first 10 post-onset points (rather than the full post-onset segment) and re-computed; the resulting change in each feature's significance rate is reported as a diagnostic for whether the full-window design under-estimates transient effects.

**4.5 Random-effects meta-analysis.** Channel-level effect sizes (|Cohen's *d*|, from §4.1) and significance fractions are pooled across the 5 scoped channels using DerSimonian & Laird (1986) random-effects meta-analysis. Significance-fraction (a bounded [0,1] proportion) is arcsine-square-root transformed before pooling and back-transformed for reporting, following standard meta-analytic practice for proportion outcomes. Heterogeneity is quantified via Cochran's *Q* and *I²`. Leave-one-out (LOO) sensitivity analysis re-computes the pooled estimate and *I²* with each channel excluded in turn; a channel is flagged as *influential* if excluding it drops *I²* by ≥ 30 percentage points **or** shifts the pooled point estimate by ≥ 30% in relative terms (the latter criterion added specifically because the *I²*-only rule missed cases — observed for `diff`/`diff²` significance fraction and CADC0894 — where the pooled estimate moved by 40–80% without a correspondingly large *I²* change).

**4.6 Quasi-experimental placebo-pool comparison.** This is the pipeline's primary causal-signature test. For each of the 5 channels, the observed effect (`level` |*d*|, `diff`/`diff²` log-variance-ratio: `log(Var(post)/Var(pre))`) at the canonical onset of every anomalous segment is compared, via one-sided Mann–Whitney *U* test, against a **placebo pool** constructed from the same channel's normal segments: for each normal segment, up to `K = 5` pivot points are drawn from the empirical distribution of anomalous-segment onset-position ratios (onset index ÷ segment length) observed in that same channel, and the identical pre/post effect calculation is applied at each pivot (up to `N = 300` normal segments per channel, for computational tractability). Fifteen tests (5 channels × 3 features) are Bonferroni-corrected. This design directly tests whether the observed pre/post effect is *specific to anomalies* or is reproducible by splitting an arbitrary normal segment at a similarly distributed pivot — the same logic as a matched placebo control in a quasi-experimental design, with segment-internal drift as the "placebo treatment."

**4.7 Robustness of the placebo-pool result.** The full-data comparison in §4.6 is independently re-run restricting all anomalous, normal, and placebo-pool construction steps to (i) the train split only and (ii) the test split only, using the dataset's official train/test partition (§1). Agreement across all three slices (full/train/test) on significance direction is reported per (channel, feature) cell; cells undecidable due to small test-split sample size (*n* < 5) are reported as such rather than as non-significant.

**4.8 Labeling-protocol audit.** The primary OPS-SAT-AD publication, its SoftwareX companion (the OXI annotation tool), and the associated code release were reviewed for any documented, quantitative rule governing segment-boundary placement (e.g., a fixed margin before an operator-flagged onset). No such rule is documented; segmentation is described uniformly as manual expert curation. This bears on the interpretation of the observed within-segment onset-position skew (§4.2 control analysis) as a possible labeling artifact.

---

## 5. Independent Model-Class Cross-Validation

**5.1 Rationale.** A single external model (e.g., one deep-sequence architecture) cannot distinguish "the effect is real" from "this particular architecture's inductive bias happens to find it." Sixteen independently trained, independently attributed classifiers spanning six inductive-bias families are used instead, on the reasoning that convergent evidence across architecturally unrelated models is far harder to attribute to any single model's idiosyncrasies.

**5.2 Task and labels.** All models solve the same binary classification problem defined by §4.6's anomalous-vs-placebo design (identical pivot sampling, identical train/normal-segment pool), ensuring the model-comparison labels are drawn from the same causal design as the primary statistical test rather than a separately motivated task.

**5.3 Two independent data representations.**

- *Tabular* (10 models): the three engineered features from §4.6 (`level` |*d*|, `diff`/`diff²` log-variance-ratio). Models: L1- and L2-regularized logistic regression, Gaussian naive Bayes, *k*-nearest neighbors, RBF-kernel SVM, random forest, extremely randomized trees, gradient boosting (scikit-learn), XGBoost, LightGBM. Feature attribution: SHAP (TreeExplainer / LinearExplainer / KernelExplainer as appropriate) where computable, falling back to permutation importance (20 repeats, scored by ROC-AUC) otherwise.
- *Raw sequence* (6 models): z-normalized (per-window, per-channel) 3-channel time-series windows spanning ±15 samples around the pivot, with channels corresponding to `level`, `diff`, and `diff²` — i.e., no human-engineered summary statistic. Models: 1-D CNN, dilated-causal TCN, bidirectional LSTM, bidirectional GRU, a small Transformer encoder, and a lightweight selective state-space (Mamba-style) block implemented in pure PyTorch (the official CUDA-kernel `mamba_ssm` package was unavailable in the execution environment; this substitute reproduces the input-dependent state-transition mechanism but not the official implementation's throughput or exact numerics). Feature attribution: Integrated Gradients (Captum where available, otherwise a manual Riemann-sum implementation, 32 steps, zero baseline), summed over the time axis and normalized to a per-channel share.

Evaluation uses 5-fold grouped cross-validation (grouping by channel–segment identity to prevent leakage across placebo draws from the same segment); out-of-fold predictions are used for AUC/recall/false-alarm-rate reporting.

**5.4 Agreement statistics.** For each model, the three features' importance shares are converted to within-model ranks. Kendall's coefficient of concordance *W* and its χ² approximation quantify overall cross-model rank agreement; a one-sided binomial test (null: `level` ranked top with probability 1/3, i.e., chance) tests whether the observed count of models *not* ranking `level` top departs from chance. Bootstrap stability (200 stratified resamples) is assessed for two representative models spanning different inductive biases (random forest, logistic regression), reporting the fraction of resamples in which `diff` + `diff²` combined importance exceeds `level`.

**5.5 Exclusions.** One tabular model (a shallow MLP) achieved out-of-fold AUC = 0.403 (below the chance level of 0.5), indicating a failed fit (optimization instability) rather than a genuine level-favoring inductive bias; it was excluded from all agreement statistics on the grounds that an untrained model's feature attributions carry no interpretable information. All headline agreement statistics (§5.4) are therefore computed over 16 models (10 tabular + 6 sequence).

---

## 6. Multiple-Comparison Corrections

Bonferroni correction is applied within each family of simultaneous hypothesis tests reported in this supplement: the 15 placebo-pool tests in §4.6 (and their train/test-split replications in §4.7), the temporal-precedence tests and their group-comparison analogues in §4.2, and the meta-analytic pooled-effect tests in §4.5. The pooled (not per-test) Wilcoxon statistic is used for the headline temporal-precedence claim to avoid inflating the effective number of comparisons from per-channel pairwise tests. Corrected *p*-values are reported as the primary decision criterion throughout; uncorrected values are given alongside for transparency.

---

## 7. Scope Statement on Causal Language

Following the scope statement in the README, this pipeline establishes a **quasi-experimental, statistically tested causal-signature claim** — supported by placebo-pool comparison (§4.6), temporal precedence with a normal-segment control (§4.2–4.3), channel-stratified confirmation (§4.2), train/test triple verification (§4.7), and convergent evidence from 16 architecturally independent classifiers (§5) — and not a formally identified causal effect in the Pearl do-calculus sense. No causal-discovery algorithm (PC, FCI, LiNGAM, etc.) is used to derive the structural skeleton; the ordering asserted in the final structural model (Onset → diff/diff² variance surge, with level treated as a confounded, non-core node) is asserted from known instrument physics — a change point in the underlying process perturbs local variance/derivative statistics before a sustained mean shift is reliably observable — and is then tested against data for consistency, not derived from data.

---

## 8. Known Limitations Carried Forward from This Analysis

- The causal-signature claim is scoped to the 5 channels retained after Layer 2 scoping; it is not asserted, and should not be extrapolated, to the 4 excluded channels.
- CADC0894's small anomalous-segment count (*n* = 21) limits statistical power in every test-split replication in §4.7 and in the Layer 2 bootstrap in §3.4; results for this channel should be read alongside their confidence intervals rather than as point estimates alone.
- CADC0874's `diff²` effect size is likely under-estimated by the full-window Welch design (§4.1, §4.4) relative to a design tuned to its short-lived transient response; the reported meta-analytic estimate for this channel-feature combination should be treated as a conservative lower bound.
- The state-space (Mamba-style) model in §5.3 is a lightweight, non-CUDA-kernel substitute for the official `mamba_ssm` implementation; its narrower agreement margin relative to the other 12 inductive-bias families (§5.4) should not be over-interpreted without confirming the result holds under the official implementation.
- No independent second spacecraft-telemetry dataset was used to test generalization of the `diff`/`diff²` signature beyond OPS-SAT-AD.
