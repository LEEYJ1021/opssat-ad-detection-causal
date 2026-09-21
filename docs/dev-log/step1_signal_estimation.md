# Step 1 — Signal Estimation

**Pipeline stage:** Layer 1 (`layer1_signal_estimation/`)
**Purpose:** Produce a denoised, calibrated per-channel state estimate before any change-point logic is applied, so that downstream detection is not confounded by channel-specific noise characteristics.
**Status:** Locked. Consumed by Layer 2 (`docs/dev-log/step2_anomaly_detection.md`).

---

## 1. Channel typing

Each of the 9 OPS-SAT-AD channels is automatically classified from the empirical distribution of first differences (Δ) computed within nominal (`anomaly = 0`) segments only:

- **`quantized`** if the fraction of exact-zero Δ is ≥ 0.10 (quantization step estimated as the 25th percentile of non-zero |Δ|).
- **`float_noise_suspect`** if not quantized and the smallest non-zero |Δ| is < 1 × 10⁻⁶.
- **`continuous`** otherwise.

**Result:**

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

Thresholds (0.10, 1e-6) were set from inspection of this dataset's empirical diagnostics, not derived from first principles; they should be re-examined if this pipeline is applied to a different telemetry source.

## 2. Noise estimation

Observation-noise variance `r` and process-noise variance `q` (increments of local level and local trend) are estimated from pooled nominal-segment differences as:

```
r_robust  = Var(Δ)  / 2
q_robust  = Var(Δ²) / 6
```

**Discarded alternative — Gaussian-mixture ("jump probability × jump-size MAD") estimator.** Rationale for trying it: down-weight spurious jumps relative to naïve variance. Rejected because:

1. Applied to the full difference distribution of a zero-inflated (quantized) channel, MAD collapses to zero (the median of a ≥ 50%-zero sample is zero).
2. Restricting MAD to non-zero jumps only (weighted by jump probability `p = P(Δ ≠ 0)`) avoids the collapse but produces *worse* downstream calibration: under the mixture estimator, steady-state standardized innovations for CADC0890 and CADC0894 inflate to SD ≈ 2.25–2.88 (should be ≈ 1), collapsing back to SD ≈ 0.95–0.98 once the naïve estimator is restored.

**Locked choice:** naïve `Var(Δ)/2`, `Var(Δ²)/6` estimator (`USE_MIXTURE_NOISE_ESTIMATOR = False`).

## 3. Quantization floor

For channels classified `quantized`, a term `step² / 12` (variance of a Uniform(−step/2, step/2) distribution) is added to the observation-noise variance, reflecting the finite resolution of the sensor's analog-to-digital conversion. This is the one place in Layer 1 where a physical property of the instrument is folded directly into the noise model, short of a full physics-based observation model (cf. `step3b`/README §"Relationship to formal causal inference" for the broader physics-informed-SCM scope discussion).

## 4. Hierarchical Bayesian shrinkage

Per-channel `log(r)` and `log(q)` are pooled via an Efron–Morris-type empirical-Bayes shrinkage estimator:

```
global_mean  = inverse-variance-weighted mean of log(r_i) across channels
tau^2        = max(0, (weighted_SS − df) / denom)      # between-channel variance, moment estimator
w_i          = tau^2 / (tau^2 + SE_i^2)                # per-channel shrinkage weight
r_shrunk_i   = exp( w_i * log(r_i) + (1 − w_i) * global_mean )
```

This stabilizes `r`, `q` for the two channels with very few nominal points (CADC0886: n = 8 nominal segments; CADC0890: n = 3), which would otherwise be dominated by sampling noise.

**Result (locked configuration, `USE_MIXTURE_NOISE_ESTIMATOR = False`, `USE_BOCPD_FORGETTING = True`):**

| Channel | Type | r_robust | q_robust |
|---|---|---|---|
| CADC0872 | float_noise_suspect | 2.65 × 10⁻¹² | 4.83 × 10⁻¹³ |
| CADC0873 | float_noise_suspect | 2.51 × 10⁻¹² | 2.94 × 10⁻¹³ |
| CADC0874 | float_noise_suspect | 8.02 × 10⁻¹³ | 2.73 × 10⁻¹³ |
| CADC0884 | quantized | 4.86 × 10⁻⁴ | 2.83 × 10⁻⁵ |
| CADC0886 | quantized | 1.33 × 10⁻³ | 7.37 × 10⁻⁵ |
| CADC0888 | quantized | 1.28 × 10⁻³ | 8.23 × 10⁻⁵ |
| CADC0890 | continuous | 2.32 × 10⁻³ | 2.69 × 10⁻⁴ |
| CADC0892 | quantized | 1.51 × 10⁻⁴ | 3.79 × 10⁻⁵ |
| CADC0894 | quantized | 1.33 × 10⁻⁴ | 1.42 × 10⁻⁵ |

## 5. State estimation

A local-linear-trend Kalman filter is fit per channel:

```
state          x = [level, trend]
transition     A = [[1, 1], [0, 1]]
observation    H = [1, 0]
Q              = diag(0, q_shrunk)
R_effective    = r_shrunk + quantization_floor   (a Wasserstein/DR-covariance-inflation term "theta"
                                                    was scoped in an earlier design draft but never
                                                    implemented; no downstream stage depends on it)
```

This is a signal-processing convenience model, not a physical model of spacecraft attitude or orbital dynamics. Standardized innovations `z_t = (y_t − ŷ_t) / √S_t` from this filter are the sole input to Layer 2 (BOCPD).

## 6. Verification checks performed on the raw data (not model-dependent)

- `dataset.csv` (2,123 segment-level summaries) and `segments.csv` (303,493 raw observations) join cleanly on `segment`; `duration`/`len` recomputed from raw timestamps match `dataset.csv` exactly (0 mismatches).
- Every segment belongs to exactly one channel (0 multi-channel segments).
- `anomaly` (0/1) is internally consistent within every segment in `segments.csv` (0 segments with mixed anomaly status) and matches `dataset.csv`'s `anomaly` column exactly (0 mismatches).
- The secondary `label` field (`a2`/`a3`/`a4`/`anomaly`) has no documented definition in the primary publication or released code, and the original authors' own modeling code does not use it; it is therefore not used anywhere downstream, matching the source authors' practice.
- Channel observation windows are highly heterogeneous in absolute date range (CADC0872: 149 days from 2022‑01‑04; CADC0886 and CADC0890: single ≈ 6.5‑hour windows) — a contributing factor to CADC0886/0890's small sample sizes (§ Layer 2 scoping).

## 7. Outputs consumed downstream

- `channel_profiles_v2.csv` — per-channel type, noise estimates, quantization step.
- Shrunk `r`, `q`, and quantization-floor dictionaries, passed directly into the per-channel Kalman filter instantiated in Layer 2.
