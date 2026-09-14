"""
Figure 3 — Quasi-experimental placebo-pool comparison (Mann-Whitney U, full data)
and the full/train-only/test-only triple-verification agreement matrix.

Source: docs/dev-log/step3_causal_analysis.md (Sec.5 Mann-Whitney, full data)
        docs/dev-log/step3b_labeling_protocol_revalidation.md (triple comparison table)
Reproduces: results/figures/fig03_quasi_experimental.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

channels = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]
# -log10(p_bonferroni), full-data Mann-Whitney U (anomaly seg vs same-channel placebo pool)
neglog_level  = [0.0, 0.0, 0.0, 0.0, 0.0]  # all p_bonf = 1.000
neglog_diff   = [23.10, 16.62, 15.27, 5.16, 5.41]
neglog_diff2  = [21.95, 15.51, 21.54, 12.49, 3.71]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})

# --- Left: grouped bar of -log10(p_bonferroni) ---
ax = axes[0]
x = np.arange(len(channels))
w = 0.26
ax.bar(x - w, neglog_level, width=w, label="level (|d|)", color="#fdae61", edgecolor="black", linewidth=0.4)
ax.bar(x,       neglog_diff,  width=w, label="diff (log-var ratio)", color="#2c7fb8", edgecolor="black", linewidth=0.4)
ax.bar(x + w,   neglog_diff2, width=w, label="diff\u00b2 (log-var ratio)", color="#41b6c4", edgecolor="black", linewidth=0.4)
ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="$\\alpha$ = .05 (Bonferroni)")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("CADC0", "0") for c in channels])
ax.set_ylabel("$-\\log_{10}(p_{bonferroni})$")
ax.set_title("(a) Anomaly segments vs. same-channel placebo pool\n(Mann-Whitney U, 15 tests, Bonferroni-corrected)", fontsize=10.8)
ax.legend(loc="upper right", frameon=False, fontsize=9)

# --- Right: triple verification agreement matrix ---
ax2 = axes[1]
feats = ["level", "diff", "diff\u00b2"]
# rows: channel x feature, cols: full/train/test significance (1=sig,0=not,np.nan=undecidable)
data = {
    ("0872", "level"): [0, 0, 0], ("0872", "diff"): [1, 1, 1], ("0872", "diff2"): [1, 1, 1],
    ("0873", "level"): [0, 0, 0], ("0873", "diff"): [1, 1, 1], ("0873", "diff2"): [1, 1, 1],
    ("0874", "level"): [0, 0, 0], ("0874", "diff"): [1, 1, 1], ("0874", "diff2"): [1, 1, 1],
    ("0888", "level"): [0, 0, 0], ("0888", "diff"): [1, 1, 0], ("0888", "diff2"): [1, 1, 1],
    ("0894", "level"): [0, 0, 0], ("0894", "diff"): [1, 1, np.nan], ("0894", "diff2"): [1, 1, np.nan],
}
rows = list(data.keys())
mat = np.array([data[r] for r in rows], dtype=float)

cmap = plt.cm.get_cmap("RdYlGn")
im_data = np.where(np.isnan(mat), 0.5, mat)
ax2.imshow(im_data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v):
            txt, col = "n/a", "black"
        elif v == 1:
            txt, col = "sig.", "white"
        else:
            txt, col = "n.s.", "black"
        ax2.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=col)

ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["Full data", "Train-only", "Test-only"])
ax2.set_yticks(range(len(rows)))
ax2.set_yticklabels([f"{c} \u2013 {f}" for c, f in rows], fontsize=8.7)
ax2.set_title("(b) Full / train-only / test-only\ntriple-verification agreement\n(12/13 decidable comparisons agree)", fontsize=10.5)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax2.grid(which="minor", color="white", linewidth=2)
ax2.tick_params(which="minor", length=0)

plt.tight_layout()
plt.savefig("/home/claude/repo/results/figures/fig03_quasi_experimental.png", dpi=200, bbox_inches="tight")
print("saved fig03")
