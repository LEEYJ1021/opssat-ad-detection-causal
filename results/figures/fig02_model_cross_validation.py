"""
Figure 2 — 16-model cross-validation: normalized feature-importance share
(level / diff / diff2) per model, sorted by AUC, plus Kendall's W summary.

Grayscale / print-safe version: the three feature categories are encoded with
three distinct grey levels AND three distinct hatch patterns so the stacked
bars remain distinguishable in black-and-white print. Model family is shown
as a bracketed label so the AUC values never crowd the bar labels.

Source: docs/dev-log/step_ai_model_cross_validation.md, Table 1 & Table 3.
Reproduces: results/figures/fig02_model_cross_validation.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size": 10.8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "text.color": "black",
})

# model, family, AUC, level, diff, diff2
rows = [
    ("TinyTransformer", "Attention", 0.994, 0.178, 0.438, 0.384),
    ("TCN", "Conv (dilated causal)", 0.993, 0.200, 0.380, 0.419),
    ("BiGRU", "Recurrent", 0.989, 0.285, 0.388, 0.327),
    ("CNN1D", "Convolutional", 0.983, 0.303, 0.301, 0.396),
    ("BiLSTM", "Recurrent", 0.980, 0.211, 0.497, 0.292),
    ("LightMamba", "State-space (SSM)", 0.969, 0.428, 0.441, 0.130),
    ("SVM_RBF", "Kernel", 0.948, 0.061, 0.466, 0.472),
    ("LightGBM", "Tree ens. (boosting)", 0.940, 0.232, 0.459, 0.309),
    ("XGBoost", "Tree ens. (boosting)", 0.935, 0.148, 0.419, 0.433),
    ("RandomForest", "Tree ens. (bagging)", 0.931, 0.084, 0.456, 0.460),
    ("ExtraTrees", "Tree ens. (bagging,extra)", 0.927, 0.145, 0.389, 0.466),
    ("GradBoost_sklearn", "Tree ens. (boosting)", 0.924, 0.068, 0.371, 0.561),
    ("LogReg_L2", "Linear", 0.920, 0.060, 0.407, 0.534),
    ("LogReg_L1", "Linear (sparse)", 0.920, 0.057, 0.412, 0.531),
    ("kNN", "Instance-based", 0.918, 0.129, 0.391, 0.480),
    ("GaussianNB", "Probabilistic", 0.900, 0.051, 0.446, 0.503),
]
models = [r[0] for r in rows]
auc = np.array([r[2] for r in rows])
level = np.array([r[3] for r in rows])
diff = np.array([r[4] for r in rows])
diff2 = np.array([r[5] for r in rows])

C_LEVEL, C_DIFF, C_DIFF2 = "#e8e8e8", "#8a8a8a", "#1a1a1a"
H_LEVEL, H_DIFF, H_DIFF2 = "...", "///", None

fig = plt.figure(figsize=(15.8, 8.4))
gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# --- Left: stacked horizontal bar of importance share, sorted by AUC (already sorted) ---
y = np.arange(len(models))[::-1]
ax.barh(y, level, color=C_LEVEL, edgecolor="black", height=0.70, hatch=H_LEVEL, linewidth=0.7, label="level")
ax.barh(y, diff, left=level, color=C_DIFF, edgecolor="black", height=0.70, hatch=H_DIFF, linewidth=0.7, label="diff")
ax.barh(y, diff2, left=level + diff, color=C_DIFF2, edgecolor="black", height=0.70, hatch=H_DIFF2, linewidth=0.7, label="diff\u00b2")

ax.set_yticks(y)
ax.set_yticklabels([f"{m}" for m in models], fontsize=10.2)
# AUC placed as a right-aligned annotation just past x=1.0, clear of the bars.
for yi, a in zip(y, auc):
    ax.text(1.015, yi, f"AUC={a:.3f}", va="center", ha="left", fontsize=8.7, color="#333333")

ax.set_xlabel("Normalized feature-importance share", fontsize=11.5)
ax.set_xlim(0, 1.145)
ax.set_title("(a) Feature attribution across 16 independent model classes\n"
              "(tabular: SHAP / permutation \u2014 sequence: integrated-gradients saliency)",
              fontsize=12.2, pad=14)
ax.axvline(1/3, color="black", linestyle=":", linewidth=1.1)
ax.text(1/3, len(models) - 0.15, "chance (1/3)", ha="center", va="bottom", fontsize=8, color="#444")

legend_handles = [
    mpatches.Patch(facecolor=C_LEVEL, edgecolor="black", hatch=H_LEVEL, label="level"),
    mpatches.Patch(facecolor=C_DIFF, edgecolor="black", hatch=H_DIFF, label="diff"),
    mpatches.Patch(facecolor=C_DIFF2, edgecolor="black", label="diff\u00b2"),
]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.42, -0.145),
          ncol=3, frameon=False, fontsize=10.2, handlelength=1.8)

# --- Right: summary stats panel ---
ax2.axis("off")
stats_text = (
    "Cross-model agreement (N = 16)\n"
    "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
    "Kendall's W  =  0.609\n"
    "   ($\\chi^2$ = 19.50, df = 2, p < .001)\n\n"
    "Models ranking level\n"
    "as top feature:  0 / 16\n"
    "   (binomial p = .0015)\n\n"
    "Bootstrap stability\n"
    "(200 resamples):\n"
    "   RandomForest:  100%\n"
    "   LogReg:            100%\n"
    "   \u2014 both favor diff/diff\u00b2\n\n"
    "13 / 13 inductive-bias\n"
    "families favor diff+diff\u00b2\n"
    "over level\n"
    "(narrowest margin: SSM family)\n\n"
    "Tabular (n=10) and sequence\n"
    "(n=6) representations agree\n"
    "\u2014 rules out summary-statistic\n"
    "design as a confound"
)
ax2.text(0.03, 0.97, stats_text, transform=ax2.transAxes, va="top", ha="left",
         fontsize=10.6, linespacing=1.65, family="DejaVu Sans",
         bbox=dict(boxstyle="round,pad=0.65", facecolor="white", edgecolor="black", linewidth=1.1))
ax2.set_title("(b) Agreement statistics", fontsize=12.2, pad=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.savefig("/home/claude/repo/results/figures/fig02_model_cross_validation.png", dpi=220,
            bbox_inches="tight", facecolor="white")
print("saved fig02")
