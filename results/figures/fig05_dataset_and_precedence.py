"""
Figure 5 — Dataset overview (segments/anomaly ratio per channel) and the
pooled temporal-precedence test (level vs. diff vs. diff2 first-crossing time).

Grayscale / print-safe version: the per-channel sensor-name sub-labels (previously
rotated 90 degrees and overlapping the plot title / bar tops) are moved into a
two-line x-axis tick label instead, so nothing floats inside the plotting area.
The anomaly-percentage line uses black markers with a distinct dash style rather
than a red line, and bars are grey/black with hatch for the excluded channels.

Source: docs/dev-log/step1_signal_estimation.md (Sec.2, EDA table)
        docs/dev-log/step2_anomaly_detection.md (Sec.2 Wilcoxon pooled result)
Reproduces: results/figures/fig05_dataset_and_precedence.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

plt.rcParams.update({
    "font.size": 10.8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "text.color": "black",
})

channels = ["CADC0872", "CADC0873", "CADC0874", "CADC0884", "CADC0886",
            "CADC0888", "CADC0890", "CADC0892", "CADC0894"]
sensor = ["Magn. #1", "Magn. #2", "Magn. #3", "Phot. #1", "Phot. #2",
          "Phot. #3", "Phot. #4", "Phot. #5", "Phot. #6"]
n_seg = [546, 593, 194, 158, 11, 252, 14, 211, 144]
anom_pct = [24.0, 17.7, 35.6, 0.0, 27.3, 23.8, 78.6, 16.1, 14.6]
final_scope = [True, True, True, False, False, True, False, False, True]

fig = plt.figure(figsize=(15.8, 6.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
ax3 = fig.add_subplot(gs[0, 1])

# --- Left: segment counts + anomaly ratio ---
x = np.arange(len(channels))
C_IN, C_OUT = "#333333", "#dcdcdc"
colors = [C_IN if f else C_OUT for f in final_scope]
hatches = [None if f else "////" for f in final_scope]
for xi, n, c, h in zip(x, n_seg, colors, hatches):
    ax.bar(xi, n, color=c, edgecolor="black", linewidth=0.8, hatch=h, width=0.68)

# Single-line channel-code tick labels only (sensor type is stated once in the
# panel title / caption instead of repeated per-bar, which is what caused the
# earlier crowding). A thin sensor-type strip beneath the axis replaces the
# per-bar text label with a compact, non-overlapping visual code instead.
ax.set_xticks(x)
ax.set_xticklabels([c.replace("CADC0", "0") for c in channels], fontsize=10.5)
ax.tick_params(axis="x", length=0, pad=20)

is_magnetometer = [s.startswith("Magn") for s in sensor]
for xi, is_m in zip(x, is_magnetometer):
    ax.annotate("Magn." if is_m else "Phot.", xy=(xi, -0.135), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=7.6, color="#555555", annotation_clip=False)
ax.set_ylabel("# segments", fontsize=11.5)
ax.set_ylim(0, 660)

ax3b = ax.twinx()
ax3b.plot(x, anom_pct, marker="o", markersize=7, linewidth=1.8, color="black",
          markerfacecolor="white", markeredgewidth=1.6, linestyle="--")
ax3b.set_ylabel("Anomaly segments (%)", fontsize=11)
ax3b.set_ylim(0, 92)

ax.set_title("(a) OPS-SAT-AD dataset: 2,123 univariate segments\n9 channels (3 magnetometer + 6 photodiode)",
              fontsize=12.2, pad=10)

legend_handles = [
    mpatches.Patch(facecolor=C_IN, edgecolor="black", label="final 5-channel scope"),
    mpatches.Patch(facecolor=C_OUT, edgecolor="black", hatch="////", label="excluded"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="--", markerfacecolor="white",
                  markersize=7, label="anomaly %  (right axis)"),
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.46, 1.30),
          ncol=3, frameon=False, fontsize=9.2, handlelength=1.7, columnspacing=1.2)

# --- Right: pooled Wilcoxon precedence test ---
pairs = ["level vs. diff", "level vs. diff\u00b2", "diff vs. diff\u00b2"]
p_vals = [0.000014, 0.000032, 0.367707]
frac_level_first = [0.157, 0.171, None]
neglogp = [-np.log10(p) for p in p_vals]
colors3 = ["#4d4d4d", "#1a1a1a", "#bfbfbf"]
hatches3 = ["///", None, "..."]
ax3.bar(pairs, neglogp, color=colors3, edgecolor="black", linewidth=0.8, hatch=hatches3, width=0.55)
ax3.set_xlim(-0.65, 2.65)
ax3.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1.1)
ax3.text(-0.58, -np.log10(0.05) + 0.22, "$\\alpha$=.05", ha="left", va="bottom", fontsize=8.6, color="#333333")

for xi, (p, frac) in enumerate(zip(p_vals, frac_level_first)):
    label = f"p={p:.1e}"
    if frac is not None:
        label += f"\ndiff/diff\u00b2 precedes level\nin {(1-frac)*100:.1f}% of pairs"
    else:
        label += "\n(n.s. \u2014 no reliable\nordering)"
    va = "bottom"
    y0 = neglogp[xi] + 0.35 if neglogp[xi] > 1.0 else neglogp[xi] + 1.4
    ax3.text(xi, y0, label, ha="center", va=va, fontsize=8.4)

ax3.set_ylabel("$-\\log_{10}(p)$", fontsize=11.5)
ax3.set_ylim(0, 7.6)
ax3.set_title("(b) Pooled temporal-precedence test (n=70\u2013161\npaired segments, 5 channels)",
              fontsize=12.2, pad=10)

plt.savefig("/home/claude/repo/results/figures/fig05_dataset_and_precedence.png", dpi=220,
            bbox_inches="tight", facecolor="white")
print("saved fig05")
