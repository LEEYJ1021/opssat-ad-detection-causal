"""
Figure 4 — Final SCM skeleton (Path A) and channel-level heterogeneity (I^2)
from the DerSimonian-Laird random-effects meta-analysis.

Grayscale / print-safe version: SCM boxes use a black/dark-grey/light-grey scale
instead of purple/blue. The "very high heterogeneity" threshold annotation is
moved below the axis frame (previously it collided with the panel title) and the
three outcome-feature groups are distinguished by grey level + hatch, not colour.

Source: docs/dev-log/step3_causal_analysis.md (SCM artifact 1; meta-analysis I^2 table)
Reproduces: results/figures/fig04_scm_and_heterogeneity.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10.8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "text.color": "black",
})

fig = plt.figure(figsize=(15.8, 7.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.32)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# --- Left: SCM diagram (greyscale) ---
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

def box(ax, xy, w, h, text, fc, fontsize=11, fontcolor="white", weight="bold", edgecolor="black"):
    x, y = xy
    ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=fc, edgecolor=edgecolor,
                                linewidth=1.3, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=fontcolor,
            fontweight=weight, zorder=3)

box(ax, (0.5, 0.90), 0.52, 0.115, "Onset\n(BOCPD change-point)", "#1a1a1a")
box(ax, (0.5, 0.615), 0.64, 0.155, "Diff / Diff\u00b2 variance surge\n(transient, onset-localized)", "#4d4d4d")
box(ax, (0.5, 0.30), 0.60, 0.155, "Level shift\n(persistent, but confounded)", "#e0e0e0", fontcolor="black")

ax.annotate("", xy=(0.5, 0.695), xytext=(0.5, 0.845),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))
ax.annotate("", xy=(0.5, 0.38), xytext=(0.5, 0.535),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.5, linestyle="dashed"))
ax.text(0.57, 0.458, "weak / uncertain\n(not part of causal core)", fontsize=8.2, color="#555555", ha="left")

ax.text(0.5, 0.135,
        "moderator: float_noise_suspect (0872/0873/0874) \u2192 strong effect\n"
        "quantized (0888/0894) \u2192 weak effect",
        fontsize=8.4, ha="center", va="center", color="#222222",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#999999", linewidth=0.8))

ax.text(0.5, 0.02,
        "Level excluded as causal core: Mann-Whitney vs. placebo pool\nnon-significant in all 5 channels (p_bonf = 1.000)",
        ha="center", va="bottom", fontsize=8.8, color="black",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor="black", linewidth=1.0))

ax.set_title("(a) Final SCM skeleton \u2014 Path A\n(signal-processing-structure-based causal core)",
              fontsize=12.2, pad=14)

# --- Right: I^2 heterogeneity summary (2 outcomes x 3 features), pure greyscale ---
outcomes = ["Effect size |d|  (level)", "Effect size |d|  (diff)", "Effect size |d|  (diff\u00b2)",
            "frac_sig  (level)", "frac_sig  (diff)", "frac_sig  (diff\u00b2)"]
I2 = [65.7, 97.2, 93.8, 82.8, 93.8, 92.2]
Q_p = [0.0202, 0.0001, 2.6e-13, 1.1e-4, 3.2e-13, 1.9e-10]

def style_for(o):
    if "(level)" in o:
        return "#e8e8e8", "..."
    elif "(diff)" in o and "diff\u00b2" not in o:
        return "#8a8a8a", "///"
    else:
        return "#1a1a1a", None

colors, hatches = zip(*[style_for(o) for o in outcomes])

y = np.arange(len(outcomes))[::-1]
for yi, v, c, h in zip(y, I2, colors, hatches):
    ax2.barh(yi, v, color=c, edgecolor="black", linewidth=0.8, hatch=h, height=0.62)
for yi, v, p in zip(y, I2, Q_p):
    pstr = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
    ax2.text(v + 1.8, yi, f"{v:.1f}%   (Q-test p={pstr})", va="center", fontsize=8.8, color="black")

ax2.set_yticks(y)
ax2.set_yticklabels(outcomes, fontsize=9.6)
ax2.set_xlim(0, 128)
ax2.set_xlabel("$I^2$ (% of total variance due to between-channel heterogeneity)", fontsize=10.6)

# Threshold line + its label placed BELOW the plotted bars (own row), never touching the title.
ax2.axvline(75, color="black", linestyle="--", linewidth=1.1)
ax2.annotate('"very high" heterogeneity threshold  ($I^2$ = 75%)',
             xy=(75, y.min() - 0.85), xycoords="data",
             ha="center", va="top", fontsize=8.4, color="#333333", annotation_clip=False)
ax2.set_ylim(y.min() - 1.35, y.max() + 0.75)

ax2.set_title("(b) DerSimonian\u2013Laird random-effects meta-analysis:\nheterogeneity ($I^2$), k = 5 channels",
              fontsize=12.2, pad=14)

plt.savefig("/home/claude/repo/results/figures/fig04_scm_and_heterogeneity.png", dpi=220,
            bbox_inches="tight", facecolor="white")
print("saved fig04")
