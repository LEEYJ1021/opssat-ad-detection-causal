"""
fig04_scm_and_heterogeneity.py

Reproduces Figure 4 (descriptive structural working model, and
DerSimonian-Laird heterogeneity across the 5 scoped OPS-SAT-AD channels)
from:
    results/layer3/heterogeneity_summary.csv

Falls back to Table 4 of the README when that artifact is not present.

Output: results/figures/fig04_scm_and_heterogeneity.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig04_scm_and_heterogeneity.png")

HETEROGENEITY_CSV = os.path.join(RESULTS, "layer3", "heterogeneity_summary.csv")

# --- Table 4 fallback ------------------------------------------------------
TABLE4 = pd.DataFrame(
    [
        ("Effect size |d| (level)", 65.7, 0.0202),
        ("Effect size |d| (diff)", 97.2, 0.0001),
        ("Effect size |d| (diff\u00b2)", 93.8, 2.6e-13),
        ("frac_sig (level)", 82.8, 1.1e-4),
        ("frac_sig (diff)", 93.8, 3.2e-13),
        ("frac_sig (diff\u00b2)", 92.2, 1.9e-10),
    ],
    columns=["outcome", "i2", "q_p"],
)


def load_heterogeneity():
    if os.path.exists(HETEROGENEITY_CSV):
        return pd.read_csv(HETEROGENEITY_CSV)
    print(f"[fig04] {HETEROGENEITY_CSV} not found; using Table 4 fallback.")
    return TABLE4


def draw_scm(ax):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    def box(y, h, text, facecolor, textcolor="black", dashed=False, fontsize=9):
        style = dict(facecolor=facecolor, edgecolor="black", linewidth=1.4)
        if dashed:
            style["linestyle"] = (0, (5, 3))
        ax.add_patch(plt.Rectangle((0.3, y), 6.4, h, **style))
        ax.text(0.3 + 3.2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=textcolor)

    box(9.6, 1.5, "Onset\n(BOCPD-estimated change-point)", "0.05", "white", fontsize=10)
    ax.annotate("", xy=(3.5, 8.0), xytext=(3.5, 9.6), arrowprops=dict(arrowstyle="->", lw=2))
    box(6.3, 1.7, "diff / diff\u00b2 variance surge\n(transient, onset-localised)", "0.85", fontsize=9.5)
    ax.annotate("weak / dataset-conditional", (3.5, 5.8), ha="center", fontsize=8, style="italic")
    ax.annotate("", xy=(3.5, 4.4), xytext=(3.5, 6.3), arrowprops=dict(arrowstyle="->", lw=1.2, linestyle="dashed", color="0.4"))
    box(2.7, 1.7, "Level shift\n(persistent in ~half of segments; no\ndetectable excess over placebo in\n"
                  "OPS-SAT-AD, but detectable in\nSMAP/MSL and SMD)", "white", dashed=True, fontsize=8)
    box(0.2, 1.8, "Moderator (aliased with sensor family):\nfloat-noise / magnetometer (872/873/874) \u2192 strong\n"
                  "quantized / photodiode (888/894) \u2192 weaker", "0.92", fontsize=7.5)

    ax.text(7.4, 8.7, "Effect size |d|", rotation=90, ha="center", va="center", fontsize=8)
    ax.text(7.4, 3.5, "Significant-segment\nfraction", rotation=90, ha="center", va="center", fontsize=8)
    ax.plot([7.1, 7.1], [6.3, 9.6], color="black", linewidth=0.8)
    ax.plot([7.1, 7.1], [2.7, 4.4], color="black", linewidth=0.8)


def main():
    het = load_heterogeneity()

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    ax_a.set_title("a  Descriptive working model (not identified)", loc="left", fontweight="bold", fontsize=12)
    draw_scm(ax_a)

    # --- Panel b: heterogeneity bars ---------------------------------------
    y = range(len(het))[::-1]
    hatches = []
    colors = []
    for outcome in het["outcome"]:
        if "level" in outcome:
            colors.append("white"); hatches.append("..")
        elif "(diff)" in outcome:
            colors.append("0.75"); hatches.append("//")
        else:
            colors.append("0.1"); hatches.append(None)

    bars = ax_b.barh(list(y), het["i2"], color=colors, hatch=hatches, edgecolor="black", height=0.6)
    for yi, i2, qp in zip(y, het["i2"], het["q_p"]):
        ax_b.text(i2 + 1.5, yi, f"I\u00b2 = {i2:.1f}%   Q p = {qp:.2g}", va="center", fontsize=9)

    ax_b.set_yticks(list(y))
    ax_b.set_yticklabels(het["outcome"])
    ax_b.set_xlim(0, 100)
    for v, lab in [(25, "low"), (50, "moderate"), (75, "high")]:
        ax_b.axvline(v, color="0.6", linestyle=":", linewidth=1)
        ax_b.text(v, len(het) + 0.3, lab, ha="center", fontsize=9, style="italic")
    ax_b.set_xlabel("I\u00b2 (% of variance from between-channel heterogeneity)")
    ax_b.set_title("DerSimonian\u2013Laird random-effects heterogeneity, k = 5 channels", fontsize=11)
    ax_b.text(0, -1.3, "Descriptive: with k = 5, I\u00b2 and Q are imprecise. Bands follow Higgins\u2013Thompson conventions.",
              fontsize=8, style="italic")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig04] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
