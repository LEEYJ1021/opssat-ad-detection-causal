"""
fig03_quasi_experimental.py

Reproduces Figure 3 (quasi-experimental placebo-pool test and full/train/
test triple-verification agreement, OPS-SAT-AD) from:
    results/layer3/placebo_comparison.csv
    results/layer3/triple_verification_matrix.csv

Falls back to Table 3 / Table 5 of the README when those artifacts are
not present.

Output: results/figures/fig03_quasi_experimental.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig03_quasi_experimental.png")

PLACEBO_CSV = os.path.join(RESULTS, "layer3", "placebo_comparison.csv")
TRIPLE_CSV = os.path.join(RESULTS, "layer3", "triple_verification_matrix.csv")

# --- Table 3 fallback ----------------------------------------------------
TABLE3 = pd.DataFrame(
    [
        ("CADC0872", "magnetometer\nfloat-noise", 1.0, 7.94e-24, 1.12e-22),
        ("CADC0873", "magnetometer\nfloat-noise", 1.0, 2.40e-17, 3.09e-16),
        ("CADC0874", "magnetometer\nfloat-noise", 1.0, 5.37e-16, 2.88e-22),
        ("CADC0888", "photodiode\nquantized", 1.0, 6.92e-6, 3.24e-13),
        ("CADC0894", "photodiode\nquantized", 1.0, 3.89e-6, 1.95e-4),
    ],
    columns=["channel", "group", "level_p_bonf", "diff_p_bonf", "diff2_p_bonf"],
)

# --- Table 5 fallback ------------------------------------------------------
TABLE5 = pd.DataFrame(
    [
        ("0872", "level", "n.s.", "n.s.", "n.s.", "Y"),
        ("0872", "diff", "sig.", "sig.", "sig.", "Y"),
        ("0872", "diff\u00b2", "sig.", "sig.", "sig.", "Y"),
        ("0873", "level", "n.s.", "n.s.", "n.s.", "Y"),
        ("0873", "diff", "sig.", "sig.", "sig.", "Y"),
        ("0873", "diff\u00b2", "sig.", "sig.", "sig.", "Y"),
        ("0874", "level", "n.s.", "n.s.", "n.s.", "Y"),
        ("0874", "diff", "sig.", "sig.", "sig.", "Y"),
        ("0874", "diff\u00b2", "sig.", "sig.", "sig.", "Y"),
        ("0888", "level", "n.s.", "n.s.", "n.s.", "Y"),
        ("0888", "diff", "sig.", "sig.", "n.s. (p=0.051)", "N"),
        ("0888", "diff\u00b2", "sig.", "sig.", "sig.", "Y"),
        ("0894", "level", "n.s.", "n.s.", "n.s.", "Y"),
        ("0894", "diff", "sig.", "sig.", "n/a (n=4<5)", "\u2013"),
        ("0894", "diff\u00b2", "sig.", "sig.", "n/a (n=4<5)", "\u2013"),
    ],
    columns=["channel", "feature", "full", "train", "test", "agree"],
)


def load_placebo():
    if os.path.exists(PLACEBO_CSV):
        return pd.read_csv(PLACEBO_CSV)
    print(f"[fig03] {PLACEBO_CSV} not found; using Table 3 fallback.")
    return TABLE3


def load_triple():
    if os.path.exists(TRIPLE_CSV):
        return pd.read_csv(TRIPLE_CSV)
    print(f"[fig03] {TRIPLE_CSV} not found; using Table 5 fallback.")
    return TABLE5


def main():
    placebo = load_placebo()
    triple = load_triple()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.3, 1]})

    # --- Panel a: -log10(p_bonf) bars ---------------------------------------
    x = np.arange(len(placebo))
    width = 0.35
    diff_vals = -np.log10(placebo["diff_p_bonf"])
    diff2_vals = -np.log10(placebo["diff2_p_bonf"])

    ax_a.bar(x - width / 2, diff_vals, width, facecolor="0.75", hatch="//", edgecolor="black", label="diff")
    ax_a.bar(x + width / 2, diff2_vals, width, facecolor="0.1", edgecolor="black", label="diff\u00b2")
    ax_a.scatter(x, np.zeros(len(x)), marker="o", facecolor="white", edgecolor="black",
                s=60, zorder=5, label="level: p_Bonf capped at 1.000 (no bar)")

    for xi, v in zip(x, diff_vals):
        ax_a.text(xi - width / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x, diff2_vals):
        ax_a.text(xi + width / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    ax_a.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax_a.text(len(x) - 0.5, -np.log10(0.05) + 0.3, "\u03b1 = 0.05", fontsize=8, ha="right")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([c.replace("CADC0", "") for c in placebo["channel"]])
    ax_a.set_ylabel(r"$-\log_{10}\ p_{Bonferroni}$")
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=14)
    ax_a.legend(fontsize=8, loc="upper right")

    # sensor-family grouping underline
    if "group" in placebo.columns:
        groups = placebo["group"].tolist()
        seen = []
        start = 0
        for i, g in enumerate(groups + [None]):
            if g != (groups[start] if i < len(groups) else None) or i == len(groups):
                if start < i:
                    mid = (start + i - 1) / 2
                    ax_a.annotate("", xy=(start - 0.4, -0.06), xytext=(i - 1 + 0.4, -0.06),
                                  xycoords=("data", "axes fraction"),
                                  arrowprops=dict(arrowstyle="-", lw=1))
                    ax_a.annotate(groups[start], (mid, -0.13), xycoords=("data", "axes fraction"),
                                  ha="center", fontsize=8, style="italic", annotation_clip=False)
                start = i

    ax_a.text(0, -1, "Dashed line: \u03b1 = 0.05 after Bonferroni correction (15 tests). Each feature is "
              "tested on its own statistic\n(one-sided Mann\u2013Whitney U vs. same-channel placebo pool); "
              "bar heights compare distinguishability\nfrom placebo, not effect sizes across features. "
              "Type is aliased with sensor.",
              transform=ax_a.transAxes, fontsize=7.5, style="italic", va="top")

    # --- Panel b: full/train/test agreement matrix --------------------------
    ax_b.axis("off")
    rows = list(triple.itertuples(index=False))
    n_rows = len(rows)
    col_labels = ["", "Full", "Train", "Test", "3-way"]
    col_x = [0, 3.0, 4.2, 5.4, 6.8]
    row_h = 0.6

    def cellcolor(v):
        v = str(v)
        if v.startswith("sig"):
            return "0.1", "white"
        if v.startswith("n.s"):
            return "white", "black"
        return "0.9", "black"

    for j, lab in enumerate(col_labels):
        ax_b.text(col_x[j], n_rows * row_h + 0.5, lab, fontsize=9, fontweight="bold", ha="left")

    for i, row in enumerate(rows):
        y = (n_rows - 1 - i) * row_h
        label = f"{row.channel}  {row.feature}"
        ax_b.text(col_x[0], y, label, fontsize=8.5, ha="left", va="center")
        for j, val in enumerate([row.full, row.train, row.test], start=1):
            fc, tc = cellcolor(val)
            ax_b.add_patch(plt.Rectangle((col_x[j] - 0.1, y - row_h / 2 + 0.05), 1.1, row_h - 0.1,
                                          facecolor=fc, edgecolor="black"))
            ax_b.text(col_x[j] + 0.45, y, str(val), ha="center", va="center", fontsize=7.5,
                      color=tc)
        agree_style = dict(fontweight="bold")
        if str(row.agree) == "N":
            ax_b.add_patch(plt.Rectangle((col_x[0] - 0.1, y - row_h / 2), col_x[3] - col_x[0] + 1.4, row_h,
                                          fill=False, edgecolor="black", linewidth=2.2))
        ax_b.text(col_x[4], y, str(row.agree), ha="center", va="center", fontsize=9, **agree_style)

    ax_b.set_xlim(-0.5, 8)
    ax_b.set_ylim(-1, n_rows * row_h + 1)
    ax_b.set_title("b  12 of 13 decidable comparisons agree across slices",
                    loc="left", fontweight="bold", fontsize=12)
    ax_b.text(-0.5, -0.8,
              "Framed row: the single disagreement (p just above threshold\nin the smaller test slice). "
              "\u2013 = not decidable (<5 anomalous\nsegments in the test slice).",
              fontsize=7.5, style="italic", va="top")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig03] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
