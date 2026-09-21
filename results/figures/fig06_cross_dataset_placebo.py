"""
fig06_cross_dataset_placebo.py

Reproduces Figure 6 (placebo-pool separation across OPS-SAT-AD, SMAP/MSL
and SMD) from:
    results/layer3/placebo_comparison.csv                          (OPS-SAT-AD)
    results/external_validation/stage1_placebo_pooled_dataset.csv  (Table 12 pooled rows)
    results/external_validation/stage1_channel_level_significance.csv (SMD channel-level counts)

Falls back to Table 3 (weakest OPS-SAT-AD channel) and Table 12 / Stage 1
text of the README when those artifacts are not present.

Output: results/figures/fig06_cross_dataset_placebo.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig06_cross_dataset_placebo.png")

OPSSAT_PLACEBO_CSV = os.path.join(RESULTS, "layer3", "placebo_comparison.csv")
POOLED_CSV = os.path.join(RESULTS, "external_validation", "stage1_placebo_pooled_dataset.csv")
CHANNEL_LEVEL_CSV = os.path.join(RESULTS, "external_validation", "stage1_channel_level_significance.csv")

# --- Table 3 fallback: weakest-of-5 OPS-SAT-AD channel (CADC0894) ----------
# level: p_bonf capped at 1.000 -> -log10(p) = 0
OPSSAT_WEAKEST = {"level_neglog10p": 0.0, "diff_neglog10p": 5.2, "diff2_neglog10p": 3.7}

# --- Table 12 fallback: dataset-pooled placebo comparison -------------------
POOLED = pd.DataFrame(
    [
        ("OPS-SAT-AD", "weakest of 5 channels\n(Bonferroni)", 0.0, 5.2, 3.7),
        ("SMAP/MSL", "pooled, 88 windows\n(BH-FDR)", 1.5, 8.3, 5.9),
        ("SMD", "pooled, 11,493 windows\n(BH-FDR)", 37.1, 166.4, 165.9),
    ],
    columns=["dataset", "scope", "level_neglog10p", "diff_neglog10p", "diff2_neglog10p"],
)

# --- SMD channel-level breadth fallback (§ Stage 1 text) -------------------
CHANNEL_LEVEL = pd.DataFrame(
    [
        ("level", 97, 1038),
        ("diff", 142, 1038),
        ("diff2", 145, 1038),
    ],
    columns=["feature", "k_significant", "n_channels"],
)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    adj = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)
    return (center - adj) / denom, (center + adj) / denom


def load_pooled():
    if os.path.exists(POOLED_CSV):
        return pd.read_csv(POOLED_CSV)
    print(f"[fig06] {POOLED_CSV} not found; using Table 12 fallback.")
    return POOLED


def load_channel_level():
    if os.path.exists(CHANNEL_LEVEL_CSV):
        return pd.read_csv(CHANNEL_LEVEL_CSV)
    print(f"[fig06] {CHANNEL_LEVEL_CSV} not found; using Stage-1 text fallback.")
    return CHANNEL_LEVEL


def main():
    pooled = load_pooled()
    chan = load_channel_level()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1]})

    # --- Panel a: dot plot of -log10(p) with a broken axis ------------------
    y_positions = np.arange(len(pooled))[::-1]
    markers = {"level": ("o", "white"), "diff": ("s", "0.7"), "diff2": ("D", "0.05")}
    offset = {"level": 0.22, "diff": 0, "diff2": -0.22}

    for feat, (marker, color) in markers.items():
        col = f"{feat}_neglog10p"
        for yi, val in zip(y_positions, pooled[col]):
            plot_val = min(val, 10) if val > 10 else val  # compress values beyond the break for display
            ax_a.plot([plot_val], [yi + offset[feat]], marker=marker, color=color,
                      markeredgecolor="black", markersize=10, linestyle="none")
            ax_a.text(plot_val + 0.3, yi + offset[feat], f"{val:.1f}", va="center", fontsize=8)

    ax_a.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax_a.text(-np.log10(0.05), len(pooled) - 0.2, "\u03b1 = 0.05", fontsize=8, ha="left")
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([f"{d}\n{s}" for d, s in zip(pooled["dataset"], pooled["scope"])], fontsize=9)
    ax_a.set_xlim(0, 12)
    ax_a.set_xticks([0, 2, 4, 6, 8, 10])
    ax_a.set_xticklabels(["0", "2", "4", "6", "8", "150\u2013200"])
    ax_a.set_xlabel(r"$-\log_{10}$ adjusted p  (axis broken)")
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=14)
    from matplotlib.lines import Line2D
    ax_a.legend(handles=[
        Line2D([0], [0], marker="o", color="white", markeredgecolor="black", markersize=10, linestyle="none", label="level"),
        Line2D([0], [0], marker="s", color="0.7", markeredgecolor="black", markersize=10, linestyle="none", label="diff"),
        Line2D([0], [0], marker="D", color="0.05", markeredgecolor="black", markersize=10, linestyle="none", label="diff\u00b2"),
    ], loc="lower right", fontsize=9)
    ax_a.text(6, len(pooled) - 0.3,
              "nominal: windows are nested in\nchannels/machines, not independent",
              fontsize=8, style="italic", ha="center")

    # --- Panel b: SMD channel-level significant fraction with Wilson CI ----
    x = np.arange(len(chan))
    fracs = chan["k_significant"] / chan["n_channels"] * 100
    los, his = zip(*[wilson_ci(k, n) for k, n in zip(chan["k_significant"], chan["n_channels"])])
    los = np.array(los) * 100
    his = np.array(his) * 100
    yerr = np.vstack([fracs - los, his - fracs])

    bar_colors = ["white", "0.7", "0.05"]
    bar_hatches = ["..", "//", None]
    bars = ax_b.bar(x, fracs, color=bar_colors, hatch=bar_hatches, edgecolor="black",
                    yerr=yerr, capsize=5)
    for xi, f, k, n in zip(x, fracs, chan["k_significant"], chan["n_channels"]):
        ax_b.text(xi, f + 1.5, f"{int(k)}/{int(n)}\n{f:.1f}%", ha="center", fontsize=9)

    feat_labels = {"level": "level", "diff": "diff", "diff2": "diff\u00b2"}
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([feat_labels.get(f, f) for f in chan["feature"]])
    ax_b.set_ylabel("Channels significant (%)")
    ax_b.set_title("SMD, channel level\n(derivative features flag ~1.5\u00d7 as many channels)", fontsize=10)
    ax_b.text(0, -0.15, "95% Wilson intervals", transform=ax_b.transAxes, fontsize=8, style="italic")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig06] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
