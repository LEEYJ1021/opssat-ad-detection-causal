"""
fig07_precedence_baseline.py

Reproduces Figure 7 (operator-level baseline vs. anomalous precedence
ordering, and the anomaly-attributable increment, across OPS-SAT-AD,
SMAP/MSL and SMD) from:
    results/layer3/temporal_precedence_normal_control.csv          (OPS-SAT-AD)
    results/external_validation/stage2_temporal_precedence.csv     (Table 13 precedence rows)
    results/external_validation/stage2_baseline_decomposition.csv  (Table 13 increment rows)

Falls back to Table 13 of the README when those artifacts are not present.

Output: results/figures/fig07_precedence_baseline.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig07_precedence_baseline.png")

PRECEDENCE_CSV = os.path.join(RESULTS, "external_validation", "stage2_temporal_precedence.csv")
DECOMP_CSV = os.path.join(RESULTS, "external_validation", "stage2_baseline_decomposition.csv")

# --- Table 13 fallback: precedence ordering (n, % diff-first) --------------
PRECEDENCE = pd.DataFrame(
    [
        ("OPS-SAT-AD", 70, 224, 84.3, 44.2, False),
        ("SMAP/MSL", 40, 7596, 67.5, 56.5, True),   # anomalous result n.s.
        ("SMD", 3612, 7304, 76.6, 72.7, False),
    ],
    columns=["dataset", "n_anom", "n_normal", "pct_anom", "pct_normal", "anom_ns"],
)

# --- Table 13 fallback: baseline decomposition (increment, 95% CI) ---------
DECOMP = pd.DataFrame(
    [
        ("OPS-SAT-AD", 40.1, 29.4, 50.8),
        ("SMAP/MSL", 11.0, -3.6, 25.6),
        ("SMD", 3.9, 2.2, 5.6),
    ],
    columns=["dataset", "increment", "ci_lo", "ci_hi"],
)


def wilson_ci(pct, n, z=1.96):
    """Wilson interval for a percentage, given the pair count n."""
    if n == 0:
        return pct, pct
    phat = pct / 100
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    adj = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)
    return (center - adj) / denom * 100, (center + adj) / denom * 100


def load_precedence():
    if os.path.exists(PRECEDENCE_CSV):
        return pd.read_csv(PRECEDENCE_CSV)
    print(f"[fig07] {PRECEDENCE_CSV} not found; using Table 13 fallback.")
    return PRECEDENCE


def load_decomposition():
    if os.path.exists(DECOMP_CSV):
        return pd.read_csv(DECOMP_CSV)
    print(f"[fig07] {DECOMP_CSV} not found; using Table 13 fallback.")
    return DECOMP


def main():
    prec = load_precedence()
    decomp = load_decomposition()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})

    # --- Panel a: normal-control vs. anomalous fractions with Wilson CIs ---
    y = np.arange(len(prec))[::-1]
    for yi, (_, row) in zip(y, prec.iterrows()):
        lo_n, hi_n = wilson_ci(row["pct_normal"], row["n_normal"])
        lo_a, hi_a = wilson_ci(row["pct_anom"], row["n_anom"])

        ax_a.plot([lo_n, hi_n], [yi, yi], color="black", linewidth=1.2, zorder=1)
        ax_a.plot([lo_a, hi_a], [yi, yi], color="black", linewidth=1.2, zorder=1)
        ax_a.scatter([row["pct_normal"]], [yi], marker="o", facecolor="white",
                    edgecolor="black", s=90, zorder=3, label="normal-segment control" if yi == y[0] else None)
        ax_a.scatter([row["pct_anom"]], [yi], marker="o", facecolor="black",
                    edgecolor="black", s=90, zorder=3, label="anomalous segments" if yi == y[0] else None)

        ns_tag = "  (n.s.)" if row.get("anom_ns", False) else ""
        ax_a.annotate("", xy=(row["pct_anom"], yi), xytext=(row["pct_normal"], yi),
                      arrowprops=dict(arrowstyle="->", lw=1.4))
        ax_a.text(row["pct_anom"], yi + 0.18, f"{row['pct_anom']:.1f}{ns_tag}",
                  ha="center", fontsize=10, fontweight="bold")
        ax_a.text(row["pct_normal"], yi - 0.22, f"{row['pct_normal']:.1f}", ha="center", fontsize=9)
        ax_a.text(-3, yi, f"{row['dataset']}\nn = {int(row['n_anom'])} anom. /\n{int(row['n_normal'])} normal",
                  ha="right", va="center", fontsize=8.5)

    ax_a.axvline(50, color="black", linestyle="--", linewidth=1)
    ax_a.text(50, len(prec) - 0.4, "50%: no ordering", ha="center", fontsize=8)
    ax_a.set_xlim(25, 100)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Pairs in which diff crosses |z| > 3 before level (%)")
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=14)
    handles, labels = ax_a.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_a.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9, ncol=1)

    # --- Panel b: anomaly-attributable increment forest plot ---------------
    yb = np.arange(len(decomp))[::-1]
    for yi, (_, row) in zip(yb, decomp.iterrows()):
        ax_b.plot([row["ci_lo"], row["ci_hi"]], [yi, yi], color="black", linewidth=1.4)
        ax_b.scatter([row["increment"]], [yi], marker="s", color="black", s=70, zorder=3)
        ax_b.text(row["increment"], yi + 0.22, f"+{row['increment']:.1f} pp",
                  ha="center", fontsize=10, fontweight="bold")
        ax_b.text(row["increment"], yi - 0.28, f"[{row['ci_lo']:.1f}, {row['ci_hi']:.1f}]",
                  ha="center", fontsize=8)

    ax_b.axvline(0, color="black", linewidth=1)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(decomp["dataset"])
    ax_b.set_xlabel("Anomalous \u2212 normal (percentage points)")
    ax_b.set_title("Anomaly-attributable increment", fontsize=11)
    ax_b.set_xlim(-15, 60)

    fig.text(0.5, -0.02,
             "Wilson 95% intervals for each fraction; increment CI by normal approximation. Both computed from the "
             "reported fractions\nand pair counts, treating pairs as independent, so they are nominal. "
             "Ordering is diff vs. level pairs where both features cross.",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig07] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
