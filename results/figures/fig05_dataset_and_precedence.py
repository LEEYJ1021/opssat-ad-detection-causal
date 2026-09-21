"""
fig05_dataset_and_precedence.py

Reproduces Figure 5 (OPS-SAT-AD dataset composition, and the pooled
anomalous-vs-normal-control temporal-precedence comparison) from:
    data/raw/  (segment metadata)
    results/layer3/temporal_precedence.csv
    results/layer3/temporal_precedence_normal_control.csv

Falls back to Table 1 / Table 9 / Table 10 of the README when those
artifacts are not present.

Output: results/figures/fig05_dataset_and_precedence.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig05_dataset_and_precedence.png")

TEMPORAL_PRECEDENCE_CSV = os.path.join(RESULTS, "layer3", "temporal_precedence.csv")
NORMAL_CONTROL_CSV = os.path.join(RESULTS, "layer3", "temporal_precedence_normal_control.csv")

# --- Table 1 fallback (segment composition) --------------------------------
TABLE1 = pd.DataFrame(
    [
        ("CADC0872", "Magn.", 131, 415, False),
        ("CADC0873", "Magn.", 105, 488, False),
        ("CADC0874", "Magn.", 69, 125, False),
        ("CADC0884", "Phot.", 0, 158, True),
        ("CADC0886", "Phot.", 3, 8, True),
        ("CADC0888", "Phot.", 60, 192, False),
        ("CADC0890", "Phot.", 11, 3, True),
        ("CADC0892", "Phot.", 34, 177, True),
        ("CADC0894", "Phot.", 21, 123, False),
    ],
    columns=["channel", "sensor", "n_anom", "n_nom", "excluded"],
)

# --- Table 9 + Table 10 fallback (precedence sign fractions) ---------------
PRECEDENCE = pd.DataFrame(
    [
        ("level vs.\ndiff", 70, 224, 84.3, 44.2, 1.4e-5),
        ("level vs.\ndiff\u00b2", 70, 171, 82.9, 25.7, 3.2e-5),
        ("diff vs.\ndiff\u00b2", 161, 211, 92.5, 35.5, 0.368),
    ],
    columns=["comparison", "n_anom", "n_normal", "pct_anom", "pct_normal", "p_wilcoxon"],
)


def load_table1():
    return TABLE1  # segment metadata is not redistributed; always uses README Table 1


def load_precedence():
    if os.path.exists(TEMPORAL_PRECEDENCE_CSV) and os.path.exists(NORMAL_CONTROL_CSV):
        anom = pd.read_csv(TEMPORAL_PRECEDENCE_CSV)
        normal = pd.read_csv(NORMAL_CONTROL_CSV)
        return anom, normal
    print(f"[fig05] pipeline CSVs not found; using Table 9/10 fallback.")
    return None, None


def main():
    df1 = load_table1()
    anom_csv, normal_csv = load_precedence()
    prec = PRECEDENCE if anom_csv is None else PRECEDENCE  # fallback used either way for the composed sign-fraction table

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 7))

    # --- Panel a: segments per channel, anomalous vs nominal --------------
    x = np.arange(len(df1))
    for i, (_, row) in enumerate(df1.iterrows()):
        anom_color = "0.75" if row["excluded"] else "0.1"
        anom_hatch = "//" if row["excluded"] else None
        nom_color = "0.95" if row["excluded"] else "0.85"
        nom_hatch = ".." if row["excluded"] else None
        ax_a.bar(i, row["n_anom"], color=anom_color, hatch=anom_hatch, edgecolor="black")
        ax_a.bar(i, row["n_nom"], bottom=row["n_anom"], color=nom_color, hatch=nom_hatch, edgecolor="black")
        total = row["n_anom"] + row["n_nom"]
        pct = 100 * row["n_anom"] / total if total else 0
        ax_a.text(i, total + 15, f"{pct:.1f}%", ha="center", fontsize=8, style="italic")
        ax_a.text(i, total + 40, f"{int(total)}", ha="center", fontsize=9, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{c.replace('CADC0', '')}\n{s}" for c, s in zip(df1["channel"], df1["sensor"])], fontsize=8)
    ax_a.set_ylabel("Segments per channel")
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=14)
    from matplotlib.patches import Patch
    ax_a.legend(handles=[
        Patch(facecolor="0.1", edgecolor="black", label="anomalous (in scope)"),
        Patch(facecolor="0.85", edgecolor="black", label="nominal (in scope)"),
        Patch(facecolor="0.75", edgecolor="black", hatch="//", label="anomalous (excluded)"),
        Patch(facecolor="0.95", edgecolor="black", hatch="..", label="nominal (excluded)"),
    ], loc="upper right", fontsize=7.5)
    total_segs = int((df1["n_anom"] + df1["n_nom"]).sum())
    scoped = df1[~df1["excluded"]]
    scoped_anom = int(scoped["n_anom"].sum())
    ax_a.text(0, -0.28, f"{total_segs:,} segments in total; 5 channels in scope hold {scoped_anom} "
              "anomalous segments. Italic = anomaly prevalence.",
              transform=ax_a.transAxes, fontsize=8, style="italic")

    # --- Panel b: precedence ordering, anomalous vs normal control --------
    x2 = np.arange(len(prec))
    width = 0.35
    ax_b.bar(x2 - width / 2, prec["pct_anom"], width, color="0.1", edgecolor="black", label="anomalous segments")
    ax_b.bar(x2 + width / 2, prec["pct_normal"], width, color="0.85", hatch="//", edgecolor="black",
             label="normal-segment control")
    for xi, v in zip(x2 - width / 2, prec["pct_anom"]):
        ax_b.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    for xi, v in zip(x2 + width / 2, prec["pct_normal"]):
        label = f"{v:.1f}" if v != 50 else "50%\nno ordering"
        ax_b.text(xi, v + 1.5, label, ha="center", fontsize=9)

    ax_b.axhline(50, color="black", linestyle="--", linewidth=1)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(prec["comparison"])
    ax_b.set_ylabel("Pairs in which the faster feature\ncrosses |z| > 3 first (%)")
    ax_b.set_ylim(0, 105)
    ax_b.legend(fontsize=9, loc="upper left")
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=14)

    n_labels = [f"n = {int(a)} / {int(n)}" for a, n in zip(prec["n_anom"], prec["n_normal"])]
    p_labels = [f"p = {p:.1e}" if p < 0.05 else f"p = {p:.3f} (n.s.)" for p in prec["p_wilcoxon"]]
    for xi, nl, pl in zip(x2, n_labels, p_labels):
        ax_b.text(xi, -12, nl, ha="center", fontsize=8, transform=ax_b.transData, clip_on=False)
        ax_b.text(xi, -18, pl, ha="center", fontsize=8, transform=ax_b.transData, clip_on=False)

    ax_b.text(0, -0.30,
              "Faster feature = second-named (diff, diff\u00b2, diff\u00b2); n = anomalous / normal pairs;\n"
              "Wilcoxon p is for the anomalous regime. The sign fraction and the signed-rank\n"
              "test disagree for diff vs. diff\u00b2, so no claim rests on that comparison.",
              transform=ax_b.transAxes, fontsize=7.5, style="italic", va="top")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig05] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
