"""
fig01_channel_scope.py

Reproduces Figure 1 (channel scoping: MCC/Youden's J per channel, and the
9 -> 5 channel-scoping funnel) from:
    results/layer2/channel_scope.json
    results/layer2/bootstrap_ci.csv

If those artifacts are not present (e.g. running this script standalone,
before `python -m layer2_anomaly_detection.run` has been executed), the
script falls back to the exact values reported in Table 1 of the README,
so the figure is always reproducible offline.

Output: results/figures/fig01_channel_scope.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig01_channel_scope.png")

CHANNEL_SCOPE_JSON = os.path.join(RESULTS, "layer2", "channel_scope.json")
BOOTSTRAP_CI_CSV = os.path.join(RESULTS, "layer2", "bootstrap_ci.csv")

# --- Table 1 fallback (channel scoping decision table) ---------------------
TABLE1 = pd.DataFrame(
    [
        ("CADC0872", "Magn.", 546, 24.0, 131, 415, 0.514, 0.321, "incl."),
        ("CADC0873", "Magn.", 593, 17.7, 105, 488, 0.489, 0.276, "incl."),
        ("CADC0874", "Magn.", 194, 35.6, 69, 125, 0.740, 0.693, "incl."),
        ("CADC0884", "Phot.", 158, 0.0, 0, 158, np.nan, np.nan, "excl. (S)"),
        ("CADC0886", "Phot.", 11, 27.3, 3, 8, 0.000, 0.000, "excl. (U)"),
        ("CADC0888", "Phot.", 252, 23.8, 60, 192, 0.194, 0.226, "incl."),
        ("CADC0890", "Phot.", 14, 78.6, 11, 3, 0.826, 0.909, "excl. (U)"),
        ("CADC0892", "Phot.", 211, 16.1, 34, 177, -0.090, -0.024, "excl. (C)"),
        ("CADC0894", "Phot.", 144, 14.6, 21, 123, 0.189, 0.203, "incl."),
    ],
    columns=["channel", "sensor", "n_segments", "anomaly_pct", "n_anom",
             "n_nom", "mcc", "youden_j", "status"],
)


def load_channel_table():
    """Load MCC / Youden's J per channel, preferring pipeline artifacts."""
    if os.path.exists(CHANNEL_SCOPE_JSON):
        with open(CHANNEL_SCOPE_JSON) as f:
            scope = json.load(f)
        df = pd.DataFrame(scope["channels"])
        return df
    print(f"[fig01] {CHANNEL_SCOPE_JSON} not found; using Table 1 fallback.")
    return TABLE1


def build_funnel(df):
    """9 -> structural/power screen -> chance screen -> train -> test."""
    included = df[df["status"] == "incl."]["channel"].tolist()
    n_total = len(df)
    n_structural_power = len(df[df["status"].isin(["incl.", "excl. (C)"])])
    n_chance = len(included)
    return [
        ("All telemetry\nchannels", n_total),
        ("Structural +\npower screen", n_structural_power),
        ("Chance-level\nscreen (MCC > 0)", n_chance),
        ("Train-only\nreselection", n_chance),
        ("Test-only\nconfirmation", n_chance),
    ], included


def main():
    df = load_channel_table()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.6, 1]})

    # --- Panel a: MCC / Youden's J bars -----------------------------------
    x = np.arange(len(df))
    width = 0.35
    included_mask = df["status"] == "incl."

    for i, (_, row) in enumerate(df.iterrows()):
        edge_style = dict(edgecolor="black", linewidth=1.2)
        if not included_mask.iloc[i]:
            edge_style.update(linestyle=(0, (4, 2)))  # dashed = excluded
        mcc_color = "#1f3b4d" if included_mask.iloc[i] else "white"
        j_color = "#7fa8b8" if included_mask.iloc[i] else "white"
        if not np.isnan(row["mcc"]):
            ax_a.bar(i - width / 2, row["mcc"], width, color=mcc_color, hatch=None, **edge_style)
            ax_a.text(i - width / 2, row["mcc"] + (0.02 if row["mcc"] >= 0 else -0.05),
                      f"{row['mcc']:.2f}", ha="center", va="bottom" if row["mcc"] >= 0 else "top", fontsize=8)
        else:
            ax_a.text(i, 0.05, "no anom.\nsegments", ha="center", va="bottom", fontsize=7, style="italic")
        if not np.isnan(row["youden_j"]):
            ax_a.bar(i + width / 2, row["youden_j"], width, color=j_color, hatch="//" if included_mask.iloc[i] else None, **edge_style)
            ax_a.text(i + width / 2, row["youden_j"] + (0.02 if row["youden_j"] >= 0 else -0.05),
                      f"{row['youden_j']:.2f}", ha="center", va="bottom" if row["youden_j"] >= 0 else "top", fontsize=8)

    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_xticks(x)
    labels = [c.replace("CADC0", "") for c in df["channel"]]
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Score at locked hyperparameters")
    ax_a.set_ylim(-0.15, 1.0)
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=14)

    # annotate sensor / anom-nom / status rows beneath the axis
    for i, (_, row) in enumerate(df.iterrows()):
        ax_a.annotate(row["sensor"], (i, -0.20), xycoords=("data", "axes fraction"),
                      ha="center", fontsize=7, style="italic", annotation_clip=False)
        anom_nom = f"{int(row['n_anom'])}/{int(row['n_nom'])}" if not np.isnan(row["n_anom"]) else "n/a"
        ax_a.annotate(anom_nom, (i, -0.27), xycoords=("data", "axes fraction"),
                      ha="center", fontsize=7, annotation_clip=False)
        ax_a.annotate(row["status"], (i, -0.34), xycoords=("data", "axes fraction"),
                      ha="center", fontsize=7, fontweight="bold" if row["status"] == "incl." else "normal",
                      annotation_clip=False)

    from matplotlib.patches import Patch
    ax_a.legend(handles=[
        Patch(facecolor="#1f3b4d", edgecolor="black", label="MCC"),
        Patch(facecolor="#7fa8b8", edgecolor="black", hatch="//", label="Youden's J"),
        Patch(facecolor="white", edgecolor="black", linestyle=(0, (4, 2)), label="excluded (dashed outline)"),
    ], loc="upper right", fontsize=8, framealpha=0.9)

    # --- Panel b: 9 -> 5 funnel ---------------------------------------------
    stages, included = build_funnel(df)
    n_stages = len(stages)
    box_h = 0.7
    ys = np.linspace(n_stages - 1, 0, n_stages)
    for i, ((label, n), y) in enumerate(zip(stages, ys)):
        color = "0.85" if i == 0 else ("0.1" if i == n_stages - 1 else "0.2")
        textcolor = "black" if i == 0 else "white"
        ax_b.add_patch(plt.Rectangle((0, y - box_h / 2), 3, box_h, facecolor=color, edgecolor="black"))
        ax_b.text(1.5, y, str(n), ha="center", va="center", fontsize=16, fontweight="bold", color=textcolor)
        ax_b.text(3.3, y, label, ha="left", va="center", fontsize=10)
        if i < n_stages - 1:
            ax_b.annotate("", xy=(1.5, y - box_h / 2 - 0.15), xytext=(1.5, y - box_h / 2),
                          arrowprops=dict(arrowstyle="->", lw=1.2))

    ax_b.text(1.5, ys[0] + 0.55, "Scope identical across full, train-only\nand test-only re-derivations",
              ha="center", fontsize=10, fontweight="bold")
    ax_b.text(1.5, ys[-1] - 0.7,
              "Final scope:\n" + ", ".join(c.replace("CADC0", "") for c in included),
              ha="center", fontsize=10, fontweight="bold")
    ax_b.set_xlim(-0.5, 8)
    ax_b.set_ylim(-1.3, n_stages)
    ax_b.axis("off")
    ax_b.set_title("b  Scope identical across full, train-only\nand test-only re-derivations",
                    loc="left", fontweight="bold", fontsize=12)

    fig.text(0.02, 0.01,
             "S structural (no anomalous segments)   U underpowered (<5 anomalous or <5 nominal)   "
             "C chance-level (MCC \u2264 0)", fontsize=8, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig01] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
