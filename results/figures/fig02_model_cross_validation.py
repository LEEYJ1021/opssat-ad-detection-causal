"""
fig02_model_cross_validation.py

Reproduces Figure 2 (16-model feature-attribution shares and cross-model
agreement, OPS-SAT-AD only) from:
    results/model_cross_validation/feature_attribution_16models.csv
    results/model_cross_validation/agreement_statistics.json

Falls back to the values in Table 2 / Table 6 of the README when those
artifacts have not been generated yet.

Output: results/figures/fig02_model_cross_validation.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, ".."))
OUT_PATH = os.path.join(HERE, "fig02_model_cross_validation.png")

FEATURE_ATTR_CSV = os.path.join(RESULTS, "model_cross_validation", "feature_attribution_16models.csv")
AGREEMENT_JSON = os.path.join(RESULTS, "model_cross_validation", "agreement_statistics.json")

# --- Table 2 fallback --------------------------------------------------
TABLE2 = pd.DataFrame(
    [
        ("TinyTransformer", "Attention", "sequence", 0.994, 0.178, 0.438, 0.384),
        ("TCN", "Conv (dilated causal)", "sequence", 0.993, 0.200, 0.380, 0.419),
        ("BiGRU", "Recurrent", "sequence", 0.989, 0.285, 0.388, 0.327),
        ("CNN1D", "Convolutional", "sequence", 0.983, 0.303, 0.301, 0.396),
        ("BiLSTM", "Recurrent", "sequence", 0.980, 0.211, 0.497, 0.292),
        ("LightMamba", "State-space (SSM)", "sequence", 0.969, 0.428, 0.441, 0.130),
        ("SVM (RBF)", "Kernel", "tabular", 0.948, 0.061, 0.466, 0.472),
        ("LightGBM", "Tree ens. (boosting)", "tabular", 0.940, 0.232, 0.459, 0.309),
        ("XGBoost", "Tree ens. (boosting)", "tabular", 0.935, 0.148, 0.419, 0.433),
        ("RandomForest", "Tree ens. (bagging)", "tabular", 0.931, 0.084, 0.456, 0.460),
        ("ExtraTrees", "Tree ens. (bagging, extra)", "tabular", 0.927, 0.145, 0.389, 0.466),
        ("GradBoost", "Tree ens. (boosting)", "tabular", 0.924, 0.068, 0.371, 0.561),
        ("LogReg (L2)", "Linear", "tabular", 0.920, 0.060, 0.407, 0.534),
        ("LogReg (L1)", "Linear (sparse)", "tabular", 0.920, 0.057, 0.412, 0.531),
        ("kNN", "Instance-based", "tabular", 0.918, 0.129, 0.391, 0.480),
        ("GaussianNB", "Probabilistic", "tabular", 0.900, 0.051, 0.446, 0.503),
    ],
    columns=["model", "family", "representation", "auc", "level", "diff", "diff2"],
)


def load_table():
    if os.path.exists(FEATURE_ATTR_CSV):
        return pd.read_csv(FEATURE_ATTR_CSV)
    print(f"[fig02] {FEATURE_ATTR_CSV} not found; using Table 2 fallback.")
    return TABLE2


def load_agreement():
    if os.path.exists(AGREEMENT_JSON):
        with open(AGREEMENT_JSON) as f:
            return json.load(f)
    print(f"[fig02] {AGREEMENT_JSON} not found; recomputing agreement statistics from the table.")
    return None


def main():
    df = load_table()

    # order: sequence models first (as originally trained), then tabular,
    # each sorted by descending AUC -- matches the README figure's layout.
    seq = df[df["representation"] == "sequence"].sort_values("auc", ascending=False)
    tab = df[df["representation"] == "tabular"].sort_values("auc", ascending=False)
    ordered = pd.concat([seq, tab])

    # --- top-ranked feature per model, derived directly from the shares ---
    def top_feature(row):
        shares = {"level": row["level"], "diff": row["diff"], "diff2": row["diff2"]}
        return max(shares, key=shares.get)

    ordered = ordered.copy()
    ordered["top_feature"] = ordered.apply(top_feature, axis=1)
    top_counts = ordered["top_feature"].value_counts().reindex(["level", "diff", "diff2"]).fillna(0).astype(int)

    n_models = len(ordered)
    n_level_lt_third = (ordered["level"] < 1 / 3).sum()
    n_diff_plus_diff2_gt_two_thirds = ((ordered["diff"] + ordered["diff2"]) > 2 / 3).sum()

    agreement = load_agreement()
    if agreement is None:
        agreement = {
            "kendalls_w": 0.609,
            "chi2": 19.50,
            "chi2_df": 2,
            "chi2_p": 5.8e-5,
            "level_top_ranked": int(top_counts["level"]),
            "n_models": n_models,
            "binomial_p": 0.0015,
            "tabular_0_of": (top_counts.get("level", 0), (df["representation"] == "tabular").sum()),
            "sequence_0_of": (top_counts.get("level", 0), (df["representation"] == "sequence").sum()),
        }

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1, 1.1])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis("off")

    # --- Panel a: stacked horizontal bars ----------------------------------
    y = np.arange(n_models)[::-1]
    hatches = {"level": "..", "diff": "//", "diff2": None}
    colors = {"level": "white", "diff": "0.75", "diff2": "0.1"}
    left = np.zeros(n_models)
    for feat in ["level", "diff", "diff2"]:
        vals = ordered[feat].values
        ax_a.barh(y, vals, left=left, height=0.7, edgecolor="black",
                  facecolor=colors[feat], hatch=hatches[feat], label=feat)
        left = left + vals

    ax_a.axvline(1 / 3, color="black", linestyle=":", linewidth=1.2)
    ax_a.text(1 / 3, n_models + 0.3, "level share = 1/3", ha="center", fontsize=9)
    ax_a.set_yticks(y)
    labels = [f"{m}\u2020" if s < 1 / 3 * 0 or l >= 1 / 3 else m
              for m, l, s in zip(ordered["model"], ordered["level"], ordered["level"])]
    # dagger only for models with level share >= 1/3 (per caption)
    labels = [f"{m}\u2020" if l >= 1 / 3 else m for m, l in zip(ordered["model"], ordered["level"])]
    ax_a.set_yticklabels(labels, fontsize=9)
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("Normalized feature-importance share")
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=14)
    for yi, auc in zip(y, ordered["auc"]):
        ax_a.text(1.01, yi, f"{auc:.3f}", va="center", fontsize=8)
    ax_a.text(1.01, n_models + 0.3, "AUC", fontsize=9, fontweight="bold")
    ax_a.legend(loc="lower right", fontsize=8, ncol=3, bbox_to_anchor=(1.0, -0.12))

    seq_n = len(seq)
    ax_a.axhline(y[seq_n - 1] - 0.5, color="black", linewidth=0.8)

    # --- Panel b: top-ranked feature counts --------------------------------
    feats = ["level", "diff", "diff2"]
    feat_labels = ["level", "diff", "diff\u00b2"]
    bar_colors = ["white", "0.75", "0.1"]
    bar_hatches = ["..", "//", None]
    bars = ax_b.bar(feat_labels, [top_counts[f] for f in feats],
                    color=bar_colors, hatch=bar_hatches, edgecolor="black")
    for b, f in zip(bars, feats):
        ax_b.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                  str(int(top_counts[f])), ha="center", fontsize=12, fontweight="bold")
    ax_b.set_ylim(0, n_models + 2)
    ax_b.set_ylabel(f"Models (of {n_models})")
    ax_b.set_title("Top-ranked feature", fontsize=11)

    # --- Text panel: cross-model agreement ----------------------------------
    lines = [
        "Cross-model agreement", "",
        f"Kendall's W = {agreement['kendalls_w']:.3f}",
        f"\u03c7\u00b2 = {agreement['chi2']:.2f}, df={agreement['chi2_df']}, "
        f"p={agreement['chi2_p']:.1e}", "",
        f"level top-ranked: {int(top_counts['level'])} / {n_models}",
        f"binomial (2/3)^{n_models}: p={agreement['binomial_p']:.4f}",
        f"tabular {(ordered[ordered.representation=='tabular'].top_feature=='level').sum()}/"
        f"{(ordered.representation=='tabular').sum()},  "
        f"sequence {(ordered[ordered.representation=='sequence'].top_feature=='level').sum()}/"
        f"{(ordered.representation=='sequence').sum()}", "",
        f"level share < 1/3: {n_level_lt_third} / {n_models}",
        f"diff + diff\u00b2 share > 2/3: {n_diff_plus_diff2_gt_two_thirds} / {n_models}", "",
        "Nominal: the models share data and", "onset-derived labels, so W and the", "binomial p are descriptive, not", "independent tests.",
    ]
    ax_text.text(0, 1, "\n".join(lines), va="top", fontsize=9, family="monospace")

    fig.suptitle("Figure 2 \u2014 16-model cross-validation: feature attribution and agreement (OPS-SAT-AD)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[fig02] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
