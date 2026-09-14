"""
Figure 1 — Channel scoping: MCC / Youden's J at (mixture=False, forgetting=True),
and the 9-channel -> 5-channel funnel (structural / sample-size / chance-level exclusions).

Source numbers: docs/dev-log/step1_signal_estimation.md, Sec. 1-11 (channel decision table)
and the 9->5 funnel summary.
Reproduces: results/figures/fig01_channel_scope.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

channels = ["CADC0872", "CADC0873", "CADC0874", "CADC0884", "CADC0886",
            "CADC0888", "CADC0890", "CADC0892", "CADC0894"]
mcc = [0.514, 0.489, 0.740, np.nan, 0.0, 0.194, 0.826, -0.090, 0.189]
youden = [0.321, 0.276, 0.693, np.nan, 0.0, 0.226, 0.909, -0.024, 0.203]
final_included = [True, True, True, False, False, True, False, False, True]
reason = ["", "", "", "n_anom=0\n(structural)", "n=3/8\n(underpowered)",
          "", "n=14\n(underpowered)", "MCC<0\n(chance-level)", ""]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})

# --- Left panel: MCC / Youden bar chart ---
ax = axes[0]
x = np.arange(len(channels))
w = 0.35
colors_mcc = ["#2c7fb8" if inc else "#bdbdbd" for inc in final_included]
colors_j = ["#41b6c4" if inc else "#e0e0e0" for inc in final_included]

ax.bar(x - w/2, mcc, width=w, label="MCC", color=colors_mcc, edgecolor="black", linewidth=0.5)
ax.bar(x + w/2, youden, width=w, label="Youden's J", color=colors_j, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("CADC0", "0") for c in channels], rotation=0)
ax.set_ylabel("Score")
ax.set_title("(a) MCC & Youden's J at locked hyperparameters\n(mixture=False, forgetting=True)", fontsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=2, frameon=False, fontsize=9.5)

for i, (m, r) in enumerate(zip(mcc, reason)):
    if r:
        ax.annotate(r, xy=(i, 0.06), ha="center", fontsize=7.2, color="#555555")

ax.text(0.02, 0.03, "Blue/teal = final 5-channel scope   Grey = excluded",
        transform=ax.transAxes, fontsize=8, va="bottom", style="italic", color="#444")
ax.set_ylim(-0.15, 1.0)

# --- Right panel: funnel diagram ---
ax2 = axes[1]
ax2.axis("off")
stages = [
    ("9 channels", "all telemetry channels", 9),
    ("6 channels", "MCC/Youden 1st-pass scope (E)", 6),
    ("5 channels", "Bootstrap CI stability (H)", 5),
    ("5 channels", "Train-only reselection (K)", 5),
    ("5 channels", "Test-only confirmation (M)", 5),
]
n_stages = len(stages)
y_top, y_bot = 0.97, 0.20
box_h = 0.09
ys = np.linspace(y_top, y_bot + box_h, n_stages)
for i, ((label, sub, n), y) in enumerate(zip(stages, ys)):
    ax2.add_patch(plt.Rectangle((0.12, y - box_h), 0.76, box_h,
                                 facecolor="#8c96c6" if i == 0 else "#2c7fb8",
                                 edgecolor="black", linewidth=0.8, alpha=0.9))
    ax2.text(0.5, y - box_h * 0.35, label, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax2.text(0.5, y - box_h - 0.028, sub, ha="center", va="top", fontsize=7.8, color="#333")
    if i < n_stages - 1:
        y_next = ys[i + 1]
        ax2.annotate("", xy=(0.5, y_next), xytext=(0.5, y - box_h - 0.045),
                      arrowprops=dict(arrowstyle="->", color="black", lw=1.1))

ax2.text(0.5, 0.02, "Final scope: 0872 / 0873 / 0874 / 0888 / 0894\n(0884, 0886, 0890, 0892 excluded)",
         ha="center", va="bottom", fontsize=9, fontweight="bold", color="#08519c")
ax2.set_title("(b) 9→5 channel scoping funnel\n(triple-validated: full / train-only / test-only)", fontsize=11)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("/home/claude/repo/results/figures/fig01_channel_scope.png", dpi=200, bbox_inches="tight")
print("saved fig01")
