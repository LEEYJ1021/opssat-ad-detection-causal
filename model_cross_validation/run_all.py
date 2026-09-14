"""
run_all.py
==========
Entry point: `python -m model_cross_validation.run_all`

Orchestrates the full 16-model cross-validation pipeline described in the
README ("Model Cross-Validation (16 independent architectures)"):

  1. Load segments + Layer-3 canonical onsets, build the tabular and
     raw-sequence datasets (representations/).
  2. Train all 10 tabular models (5-fold GroupKFold, grouped by
     channel-segment identity) and compute SHAP/permutation attribution
     (attribution.shap_permutation).
  3. Train all 6 sequence models (5-fold grouped CV) and compute
     Integrated-Gradients attribution (attribution.integrated_gradients).
  4. Combine both representations' results, compute agreement statistics
     (Kendall's W, binomial test, family summary) and bootstrap stability
     (agreement_statistics.py).
  5. Write every artifact listed under results/model_cross_validation/ in
     the top-level README, plus a plain-text final report.

MLP_shallow (models/shallow_mlp.py) is intentionally NOT included in the
default zoo -- see that module's docstring. The pipeline therefore always
reports "16 models" (10 tabular + 6 sequence), matching the paper's
headline numbers, not "17".

This module is a straight orchestration refactor of the original monolithic
script; no analytical logic differs from what is described in
docs/dev-log/step_ai_model_cross_validation.md.
"""

from pathlib import Path
import argparse
import random
import warnings

import numpy as np
import pandas as pd

from model_cross_validation.representations import (
    FEATURES, RANDOM_STATE, FINAL_CHANNELS,
    build_tabular_dataset, save_tabular_dataset,
    build_sequence_dataset,
)
from model_cross_validation.models import build_tabular_model_zoo, build_sequence_model_zoo
from model_cross_validation.attribution import compute_importance, compute_saliency
from model_cross_validation import agreement_statistics as agree

warnings.filterwarnings("ignore")

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False

EXPLAIN_SAMPLE_SIZE = 400
N_BOOT_TABULAR = 200
N_FOLDS = 5

# Layer-4 BOCPD/tau*-Youden reference performance, for the leaderboard
# comparison row (not commensurable with batch-classifier AUC -- see README).
REFERENCE_BOCPD_PERFORMANCE = {
    "CADC0872": {"recall": 0.587786, "fa_rate": 0.012077},
    "CADC0873": {"recall": 0.663462, "fa_rate": 0.004132},
    "CADC0874": {"recall": 0.840580, "fa_rate": 0.064000},
    "CADC0888": {"recall": 0.473684, "fa_rate": 0.296053},
    "CADC0894": {"recall": 0.857143, "fa_rate": 0.528455},
}


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# Data loading
# =============================================================================

def load_inputs(base_dir: Path):
    """Load segments.csv, Layer-3 MC draws, and the reliability-tiered
    segment summary; reconstruct the canonical (median) onset index per
    reliable anomalous segment. Reused verbatim from Layer 3 -- no
    recomputation, per the README's provenance rule."""
    seg = pd.read_csv(base_dir / "segments.csv", parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

    out_dir = base_dir / "causal_pipeline_outputs"
    draws = pd.read_csv(out_dir / "stage2_mc_draws_raw.csv")
    seg_corrected = pd.read_csv(out_dir / "stage2_mc_segment_summary_corrected.csv")

    reliable_ids = seg_corrected.loc[
        seg_corrected["reliability_tier"].isin(["high", "medium"])
        & seg_corrected["channel"].isin(FINAL_CHANNELS),
        ["channel", "segment"],
    ]
    canonical_onset = (
        draws.merge(reliable_ids, on=["channel", "segment"], how="inner")
        .groupby(["channel", "segment"])["onset_idx_sampled"]
        .median().round().astype(int).rename("canonical_onset_idx").reset_index()
    )
    print(f"Segments at reliability high+medium: {len(canonical_onset)}")
    return seg, canonical_onset, out_dir


# =============================================================================
# Tabular models: train + attribute
# =============================================================================

def run_tabular_models(tab_df: pd.DataFrame, groups: np.ndarray):
    if not HAVE_SKLEARN or len(tab_df) <= 50:
        print("[skip] scikit-learn unavailable or insufficient tabular data")
        return pd.DataFrame(), pd.DataFrame()

    X_all = tab_df[FEATURES].values.astype(float)
    y_all = tab_df["label"].values.astype(int)

    gkf = GroupKFold(n_splits=N_FOLDS)
    model_zoo = build_tabular_model_zoo()
    print(f"Registered tabular models: {len(model_zoo)} -> {[m[0] for m in model_zoo]}")

    importance_rows, perf_rows = [], []

    for name, family, estimator_template, shap_kind in model_zoo:
        print(f"\n[{name}] ({family}) training...")
        oof_prob = np.full(len(y_all), np.nan)
        fold_importances = []

        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, groups)):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

            model = estimator_template.__class__(**estimator_template.get_params())
            try:
                model.fit(X_tr_s, y_tr)
            except Exception as e:
                print(f"    fold {fold_i} training failed: {e}")
                continue

            prob = model.predict_proba(X_te_s)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_te_s)
            oof_prob[te_idx] = prob

            shap_imp = compute_importance(
                model, shap_kind, X_tr_s, X_te_s, y_te,
                explain_sample_size=EXPLAIN_SAMPLE_SIZE, random_state=RANDOM_STATE,
            )
            fold_importances.append(shap_imp)

        if not fold_importances:
            print(f"    [{name}] all folds failed -- skipping")
            continue

        mean_importance = np.mean(fold_importances, axis=0)
        for feat, imp in zip(FEATURES, mean_importance):
            importance_rows.append({"model": name, "family": family, "feature": feat, "importance_share": float(imp)})

        valid = ~np.isnan(oof_prob)
        if valid.sum() > 10 and len(np.unique(y_all[valid])) == 2:
            auc = roc_auc_score(y_all[valid], oof_prob[valid])
            fpr, tpr, _ = roc_curve(y_all[valid], oof_prob[valid])
            best_i = int(np.argmax(tpr - fpr))
            perf_rows.append({
                "model": name, "family": family, "auc": float(auc),
                "recall_at_youden": float(tpr[best_i]), "fa_rate_at_youden": float(fpr[best_i]),
            })
            print(f"    AUC={auc:.3f}  importance(level/diff/diff2)="
                  f"{mean_importance[0]:.3f}/{mean_importance[1]:.3f}/{mean_importance[2]:.3f}")

    return pd.DataFrame(importance_rows), pd.DataFrame(perf_rows)


# =============================================================================
# Sequence models: train + attribute
# =============================================================================

def run_sequence_models(seq_rows: list, seq_groups: np.ndarray):
    if not HAVE_TORCH or len(seq_rows) <= 50:
        print("[skip] torch unavailable or insufficient sequence data")
        return pd.DataFrame(), pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] torch device: {device}")
    torch.manual_seed(RANDOM_STATE)

    class WindowDataset(Dataset):
        def __init__(self, rows):
            self.X = np.stack([r["window"] for r in rows]).astype(np.float32)
            self.y = np.array([r["label"] for r in rows], dtype=np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def eval_model(model, loader, criterion):
        model.eval()
        total_loss, probs, ys = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.cpu().numpy())
        return total_loss / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)

    rng = np.random.default_rng(RANDOM_STATE)
    unique_groups = np.unique(seq_groups)
    fold_assign = {g: i % N_FOLDS for i, g in enumerate(rng.permutation(unique_groups))}
    fold_of_row = np.array([fold_assign[g] for g in seq_groups])

    model_zoo = build_sequence_model_zoo()
    print(f"Registered sequence models: {len(model_zoo)} -> {[m[0] for m in model_zoo]}")

    y_seq = np.array([r["label"] for r in seq_rows], dtype=np.float32)
    n_pos, n_neg = int(y_seq.sum()), int((1 - y_seq).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    importance_rows, perf_rows = [], []

    for name, family, ModelClass in model_zoo:
        print(f"\n[{name}] ({family}) training...")
        fold_saliency = []
        oof_prob = np.full(len(y_seq), np.nan)

        for fold_i in range(N_FOLDS):
            te_idx = np.where(fold_of_row == fold_i)[0]
            tr_idx = np.where(fold_of_row != fold_i)[0]
            if len(te_idx) < 5 or len(tr_idx) < 20:
                continue

            train_ds = WindowDataset([seq_rows[i] for i in tr_idx])
            test_ds = WindowDataset([seq_rows[i] for i in te_idx])
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

            model = ModelClass().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val_loss, patience, patience_ctr, best_state = np.inf, 6, 0, None
            for _ in range(60):
                train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, _, _ = eval_model(model, test_loader, criterion)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss, patience_ctr = val_loss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)

            _, probs, _ = eval_model(model, test_loader, criterion)
            oof_prob[te_idx] = probs

            explain_n = min(EXPLAIN_SAMPLE_SIZE, len(test_ds))
            explain_idx = np.random.choice(len(test_ds), size=explain_n, replace=False)
            X_explain = torch.from_numpy(np.stack([test_ds.X[i] for i in explain_idx])).to(device)

            chan_importance = compute_saliency(model, X_explain)
            fold_saliency.append(chan_importance)

        if not fold_saliency:
            print(f"    [{name}] all folds failed -- skipping")
            continue

        mean_saliency = np.mean(fold_saliency, axis=0)
        for feat, imp in zip(FEATURES, mean_saliency):
            importance_rows.append({"model": name, "family": family, "feature": feat, "importance_share": float(imp)})

        valid = ~np.isnan(oof_prob)
        if valid.sum() > 10 and len(np.unique(y_seq[valid])) == 2:
            auc = roc_auc_score(y_seq[valid], oof_prob[valid])
            fpr, tpr, _ = roc_curve(y_seq[valid], oof_prob[valid])
            best_i = int(np.argmax(tpr - fpr))
            perf_rows.append({
                "model": name, "family": family, "auc": float(auc),
                "recall_at_youden": float(tpr[best_i]), "fa_rate_at_youden": float(fpr[best_i]),
            })
            print(f"    AUC={auc:.3f}  saliency(level/diff/diff2)="
                  f"{mean_saliency[0]:.3f}/{mean_saliency[1]:.3f}/{mean_saliency[2]:.3f}")

    return pd.DataFrame(importance_rows), pd.DataFrame(perf_rows)


# =============================================================================
# Main
# =============================================================================

def main(base_dir: str, out_dir: str = None):
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir else base_dir / "causal_pipeline_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    section("1. Load data and build the shared tabular + sequence label sets")
    seg, canonical_onset, _ = load_inputs(base_dir)

    tab_bundle = build_tabular_dataset(seg, canonical_onset)
    print(f"\nTabular dataset: {len(tab_bundle.tab_df)} rows "
          f"(positive {int((tab_bundle.tab_df['label'] == 1).sum())} / "
          f"negative {int((tab_bundle.tab_df['label'] == 0).sum())})")
    save_tabular_dataset(tab_bundle, out_dir / "modelclass_dataset_tabular.csv")

    seq_bundle = build_sequence_dataset(seg, canonical_onset, tab_bundle.onset_ratio_pool_by_channel)
    print(f"Sequence dataset: {len(seq_bundle.rows)} rows")

    section("2. [Tabular] train + attribute up to 10 independent models")
    tab_importance_df, tab_perf_df = run_tabular_models(tab_bundle.tab_df, tab_bundle.groups)

    section("3. [Sequence] train + attribute up to 6 independent architectures")
    seq_importance_df, seq_perf_df = run_sequence_models(seq_bundle.rows, seq_bundle.groups)

    section("4. Combine results + performance leaderboard (BOCPD reference row)")
    importance_df = pd.concat([tab_importance_df, seq_importance_df], ignore_index=True)
    importance_df.to_csv(out_dir / "modelclass_feature_importance.csv", index=False)
    n_models = importance_df["model"].nunique() if len(importance_df) else 0
    print(f"[saved] modelclass_feature_importance.csv ({n_models} models)")

    perf_df = pd.concat([tab_perf_df, seq_perf_df], ignore_index=True)
    bocpd_recall = np.mean([v["recall"] for v in REFERENCE_BOCPD_PERFORMANCE.values()])
    bocpd_fa = np.mean([v["fa_rate"] for v in REFERENCE_BOCPD_PERFORMANCE.values()])
    perf_df = pd.concat([perf_df, pd.DataFrame([{
        "model": "BOCPD_tau_Youden(Stage-4 reference)", "family": "Reference(probabilistic,online)",
        "auc": np.nan, "recall_at_youden": bocpd_recall, "fa_rate_at_youden": bocpd_fa,
    }])], ignore_index=True)
    perf_df.sort_values("auc", ascending=False).to_csv(
        out_dir / "modelclass_performance_leaderboard.csv", index=False
    )
    print(perf_df.sort_values("auc", ascending=False).to_string(index=False))

    section("5. Cross-model agreement statistics")
    if len(importance_df) > 0:
        pivot, agreement_df, fam_df = agree.build_agreement_summary(importance_df)
        print(agreement_df.to_string(index=False))
        print("\nFamily-level mean importance:")
        print(fam_df.to_string(index=False))
    else:
        print("[skip] no importance data")
        pivot, agreement_df, fam_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    section(f"6. Tabular bootstrap stability (N_BOOT_TABULAR={N_BOOT_TABULAR})")
    if HAVE_SKLEARN and len(tab_bundle.tab_df) > 50:
        X_all = tab_bundle.tab_df[FEATURES].values.astype(float)
        y_all = tab_bundle.tab_df["label"].values.astype(int)
        boot_df = agree.bootstrap_stability(X_all, y_all, n_boot=N_BOOT_TABULAR, random_state=RANDOM_STATE)
        print(boot_df.to_string(index=False))
    else:
        boot_df = pd.DataFrame()

    agree.save_all(pivot, agreement_df, fam_df, boot_df, out_dir)

    section("Done")
    print(f"All artifacts saved under {out_dir}")
    return {
        "importance_df": importance_df, "perf_df": perf_df,
        "agreement_df": agreement_df, "family_df": fam_df, "bootstrap_df": boot_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="16-model cross-validation of the level/diff/diff2 causal signature")
    parser.add_argument("--base-dir", type=str, required=True,
                         help="Directory containing segments.csv and causal_pipeline_outputs/ (Layer-3 artifacts)")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Output directory (default: <base-dir>/causal_pipeline_outputs)")
    args = parser.parse_args()
    main(args.base_dir, args.out_dir)
