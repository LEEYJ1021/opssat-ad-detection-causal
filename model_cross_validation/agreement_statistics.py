"""
agreement_statistics.py
=========================
Cross-model agreement statistics underlying Fig. 2b / Table 3 / Table 6:

- Kendall's coefficient of concordance (W) across all N models' within-model
  rank of {level, diff, diff2}, plus its chi-square approximation.
- Binomial test on "how many of N models do NOT rank level as their single
  top feature" against a chance rate of 2/3 (i.e. under a uniform random
  rank, level is top 1/3 of the time, so "not top" has base rate 2/3).
- Family-level aggregation (mean importance per inductive-bias family, and
  whether diff+diff2 beats level for that family).
- Bootstrap stability: repeatedly refit two representative, architecturally
  distinct tabular models (RandomForest, LogisticRegression) on bootstrap
  resamples and record how often diff+diff2 beats level.

All functions operate on the long-format importance_df produced by
run_all.py (columns: model, family, feature, importance_share), matching
modelclass_feature_importance.csv in results/model_cross_validation/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False

from model_cross_validation.representations.tabular_summary_features import FEATURES, RANDOM_STATE


def pivot_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Long (model, family, feature, importance_share) -> wide
    (model, family, level, diff, diff2), one row per model."""
    pivot = importance_df.pivot_table(
        index=["model", "family"], columns="feature", values="importance_share"
    ).reset_index()
    return pivot.reindex(columns=["model", "family"] + FEATURES)


def kendalls_w(pivot: pd.DataFrame):
    """Kendall's coefficient of concordance across models' within-model
    ranking of {level, diff, diff2}, plus its chi-square approximation.

    Returns (W, chi2_stat, chi2_p_value, n_models, n_items).
    """
    rank_matrix = pivot[FEATURES].rank(axis=1, ascending=False, method="average").values
    n_models, n_items = rank_matrix.shape

    rank_sums = rank_matrix.sum(axis=0)
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    W = (12 * S) / (n_models ** 2 * (n_items ** 3 - n_items)) if n_models > 0 else np.nan
    chi2_stat = n_models * (n_items - 1) * W
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=n_items - 1)

    return W, chi2_stat, chi2_p, n_models, n_items


def binomial_level_not_top(pivot: pd.DataFrame):
    """How many models do NOT rank level as their single top feature, and
    the one-sided binomial p-value against the chance base rate 2/3
    (H0: under a uniform random 3-way rank, "level not top" occurs 2/3 of
    the time; H1: it occurs MORE often than that -- i.e. models
    systematically avoid ranking level top).

    Returns (n_level_not_top, n_models, binom_test_result).
    """
    is_level_top = (pivot["level"] >= pivot[["diff", "diff2"]].max(axis=1))
    n_level_not_top = int((~is_level_top).sum())
    n_models = len(pivot)
    result = stats.binomtest(n_level_not_top, n=n_models, p=2 / 3, alternative="greater")
    return n_level_not_top, n_models, result


def family_summary(pivot: pd.DataFrame) -> pd.DataFrame:
    """Family-level aggregation: mean importance per family and whether
    diff+diff2 beats level on average, for that family."""
    summary = pivot.groupby("family").agg(
        n_models=("model", "count"),
        mean_level=("level", "mean"),
        mean_diff=("diff", "mean"),
        mean_diff2=("diff2", "mean"),
    ).reset_index()
    summary["diff_plus_diff2_beats_level"] = (
        (summary["mean_diff"] + summary["mean_diff2"]) > summary["mean_level"]
    )
    return summary


def build_agreement_summary(importance_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top-level entry point: returns (pivot, agreement_summary_df, family_summary_df),
    matching modelclass_agreement_summary.csv and modelclass_family_summary.csv."""
    pivot = pivot_importance(importance_df)
    W, chi2_stat, chi2_p, n_models, n_items = kendalls_w(pivot)
    n_level_not_top, _, binom_result = binomial_level_not_top(pivot)

    agreement_df = pd.DataFrame([{
        "n_models": n_models,
        "kendalls_W": W,
        "chi2_stat": chi2_stat,
        "chi2_p_value": chi2_p,
        "n_models_level_not_top": n_level_not_top,
        "binomial_p_value(level_not_top_gt_chance)": binom_result.pvalue,
    }])

    fam_df = family_summary(pivot)
    return pivot, agreement_df, fam_df


# =============================================================================
# Bootstrap stability (Table 6 / results/model_cross_validation/bootstrap_stability.csv)
# =============================================================================

def bootstrap_stability(
    X: np.ndarray, y: np.ndarray,
    n_boot: int = 200, random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Refit RandomForest and LogisticRegression (two architecturally
    distinct representative tabular models) on n_boot stratified-by-class
    bootstrap resamples of (X, y), each time computing permutation
    importance and recording whether diff+diff2 beats level.

    X must have columns in FEATURES order ([level, diff, diff2]).
    Returns a DataFrame with columns [model, n_boot, frac_diff_or_diff2_beats_level].
    """
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; bootstrap stability cannot be "
            "computed. Install with: pip install scikit-learn"
        )

    rng = np.random.default_rng(random_state)
    pos_idx_pool = np.where(y == 1)[0]
    neg_idx_pool = np.where(y == 0)[0]

    representative_models = [
        ("RandomForest_boot", RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )),
        ("LogReg_boot", LogisticRegression(
            penalty="l2", class_weight="balanced", max_iter=2000,
        )),
    ]

    rows = []
    for name, estimator_template in representative_models:
        diff_wins = np.zeros(n_boot, dtype=bool)
        for b in range(n_boot):
            boot_pos = rng.choice(pos_idx_pool, size=len(pos_idx_pool), replace=True)
            boot_neg = rng.choice(neg_idx_pool, size=len(neg_idx_pool), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])

            X_b, y_b = X[boot_idx], y[boot_idx]
            scaler = StandardScaler().fit(X_b)
            X_bs = scaler.transform(X_b)

            model = estimator_template.__class__(**estimator_template.get_params())
            try:
                model.fit(X_bs, y_b)
            except Exception:
                continue

            perm = permutation_importance(
                model, X_bs, y_b, n_repeats=10, random_state=b, scoring="roc_auc"
            )
            imp = np.clip(perm.importances_mean, 0, None)
            diff_wins[b] = (imp[1] + imp[2]) > imp[0]  # diff + diff2 > level

        rows.append({
            "model": name,
            "n_boot": n_boot,
            "frac_diff_or_diff2_beats_level": float(np.mean(diff_wins)),
        })

    return pd.DataFrame(rows)


def save_all(pivot: pd.DataFrame, agreement_df: pd.DataFrame, fam_df: pd.DataFrame,
             boot_df: pd.DataFrame, out_dir: Path) -> None:
    """Persist agreement_summary / family_summary / bootstrap_stability CSVs
    to out_dir, matching results/model_cross_validation/ in the README."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agreement_df.to_csv(out_dir / "modelclass_agreement_summary.csv", index=False)
    fam_df.to_csv(out_dir / "modelclass_family_summary.csv", index=False)
    boot_df.to_csv(out_dir / "modelclass_bootstrap_stability.csv", index=False)
