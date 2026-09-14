"""
attribution/shap_permutation.py
==================================
Feature-attribution for the 10 tabular models: SHAP where computable
(TreeExplainer / LinearExplainer / KernelExplainer, chosen per model via the
`shap_kind` tag from models.*.build_models()), with a model-agnostic
permutation-importance fallback (always available, used whenever SHAP is
missing or fails for a given model/fold).

[PATCH 1 -- see docs/dev-log/step_ai_model_cross_validation.md] Newer SHAP
versions return a 3D (n_samples, n_features, n_classes) ndarray from
shap_values() for many explainer/model pairs (GaussianNB, kNN, SVM_RBF, and
some tree paths) instead of the legacy [class0_array, class1_array] list
format the original code only handled for tree/kernel paths. Left
unhandled, importance.shape becomes (n_features, n_classes) and the later
`float(imp)` conversion crashes with
"TypeError: only 0-dimensional arrays can be converted to Python scalars"
(reproduced with GaussianNB). _normalize_shap_values() below absorbs both
the list and 3D-ndarray cases uniformly, returning a (n_samples, n_features)
2D array or None (triggering the permutation-importance fallback) if
normalization is not possible.
"""

import numpy as np

try:
    import shap
    HAVE_SHAP = True
except (ImportError, OSError):
    HAVE_SHAP = False

try:
    from sklearn.inspection import permutation_importance
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False

RANDOM_STATE = 42


def _normalize_shap_values(sv, n_features: int):
    """[PATCH 1] Normalize shap_values() output across SHAP versions/paths
    into a (n_samples, n_features) 2D ndarray, or return None if that is not
    possible.

    Handles:
      - legacy list format [class0_array, class1_array, ...] (binary
        classification -> use the positive-class array, index 1)
      - modern 3D ndarray (n_samples, n_features, n_classes) (use the
        positive-class slice along the last axis, index 1, if present)
      - already-2D ndarray (pass through)
      - 1D ndarray (single-sample edge case; reshaped to (1, n_features))
    """
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]

    sv = np.asarray(sv)

    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)

    if sv.ndim != 2 or sv.shape[1] != n_features:
        return None

    return sv


def compute_shap_importance(model, kind: str, X_train: np.ndarray, X_explain: np.ndarray):
    """Return a normalized (n_features,) importance-share vector via SHAP,
    or None if SHAP is unavailable/fails (caller should fall back to
    permutation importance -- see permutation_importance_fallback()).

    kind: "tree" -> TreeExplainer, "linear" -> LinearExplainer,
          "kernel" -> KernelExplainer (with a 20-point KMeans background,
          nsamples=100 -- 3 features keeps this cheap regardless of kind).
    """
    if not HAVE_SHAP:
        return None
    try:
        if kind == "tree":
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_explain)
        elif kind == "linear":
            explainer = shap.LinearExplainer(model, X_train)
            sv = explainer.shap_values(X_explain)
        else:
            background = shap.kmeans(X_train, min(20, len(X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            sv = explainer.shap_values(X_explain, nsamples=100)

        n_features = X_explain.shape[1]
        sv = _normalize_shap_values(sv, n_features)
        if sv is None:
            print("    [SHAP return-shape normalization failed -> falling back to permutation]")
            return None

        importance = np.abs(sv).mean(axis=0)
        importance = np.asarray(importance).reshape(-1)
        if importance.shape[0] != n_features:
            print(f"    [SHAP importance shape mismatch ({importance.shape}) -> falling back to permutation]")
            return None

        total = importance.sum()
        return (importance / total) if total > 0 else None
    except Exception as e:
        print(f"    [SHAP failed -> falling back to permutation] {e}")
        return None


def permutation_importance_fallback(model, X_te: np.ndarray, y_te: np.ndarray,
                                     n_repeats: int = 20, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Model-agnostic fallback: ROC-AUC permutation importance, clipped to
    >= 0 and renormalized to sum to 1. Always available (requires only
    scikit-learn). Returns a uniform [1/3, 1/3, 1/3] vector in the
    degenerate case where every clipped importance is zero."""
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; permutation-importance fallback "
            "is unavailable. Install with: pip install scikit-learn"
        )
    perm = permutation_importance(
        model, X_te, y_te, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc"
    )
    imp = np.clip(perm.importances_mean, 0, None)
    total = imp.sum()
    return (imp / total) if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3])


def compute_importance(model, kind: str, X_train: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                        explain_sample_size: int = 400, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Top-level entry point used by run_all.py: try SHAP on a bounded
    explain-sample, fall back to permutation importance on the full test
    fold if SHAP is unavailable or fails."""
    rng = np.random.default_rng(random_state)
    explain_idx = rng.choice(len(X_te), size=min(explain_sample_size, len(X_te)), replace=False)

    shap_imp = compute_shap_importance(model, kind, X_train, X_te[explain_idx])
    if shap_imp is not None:
        return shap_imp
    return permutation_importance_fallback(model, X_te, y_te, random_state=random_state)
