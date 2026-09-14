"""
models/linear.py
==================
Linear / Linear (sparse) family: L2- and L1-regularized logistic regression.
Tabular representation. SHAP kind = "linear" (shap.LinearExplainer).
"""

try:
    from sklearn.linear_model import LogisticRegression
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False


def build_models():
    """Return [(name, family, sklearn_estimator, shap_kind), ...].

    Raises ImportError with a clear message if scikit-learn is unavailable
    (caller should catch and skip this family, per the "soft dependency"
    convention used throughout the package)."""
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; the Linear/Linear(sparse) family "
            "cannot be built. Install with: pip install scikit-learn"
        )
    return [
        (
            "LogReg_L2", "Linear",
            LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000),
            "linear",
        ),
        (
            "LogReg_L1", "Linear(sparse)",
            LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000),
            "linear",
        ),
    ]
