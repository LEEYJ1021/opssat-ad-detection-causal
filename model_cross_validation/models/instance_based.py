"""
models/instance_based.py
==========================
Instance-based family: k-Nearest Neighbors (k=15).
Tabular representation. SHAP kind = "kernel" (kNN has no gradient/tree
structure, so KernelExplainer with a permutation-importance fallback is used).
"""

try:
    from sklearn.neighbors import KNeighborsClassifier
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False


def build_models():
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; the Instance-based family "
            "cannot be built. Install with: pip install scikit-learn"
        )
    return [
        ("kNN", "Instance-based", KNeighborsClassifier(n_neighbors=15), "kernel"),
    ]
