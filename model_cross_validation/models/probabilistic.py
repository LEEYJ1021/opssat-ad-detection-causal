"""
models/probabilistic.py
=========================
Probabilistic family: Gaussian Naive Bayes.
Tabular representation. SHAP kind = "kernel" (GaussianNB has no linear/tree
structure SHAP can exploit directly, so KernelExplainer is used, with
permutation-importance fallback -- see attribution/shap_permutation.py).
"""

try:
    from sklearn.naive_bayes import GaussianNB
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False


def build_models():
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; the Probabilistic family cannot "
            "be built. Install with: pip install scikit-learn"
        )
    return [
        ("GaussianNB", "Probabilistic", GaussianNB(), "kernel"),
    ]
