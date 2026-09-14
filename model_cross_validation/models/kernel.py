"""
models/kernel.py
==================
Kernel family: RBF-kernel SVM (probability=True so predict_proba is
available for AUC and for the KernelExplainer SHAP path).
Tabular representation. SHAP kind = "kernel".
"""

try:
    from sklearn.svm import SVC
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False


def build_models():
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; the Kernel family cannot be "
            "built. Install with: pip install scikit-learn"
        )
    return [
        (
            "SVM_RBF", "Kernel",
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            "kernel",
        ),
    ]
