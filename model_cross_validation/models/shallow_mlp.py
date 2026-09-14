"""
models/shallow_mlp.py
=======================
Neural net (shallow) family: a small 2-hidden-layer MLP (16, 8 units).

*** EXCLUDED FROM THE 16-MODEL ZOO ***

[PATCH 3 -- see docs/dev-log/step_ai_model_cross_validation.md] When this
model was actually run, it converged to out-of-fold AUC = 0.403 -- BELOW
chance (0.5). This is judged to be a failed optimization (non-convergence /
instability), not a genuine "this inductive bias favors level" result: an
untrained model's feature attributions are uninterpretable noise, and
including them in the 16-model agreement statistics (Kendall's W, binomial
test) would risk distorting the conclusion rather than stress-testing it.

Consequently, run_all.py does NOT call build_models() from this module by
default -- it is intentionally left out of the model zoo assembled in
run_all.py, and the package-wide model count is 16 (10 tabular + 6 sequence),
not 17. This module is kept only so the excluded model's specification and
exclusion rationale are documented and versioned alongside the rest of the
zoo, and so a future investigator can re-enable it (e.g. after diagnosing
the convergence failure -- candidate causes noted below) for debugging
without hunting through git history.

Candidate causes for the convergence failure (not yet diagnosed further,
flagged for future work per the original dev-log):
  - early_stopping fraction too aggressive relative to the small positive
    class (168 positives / 4,828 negatives)
  - no explicit class-weighting equivalent for MLPClassifier (unlike the
    other tabular models, which all use class_weight="balanced")
  - default Adam learning rate poorly suited to the 3-feature, heavily
    imbalanced input
"""

try:
    from sklearn.neural_network import MLPClassifier
    HAVE_SKLEARN = True
except (ImportError, OSError):
    HAVE_SKLEARN = False

RANDOM_STATE = 42

EXCLUDED = True
EXCLUSION_REASON = (
    "AUC=0.403 (below chance); judged a failed fit (non-convergence), not a "
    "genuine level-favoring inductive bias. Excluded from the 16-model "
    "agreement statistics (Kendall's W, binomial test, family_summary)."
)


def build_models():
    """Returns the model spec for documentation/manual re-enablement only.
    NOT called by run_all.py's default pipeline -- see module docstring."""
    if not HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is not installed; the (excluded) Neural net "
            "(shallow) family cannot be built. Install with: "
            "pip install scikit-learn"
        )
    return [
        (
            "MLP_shallow", "Neural net(shallow)",
            MLPClassifier(
                hidden_layer_sizes=(16, 8), max_iter=2000,
                random_state=RANDOM_STATE, early_stopping=True,
            ),
            "kernel",
        ),
    ]
