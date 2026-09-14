"""
models/tree_ensembles.py
==========================
Tree ensemble families: bagging (RandomForest, ExtraTrees) and boosting
(GradBoost/sklearn, XGBoost, LightGBM). Tabular representation.
SHAP kind = "tree" (shap.TreeExplainer) for all five.

xgboost and lightgbm are soft dependencies -- if either is missing, that
model is silently omitted from the returned zoo (matching the original
script's "soft dependency" behavior) rather than raising.

[PATCH 2 -- carried over from the original monolithic script] XGBClassifier
is constructed WITHOUT `use_label_encoder=False`. That parameter was removed
entirely in xgboost 2.x and passing it raises
`TypeError: __init__() got an unexpected keyword argument 'use_label_encoder'`,
crashing the whole model zoo. Modern xgboost behaves identically without it.
"""

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except (ImportError, OSError):
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except (ImportError, OSError):
    HAVE_LGBM = False

RANDOM_STATE = 42


def build_models():
    """Always returns RandomForest/ExtraTrees/GradBoost(sklearn); appends
    XGBoost and/or LightGBM only if installed. Never raises -- xgboost/
    lightgbm are optional accelerants, not requirements, for this family."""
    zoo = [
        (
            "RandomForest", "Tree ensemble(bagging)",
            RandomForestClassifier(
                n_estimators=300, max_depth=6, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "tree",
        ),
        (
            "ExtraTrees", "Tree ensemble(bagging,extra-random)",
            ExtraTreesClassifier(
                n_estimators=300, max_depth=6, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "tree",
        ),
        (
            "GradBoost_sklearn", "Tree ensemble(boosting)",
            GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
            "tree",
        ),
    ]

    if HAVE_XGB:
        # [PATCH 2] `use_label_encoder=False` intentionally omitted.
        zoo.append((
            "XGBoost", "Tree ensemble(boosting)",
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
            ),
            "tree",
        ))
    else:
        print("[info] xgboost not installed -- skipping XGBoost. "
              "Install with: pip install xgboost")

    if HAVE_LGBM:
        zoo.append((
            "LightGBM", "Tree ensemble(boosting)",
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                random_state=RANDOM_STATE, verbosity=-1,
            ),
            "tree",
        ))
    else:
        print("[info] lightgbm not installed -- skipping LightGBM. "
              "Install with: pip install lightgbm")

    return zoo
