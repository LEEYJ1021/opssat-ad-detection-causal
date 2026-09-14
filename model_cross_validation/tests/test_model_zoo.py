"""
tests/test_model_zoo.py
=========================
Unit tests guarding the model-zoo composition itself:

- build_tabular_model_zoo() should register up to 10 models (fewer only if
  xgboost/lightgbm are genuinely missing from the environment).
- build_sequence_model_zoo() should register up to 6 models (or 0 if torch
  is missing, since the whole sequence representation is skipped).
- MLP_shallow must NOT appear in either zoo by default -- it is documented
  in models/shallow_mlp.py but intentionally excluded from run_all.py's
  pipeline (see that module's docstring for the AUC=0.403 rationale).
- Every returned tabular tuple has 4 elements (name, family, estimator, shap_kind);
  every sequence tuple has 3 elements (name, family, ModelClass).
"""

import pytest

from model_cross_validation.models import (
    build_tabular_model_zoo, build_sequence_model_zoo, tree_ensembles,
)


def test_tabular_zoo_excludes_mlp_shallow():
    zoo = build_tabular_model_zoo()
    names = [m[0] for m in zoo]
    assert "MLP_shallow" not in names


def test_tabular_zoo_size_matches_available_soft_deps():
    zoo = build_tabular_model_zoo()
    expected = 8 + (1 if tree_ensembles.HAVE_XGB else 0) + (1 if tree_ensembles.HAVE_LGBM else 0)
    assert len(zoo) == expected
    assert len(zoo) <= 10


def test_tabular_zoo_tuples_are_well_formed():
    zoo = build_tabular_model_zoo()
    for entry in zoo:
        assert len(entry) == 4
        name, family, estimator, shap_kind = entry
        assert isinstance(name, str) and name
        assert isinstance(family, str) and family
        assert shap_kind in {"tree", "linear", "kernel"}


def test_sequence_zoo_size_and_shape():
    zoo = build_sequence_model_zoo()
    assert len(zoo) in (0, 6)  # 0 only if torch is genuinely unavailable
    for entry in zoo:
        assert len(entry) == 3
        name, family, model_class = entry
        assert isinstance(name, str) and name
        assert isinstance(family, str) and family
        assert model_class is not None


def test_expected_family_names_present_in_tabular_zoo():
    zoo = build_tabular_model_zoo()
    families = {m[1] for m in zoo}
    # these families have no soft dependency and must always be present
    for expected_family in {"Linear", "Linear(sparse)", "Probabilistic",
                             "Instance-based", "Kernel", "Tree ensemble(bagging)"}:
        assert expected_family in families
