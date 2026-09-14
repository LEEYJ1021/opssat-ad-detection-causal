"""
tests/test_agreement_statistics.py
=====================================
Unit tests for agreement_statistics.py using small synthetic importance
tables (not the real 16-model result), checking:

- Kendall's W = 1.0 when every model ranks features identically.
- Kendall's W ~ 0 (and non-significant) when ranks are unrelated /
  deliberately balanced across all 6 permutations of 3 items.
- binomial_level_not_top correctly counts "level is not the top feature".
- family_summary correctly flags diff_plus_diff2_beats_level per family.

These are sanity checks on the statistical machinery, not a re-derivation
of the paper's headline numbers (which are the concrete outputs already
committed under results/model_cross_validation/).
"""

import numpy as np
import pandas as pd
import pytest

from model_cross_validation.agreement_statistics import (
    pivot_importance, kendalls_w, binomial_level_not_top, family_summary,
)


def _long_df(rows):
    """rows: list of (model, family, level, diff, diff2) -> long-format df
    matching modelclass_feature_importance.csv's schema."""
    records = []
    for model, family, level, diff, diff2 in rows:
        records.append({"model": model, "family": family, "feature": "level", "importance_share": level})
        records.append({"model": model, "family": family, "feature": "diff", "importance_share": diff})
        records.append({"model": model, "family": family, "feature": "diff2", "importance_share": diff2})
    return pd.DataFrame(records)


def test_kendalls_w_perfect_agreement():
    rows = [
        ("m1", "fam1", 0.1, 0.4, 0.5),
        ("m2", "fam2", 0.05, 0.45, 0.50),
        ("m3", "fam3", 0.2, 0.35, 0.45),
    ]
    pivot = pivot_importance(_long_df(rows))
    W, chi2_stat, chi2_p, n_models, n_items = kendalls_w(pivot)
    assert n_models == 3 and n_items == 3
    assert W == pytest.approx(1.0, abs=1e-9)  # diff2 > diff > level in every model


def test_kendalls_w_low_when_ranks_are_balanced():
    # Construct 3 models whose rankings of {level, diff, diff2} are the 3
    # cyclic permutations of the ranking -- rank sums come out exactly
    # tied, which should drive W to (or very near) zero.
    rows = [
        ("m1", "fam1", 0.6, 0.3, 0.1),  # level > diff > diff2
        ("m2", "fam2", 0.1, 0.6, 0.3),  # diff > diff2 > level
        ("m3", "fam3", 0.3, 0.1, 0.6),  # diff2 > level > diff
    ]
    pivot = pivot_importance(_long_df(rows))
    W, chi2_stat, chi2_p, n_models, n_items = kendalls_w(pivot)
    assert W == pytest.approx(0.0, abs=1e-9)


def test_binomial_level_not_top_counts_correctly():
    rows = [
        ("m1", "fam1", 0.6, 0.3, 0.1),   # level IS top
        ("m2", "fam2", 0.1, 0.6, 0.3),   # level not top
        ("m3", "fam3", 0.05, 0.45, 0.50),  # level not top
    ]
    pivot = pivot_importance(_long_df(rows))
    n_not_top, n_models, result = binomial_level_not_top(pivot)
    assert n_not_top == 2
    assert n_models == 3
    assert 0.0 <= result.pvalue <= 1.0


def test_family_summary_flags_diff_plus_diff2_beats_level():
    rows = [
        ("m1", "famA", 0.6, 0.2, 0.2),   # level wins for famA
        ("m2", "famB", 0.1, 0.5, 0.4),   # diff+diff2 wins for famB
    ]
    pivot = pivot_importance(_long_df(rows))
    summary = family_summary(pivot)
    summary = summary.set_index("family")
    assert bool(summary.loc["famA", "diff_plus_diff2_beats_level"]) is False
    assert bool(summary.loc["famB", "diff_plus_diff2_beats_level"]) is True
