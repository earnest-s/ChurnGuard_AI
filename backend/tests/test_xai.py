"""Unit tests for the SHAP helper layer (shape normalisation + aggregation)."""

from __future__ import annotations

import numpy as np
import pytest

from xai import (
    aggregate_to_raw_features,
    explain_matrix,
    normalized_absolute,
    source_feature,
)


class _FakeExplainer:
    """Mimics the different SHAP return conventions across versions."""

    def __init__(self, values):
        self._values = values

    def shap_values(self, matrix, check_additivity=False):
        return self._values


def test_source_feature_maps_one_hot_columns() -> None:
    raw = ["tenure", "Internet", "InternetService"]
    assert source_feature("tenure", raw) == "tenure"
    # Longest prefix wins, otherwise "Internet" would swallow "InternetService_DSL".
    assert source_feature("InternetService_DSL", raw) == "InternetService"


def test_aggregate_to_raw_features_sums_one_hot_values() -> None:
    contributions = aggregate_to_raw_features(
        values=[1.0, 2.0, -0.5],
        transformed_names=["Contract_Month-to-month", "Contract_One year", "tenure"],
        raw_names=["tenure", "MonthlyCharges", "Contract"],
    )
    assert contributions["Contract"] == pytest.approx(3.0)
    assert contributions["tenure"] == pytest.approx(-0.5)


def test_normalized_absolute_sums_to_one() -> None:
    importance = normalized_absolute({"a": -2.0, "b": 2.0, "c": 0.0})
    assert sum(importance.values()) == pytest.approx(1.0, abs=0.001)
    assert importance["a"] == importance["b"]


def test_normalized_absolute_handles_all_zero() -> None:
    assert normalized_absolute({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


@pytest.mark.parametrize(
    "values",
    [
        # Legacy list-of-arrays (class 0, class 1)
        [np.zeros((3, 2)), np.ones((3, 2))],
        # Modern 2-D array
        np.ones((3, 2)),
        # Modern 3-D array (rows, features, outputs)
        np.dstack([np.zeros((3, 2)), np.ones((3, 2))]),
    ],
)
def test_explain_matrix_normalises_shapes(values) -> None:
    matrix = explain_matrix(_FakeExplainer(values), np.zeros((3, 2)))
    assert matrix.shape == (3, 2)
    assert np.allclose(matrix, 1.0)


def test_explain_matrix_falls_back_without_check_additivity() -> None:
    class _StrictExplainer:
        def shap_values(self, matrix):
            return np.ones((matrix.shape[0], matrix.shape[1]))

    matrix = explain_matrix(_StrictExplainer(), np.zeros((2, 4)))
    assert matrix.shape == (2, 4)
