"""SHAP helpers shared by the training script and the FastAPI service.

The helpers in this module are deliberately defensive: SHAP changed its return
shapes between versions (list of arrays vs 2-D array vs 3-D array), and the
serving code must never crash because of that. Everything is normalised to a
``(n_rows, n_features)`` matrix of SHAP values for the positive class.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

try:  # SHAP is optional so the API can still boot without the heavy dependency.
    import shap
except ImportError:  # pragma: no cover - exercised only in minimal environments
    shap = None  # type: ignore[assignment]


TREE_MODEL_HINTS: tuple[str, ...] = (
    "RandomForest",
    "ExtraTrees",
    "DecisionTree",
    "GradientBoosting",
    "HistGradientBoosting",
    "XGB",
    "LGBM",
    "CatBoost",
)
LINEAR_MODEL_HINTS: tuple[str, ...] = (
    "LogisticRegression",
    "LogisticRegressionCV",
    "Ridge",
    "LinearRegression",
    "SGDClassifier",
)


def shap_available() -> bool:
    """Return ``True`` when the ``shap`` package can be imported."""
    return shap is not None


def is_tree_model(estimator: Any) -> bool:
    """Heuristic check for estimators that ``shap.TreeExplainer`` understands."""
    name = type(estimator).__name__
    return any(hint in name for hint in TREE_MODEL_HINTS)


def is_linear_model(estimator: Any) -> bool:
    """Heuristic check for estimators that ``shap.LinearExplainer`` understands."""
    name = type(estimator).__name__
    return any(hint in name for hint in LINEAR_MODEL_HINTS)


def build_explainer(estimator: Any, n_features: int) -> Any | None:
    """Create the cheapest correct SHAP explainer for ``estimator``.

    Returns ``None`` when no explainer can be built (missing SHAP, exotic
    estimator). Callers must degrade gracefully in that case.
    """
    if shap is None:
        return None

    if is_tree_model(estimator):
        try:
            return shap.TreeExplainer(estimator)
        except Exception:  # pragma: no cover - depends on SHAP internals
            return None

    if is_linear_model(estimator):
        # In scaled space the mean feature vector is all zeros. Shap's linear
        # explainer only needs *a* background distribution to centre on.
        background = np.zeros((1, max(1, int(n_features))), dtype=float)
        masker = shap.maskers.Independent(background, max_samples=1)
        try:
            return shap.LinearExplainer(estimator, masker)
        except Exception:  # pragma: no cover - depends on SHAP internals
            return None

    return None


def _as_positive_class(values: Any) -> np.ndarray:
    """Normalise any SHAP output into a 2-D positive-class matrix."""
    if isinstance(values, list):
        # Legacy SHAP API: [class_0_values, class_1_values]
        array = np.asarray(values[1] if len(values) >= 2 else values[0], dtype=float)
    else:
        array = np.asarray(values, dtype=float)

    if array.ndim == 3:
        # (n_rows, n_features, n_outputs)
        array = array[:, :, 1] if array.shape[2] >= 2 else array[:, :, 0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def explain_matrix(explainer: Any, matrix: np.ndarray) -> np.ndarray:
    """Return positive-class SHAP values for every row of ``matrix``."""
    try:
        values = explainer.shap_values(matrix, check_additivity=False)
    except TypeError:
        # LinearExplainer does not accept ``check_additivity``.
        values = explainer.shap_values(matrix)
    return _as_positive_class(values)


def base_value(explainer: Any) -> float:
    """Return the scalar expected value for the positive class."""
    expected = getattr(explainer, "expected_value", 0.0)
    array = np.asarray(expected, dtype=float).ravel()
    if array.size == 0:
        return 0.0
    return float(array[1]) if array.size >= 2 else float(array[0])


def source_feature(transformed_name: str, raw_names: Iterable[str]) -> str:
    """Map a transformed column (``Contract_One year``) back to ``Contract``."""
    raw = list(raw_names)
    if transformed_name in raw:
        return transformed_name
    # Longest name first so that ``InternetService`` wins over ``Internet``.
    for name in sorted(raw, key=len, reverse=True):
        if transformed_name.startswith(f"{name}_"):
            return name
    return transformed_name


def aggregate_to_raw_features(
    values: Iterable[float],
    transformed_names: Iterable[str],
    raw_names: Iterable[str],
) -> dict[str, float]:
    """Sum one-hot SHAP values back onto the feature the user actually supplied."""
    totals: dict[str, float] = {}
    for name, value in zip(list(transformed_names), np.asarray(values, dtype=float).ravel()):
        source = source_feature(str(name), raw_names)
        totals[source] = totals.get(source, 0.0) + float(value)
    return totals


def normalized_absolute(contributions: Mapping[str, float]) -> dict[str, float]:
    """Absolute SHAP values scaled to sum to 1 (used for the API bar chart)."""
    magnitudes = {str(key): abs(float(value)) for key, value in contributions.items()}
    total = float(sum(magnitudes.values()))
    if total <= 0.0:
        return {key: 0.0 for key in magnitudes}
    return {key: round(value / total, 4) for key, value in magnitudes.items()}


def global_importance(
    matrix: np.ndarray,
    transformed_names: Iterable[str],
    raw_names: Iterable[str],
) -> list[dict[str, float | str]]:
    """Mean absolute SHAP value per raw feature, normalised and sorted."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    per_raw: dict[str, float] = {}
    names = list(transformed_names)
    for column, name in enumerate(names):
        if column >= matrix.shape[1]:
            break
        contribution = float(np.mean(np.abs(matrix[:, column])))
        source = source_feature(str(name), raw_names)
        per_raw[source] = per_raw.get(source, 0.0) + contribution

    total = sum(per_raw.values())
    if total > 0:
        per_raw = {key: value / total for key, value in per_raw.items()}

    ordered = sorted(per_raw.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "importance": round(value, 4)} for feature, value in ordered]


def top_factors(contributions: Mapping[str, float], limit: int = 3) -> list[dict[str, float | str]]:
    """Signed contributions ranked by magnitude (drivers of this prediction)."""
    ordered = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return [
        {"feature": feature, "contribution": round(float(value), 4)}
        for feature, value in ordered[:limit]
    ]
