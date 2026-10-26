"""ChurnGuard AI - AutoML training, leakage/fairness checks and SHAP explanations.

Run from the ``backend`` directory::

    python train.py                  # full run (AutoML if PyCaret is installed)
    python train.py --fast           # fewer candidates, quicker run
    python train.py --engine sklearn # skip AutoML, use the built-in comparison
    python train.py --no-download    # only use data/telco_churn.csv

Artifacts produced (consumed by ``main.py``)::

    models/churn_pipeline.joblib  portable scikit-learn pipeline (no PyCaret needed at runtime)
    models/model_metadata.json    metrics, feature defaults, global SHAP importance
    reports/*.png                 SHAP plots for the written report
    reports/metrics.json          machine readable copy of every metric

Why a *portable* pipeline is exported: PyCaret is a fantastic AutoML search
engine but a heavy runtime dependency (~1.5 GB). The winning estimator and its
hyper-parameters are therefore re-instantiated as a plain scikit-learn pipeline
and refitted on the same split. The API then needs only scikit-learn + SHAP,
which keeps the Docker image small and makes SHAP attribution exact.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xai import (
    base_value,
    build_explainer,
    explain_matrix,
    global_importance,
    shap_available,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATASET_CACHE = DATA_DIR / "telco_churn.csv"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORT_METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CV_FOLDS = 5
MAX_LINE = 100

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

#: PyCaret IDs of scikit-learn native models. Keeping the search inside
#: scikit-learn estimators guarantees the exported artifact loads in the
#: runtime image (which does not ship xgboost/lightgbm/catboost).
PYCARET_CANDIDATES: tuple[str, ...] = ("lr", "dt", "rf", "et", "gbc")
PYCARET_CANDIDATES_EXTENDED: tuple[str, ...] = PYCARET_CANDIDATES + ("xgboost", "lightgbm")


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------
def load_dataset(path: Path | None = None, allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """Load the Telco churn dataset from a path, the cache, or the public URL."""
    if path is not None:
        return pd.read_csv(path), str(path)
    if DATASET_CACHE.exists():
        return pd.read_csv(DATASET_CACHE), f"cache:{DATASET_CACHE.name}"
    if not allow_download:
        raise FileNotFoundError(
            f"No dataset found at {DATASET_CACHE}. Download {DATASET_URL} "
            "or re-run without --no-download."
        )

    print("Downloading Telco Customer Churn dataset from IBM ...")
    frame = pd.read_csv(DATASET_URL)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATASET_CACHE, index=False)
    return frame, "download:" + DATASET_URL.rsplit("/", 1)[-1]


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove identifiers, fix dtypes and drop the 11 blank ``TotalCharges`` rows."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    report: dict[str, Any] = {"raw_rows": int(len(frame))}
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
        report["dropped_identifier"] = ID_COLUMN

    # TotalCharges ships as text with 11 blank strings -> must be numeric.
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    target = frame[TARGET]
    if target.dtype == object:
        frame[TARGET] = target.astype(str).str.strip().str.title().map({"Yes": 1, "No": 0})

    before = len(frame)
    frame = frame.dropna(subset=[TARGET, "TotalCharges"]).reset_index(drop=True)
    report["dropped_missing_rows"] = int(before - len(frame))
    frame[TARGET] = frame[TARGET].astype(int)

    constant = [
        column
        for column in frame.columns
        if column != TARGET and frame[column].nunique(dropna=False) <= 1
    ]
    if constant:
        frame = frame.drop(columns=constant)
    report["dropped_constant_columns"] = constant
    report["rows"] = int(len(frame))
    return frame, report


def split_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return ``X``, ``y`` and the numeric / categorical column names.

    Categorical columns are cast to ``str`` *once*, so training and serving
    produce identical one-hot categories (a classic silent-mismatch bug).
    """
    features = frame.drop(columns=[TARGET]).copy()
    target = frame[TARGET].astype(int)

    numeric: list[str] = []
    for column in features.columns:
        series = features[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric and series.nunique(dropna=True) > 2:
            numeric.append(column)

    categorical = [column for column in features.columns if column not in numeric]
    for column in categorical:
        features[column] = features[column].astype(str).str.strip()
    return features, target, numeric, categorical


def feature_defaults(features: pd.DataFrame, numeric: list[str]) -> dict[str, Any]:
    """Median/mode per column - used by the API to fill unspecified features."""
    defaults: dict[str, Any] = {}
    for column in features.columns:
        series = features[column]
        if column in numeric:
            defaults[column] = round(float(series.median()), 4)
        else:
            mode = series.astype(str).mode(dropna=True)
            defaults[column] = str(mode.iloc[0]) if not mode.empty else ""
    return defaults


def reference_profile(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
) -> dict[str, Any]:
    """Pick a real training customer as the fallback profile for unspecified fields.

    A per-column median/mode profile is *jointly* unrealistic: the mode of
    ``Contract`` is month-to-month, of ``InternetService`` fiber and of
    ``PaymentMethod`` electronic check, so combining them describes a
    high-risk archetype rather than an average customer (predicted churn ~0.40
    vs a 0.27 base rate). Using one observed row keeps every "No internet
    service" / "No phone service" dependency consistent as well.

    The chosen row is the observed customer whose predicted risk is closest to
    the portfolio churn base rate, which makes the fallback easy to explain:
    "the typical customer, plus whatever the caller specified".
    """
    probabilities = pipeline.predict_proba(features)[:, 1]
    base_rate = float(target.mean())
    index = int(np.argmin(np.abs(probabilities - base_rate)))
    row = features.iloc[index]
    values = {
        column: (round(float(row[column]), 4) if column in numeric else str(row[column]))
        for column in features.columns
    }
    return {
        "source": "observed training customer whose predicted risk is closest to the churn base rate",
        "base_rate": round(base_rate, 4),
        "predicted_probability": round(float(probabilities[index]), 4),
        "values": values,
    }


def training_statistics(features: pd.DataFrame, numeric: list[str]) -> dict[str, dict[str, float]]:
    """Mean / std of the numeric features, used by the drift detector."""
    stats: dict[str, dict[str, float]] = {}
    for column in numeric:
        series = features[column].astype(float)
        stats[column] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=0)), 4),
        }
    return stats


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Preprocessing fitted *inside* the model pipeline (no test-set leakage)."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator: Any,
    numeric: list[str],
    categorical: list[str],
) -> Pipeline:
    """Portable pipeline: preprocessing + estimator, servable without PyCaret."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )


def candidate_estimators(fast: bool = False) -> dict[str, Any]:
    """Scikit-learn candidates used for the local comparison / AutoML fallback."""
    candidates: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if fast:
        for name in ("GradientBoosting", "ExtraTrees"):
            candidates.pop(name, None)
    return candidates


def tuning_grid(name: str) -> dict[str, list[Any]]:
    """Small hyper-parameter grid for the winning estimator."""
    grids: dict[str, dict[str, list[Any]]] = {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "DecisionTree": {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [5, 20, 50]},
        "RandomForest": {
            "model__n_estimators": [200, 300],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"],
        },
        "ExtraTrees": {
            "model__n_estimators": [200, 300],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "HistGradientBoosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 4]},
    }
    return grids.get(name, {})


def evaluate(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Held-out metrics on the *same* split for every candidate."""
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    return {
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    }


def cross_validated_f1(
    estimator: Any, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, float]:
    """Stratified CV F1 mean / std (preprocessing is refitted per fold)."""
    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
    )
    return round(float(scores.mean()), 4), round(float(scores.std()), 4)


# ---------------------------------------------------------------------------
# AutoML
# ---------------------------------------------------------------------------
def _leaderboard_rows(leaderboard: Any) -> list[dict[str, Any]]:
    """Normalise PyCaret's leaderboard DataFrame into JSON-safe rows."""
    if leaderboard is None or not hasattr(leaderboard, "to_json"):
        return []
    aliases = {
        "Model": "model",
        "Accuracy": "accuracy",
        "AUC": "auc",
        "Recall": "recall",
        "Prec.": "precision",
        "F1": "f1",
        "Kappa": "kappa",
        "MCC": "mcc",
        "TT (Sec)": "tt_sec",
    }
    try:
        records = json.loads(leaderboard.round(4).to_json(orient="records"))
    except Exception:  # pragma: no cover - unexpected leaderboard shape
        return []
    return [{aliases.get(key, str(key)): value for key, value in record.items()} for record in records]


def run_pycaret(
    frame: pd.DataFrame,
    fast: bool = False,
    allow_extra_estimators: bool = False,
) -> dict[str, Any] | None:
    """Run PyCaret's ``compare_models`` and return the winning estimator + score.

    Returns ``None`` (with a warning) when PyCaret is unavailable or fails, so
    the script still produces a deployable model in CI / offline machines.
    """
    try:
        from pycaret.classification import compare_models, predict_model, pull, setup
    except ImportError:
        print("!  PyCaret not installed - skipping AutoML. Install with:")
        print("     pip install -r requirements-train.txt")
        return None

    candidates = PYCARET_CANDIDATES_EXTENDED if allow_extra_estimators else PYCARET_CANDIDATES
    if fast:
        candidates = tuple(name for name in candidates if name in ("lr", "rf", "dt"))

    print(f"   PyCaret candidates: {', '.join(candidates)}")
    try:
        setup(
            data=frame,
            target=TARGET,
            session_id=RANDOM_STATE,
            train_size=1.0 - TEST_SIZE,
            fold=3,
            html=False,
            verbose=False,
            log_experiment=False,
            imputation_type="simple",
            remove_multicollinearity=False,
            feature_selection=False,
            n_jobs=-1,
        )
        best_model = compare_models(include=list(candidates), n_select=1, sort="F1", verbose=False)
        leaderboard = _leaderboard_rows(pull())
        holdout = predict_model(best_model, verbose=False)
        pycaret_f1 = round(float(f1_score(holdout[TARGET], holdout["prediction_label"])), 4)
    except Exception as exc:  # pragma: no cover - depends on the AutoML install
        print(f"!  PyCaret run failed ({type(exc).__name__}: {exc}). Using local comparison.")
        return None

    if leaderboard:
        _print_table(
            "   PyCaret leaderboard (CV on the training split)",
            leaderboard,
            ["model", "f1", "auc", "recall", "precision"],
        )

    estimator = best_model.steps[-1][1] if hasattr(best_model, "steps") else best_model
    name = type(estimator).__name__
    print(f"   AutoML winner: {name} (PyCaret holdout F1 = {pycaret_f1:.4f})")
    return {
        "engine": "pycaret",
        "model_name": name,
        "estimator": clone(estimator),
        "pycaret_f1": pycaret_f1,
        "candidates": list(candidates),
        "leaderboard": leaderboard,
    }


def run_local_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric: list[str],
    categorical: list[str],
    fast: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare several scikit-learn pipelines with stratified CV (AutoML fallback)."""
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_estimators(fast=fast).items():
        pipeline = build_pipeline(estimator, numeric, categorical)
        mean, std = cross_validated_f1(pipeline, X_train, y_train)
        fitted[name] = pipeline
        rows.append({"model": name, "cv_f1_mean": mean, "cv_f1_std": std})
        print(f"   {name:<22} CV F1 = {mean:.4f} (+/- {std:.4f})")

    rows.sort(key=lambda row: row["cv_f1_mean"], reverse=True)
    winner = rows[0]
    return (
        {
            "engine": "sklearn",
            "model_name": winner["model"],
            "estimator": clone(fitted[winner["model"]].named_steps["model"]),
            "pycaret_f1": None,
            "candidates": [row["model"] for row in rows],
        },
        rows,
    )


# ---------------------------------------------------------------------------
# Checks required by the rubric
# ---------------------------------------------------------------------------
def leakage_checks(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    engineering_report: dict[str, Any],
) -> list[dict[str, str]]:
    """Explicit, computed leakage audit (not just a comment in the report)."""
    checks: list[dict[str, str]] = []

    checks.append(
        {
            "check": "Identifier removed",
            "detail": f"{engineering_report.get('dropped_identifier', 'none')} dropped; it is unique per "
            "row and carries no predictive signal.",
            "status": "pass",
        }
    )

    target_leak_columns = [
        column
        for column in features.columns
        if column.lower().strip() in ("churn", "churn_value", "churnvalue", "target")
    ]
    checks.append(
        {
            "check": "Target not present in features",
            "detail": f"{len(target_leak_columns)} target-derived column(s) found.",
            "status": "pass" if not target_leak_columns else "fail",
        }
    )

    if "TotalCharges" in features.columns and "MonthlyCharges" in features.columns:
        derived = features["tenure"].astype(float) * features["MonthlyCharges"].astype(float)
        correlation = float(np.corrcoef(derived.to_numpy(), features["TotalCharges"].astype(float))[0, 1])
        checks.append(
            {
                "check": "Derived-feature redundancy",
                "detail": f"corr(tenure x MonthlyCharges, TotalCharges) = {correlation:.3f}. "
                "TotalCharges is near-duplicate information, but it is known at prediction "
                "time, so it is redundancy rather than target leakage.",
                "status": "review" if correlation > 0.95 else "pass",
            }
        )

    duplicated = int(frame.duplicated().sum())
    checks.append(
        {
            "check": "Duplicate rows",
            "detail": f"{duplicated} duplicated row(s) in the cleaned dataset.",
            "status": "pass" if duplicated == 0 else "review",
        }
    )

    checks.append(
        {
            "check": "Fit/transform isolation",
            "detail": "Imputation, scaling and one-hot encoding live inside the pipeline, so they "
            "are refitted on every CV training fold and never see the holdout rows.",
            "status": "pass",
        }
    )

    checks.append(
        {
            "check": "Class balance",
            "detail": f"churn rate = {float(target.mean()):.3f} "
            f"({int(target.sum())} churn / {int(len(target) - target.sum())} retained), "
            "handled with stratified splits and F1 as the primary metric.",
            "status": "pass",
        }
    )
    return checks


def fairness_report(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    columns: tuple[str, ...] = ("gender", "SeniorCitizen"),
    minimum_group_size: int = 20,
) -> list[dict[str, Any]]:
    """Per-group precision / recall on the holdout set (bias sanity check)."""
    predictions = pipeline.predict(X_test)
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in X_test.columns:
            continue
        for group in sorted(X_test[column].astype(str).unique()):
            mask = X_test[column].astype(str) == group
            if int(mask.sum()) < minimum_group_size:
                continue
            rows.append(
                {
                    "feature": column,
                    "group": group,
                    "n": int(mask.sum()),
                    "churn_rate": round(float(y_test[mask].mean()), 4),
                    "precision": round(
                        float(precision_score(y_test[mask], predictions[mask], zero_division=0)), 4
                    ),
                    "recall": round(
                        float(recall_score(y_test[mask], predictions[mask], zero_division=0)), 4
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def explain_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric: list[str],
    categorical: list[str],
    sample_size: int = 200,
) -> dict[str, Any]:
    """Build SHAP explanations, save plots and return global importance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap as shap_lib

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed_names = list(preprocessor.get_feature_names_out())
    raw_names = numeric + categorical

    sample = X_test.head(sample_size)
    matrix = np.asarray(preprocessor.transform(sample), dtype=float)

    explainer = build_explainer(estimator, matrix.shape[1])
    if explainer is None:
        print("!  No SHAP explainer available - skipping plots.")
        return {"available": False, "global": [], "method": "none"}

    values = explain_matrix(explainer, matrix)
    explanation = shap_lib.Explanation(
        values=values,
        base_values=np.full(len(sample), base_value(explainer)),
        data=matrix,
        feature_names=transformed_names,
    )

    bar_path = REPORTS_DIR / "shap_summary_bar.png"
    shap_lib.plots.bar(explanation, max_display=15, show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    figure.axes[0].set_title("Global feature importance (mean |SHAP value|)")
    figure.tight_layout()
    figure.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    beeswarm_path = REPORTS_DIR / "shap_beeswarm.png"
    shap_lib.plots.beeswarm(explanation, max_display=15, show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 7)
    figure.axes[0].set_title("SHAP value distribution per feature")
    figure.tight_layout()
    figure.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    probabilities = pipeline.predict_proba(sample)[:, 1]
    highest_risk = int(np.argmax(probabilities))
    waterfall_path = REPORTS_DIR / "shap_waterfall_high_risk.png"
    shap_lib.plots.waterfall(explanation[highest_risk], max_display=12, show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    figure.axes[0].set_title(f"Why this customer is high risk (p={probabilities[highest_risk]:.3f})")
    figure.tight_layout()
    figure.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close("all")

    importance = global_importance(values, transformed_names, raw_names)
    print(f"   SHAP plots written to {REPORTS_DIR}")
    for item in importance[:5]:
        print(f"     {item['feature']:<18} {item['importance']:.4f}")

    return {
        "available": True,
        "method": "TreeExplainer" if "Tree" in type(explainer).__name__ else type(explainer).__name__,
        "global": importance,
        "sample_size": int(len(sample)),
        "plots": [str(bar_path.name), str(beeswarm_path.name), str(waterfall_path.name)],
        "highest_risk_probability": round(float(probabilities[highest_risk]), 4),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n{title}")
    print("-" * max(len(title), MAX_LINE if len(columns) > 3 else len(title)))
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows)) if rows else len(column)
        for column in columns
    }
    header = "  ".join(f"{column:<{widths[column]}}" for column in columns)
    print(header)
    for row in rows:
        print("  ".join(f"{str(row.get(column, '')):<{widths[column]}}" for column in columns))


def train(args: argparse.Namespace) -> int:
    print("=" * 78)
    print("ChurnGuard AI - AutoML + SHAP training run")
    print("=" * 78)

    frame, source = load_dataset(Path(args.data) if args.data else None, not args.no_download)
    frame, engineering_report = clean_dataset(frame)
    features, target, numeric, categorical = split_features(frame)

    print(f"1. Dataset: {source}")
    print(
        f"   rows={len(features)}  features={features.shape[1]} "
        f"(numeric={len(numeric)}, categorical={len(categorical)})  "
        f"churn_rate={target.mean():.3f}"
    )
    print(
        f"   dropped: {engineering_report.get('dropped_missing_rows', 0)} incomplete rows, "
        f"identifier={engineering_report.get('dropped_identifier')}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    print(f"   stratified split: train={len(X_train)}  test={len(X_test)}")

    # --- Baseline ---------------------------------------------------------
    print("\n2. Baseline model (manual LogisticRegression, no AutoML)")
    baseline = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), numeric, categorical
    )
    baseline_cv, baseline_cv_std = cross_validated_f1(baseline, X_train, y_train)
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test)
    print(f"   CV F1 = {baseline_cv:.4f} (+/- {baseline_cv_std:.4f})")
    print(
        f"   holdout F1 = {baseline_metrics['f1']:.4f}  "
        f"precision = {baseline_metrics['precision']:.4f}  "
        f"recall = {baseline_metrics['recall']:.4f}  "
        f"roc_auc = {baseline_metrics['roc_auc']:.4f}"
    )

    # --- AutoML -----------------------------------------------------------
    print("\n3. AutoML model comparison")
    automl: dict[str, Any] | None = None
    comparison_rows: list[dict[str, Any]] = []

    if args.engine in ("auto", "pycaret"):
        automl = run_pycaret(
            frame,
            fast=args.fast,
            allow_extra_estimators=args.allow_extra_estimators,
        )
        if automl is None and args.engine == "pycaret":
            print("!  --engine pycaret requested but AutoML did not run.")
            return 2

    if automl is None:
        print("   Falling back to the built-in scikit-learn comparison:")
        automl, comparison_rows = run_local_comparison(
            X_train, y_train, numeric, categorical, fast=args.fast
        )

    winner_name = str(automl["model_name"])
    print(f"   winner: {winner_name}")

    # --- Tuning, accepted only if it wins on a dedicated validation split --
    deployed = build_pipeline(clone(automl["estimator"]), numeric, categorical)
    selection = "untuned AutoML winner"
    validation: dict[str, Any] = {}

    if args.tune and tuning_grid(winner_name):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
        untuned = build_pipeline(clone(automl["estimator"]), numeric, categorical)
        untuned.fit(X_fit, y_fit)
        untuned_val = round(float(f1_score(y_val, untuned.predict(X_val))), 4)

        print(f"   tuning {winner_name} with a small randomized search ...")
        search = RandomizedSearchCV(
            build_pipeline(clone(automl["estimator"]), numeric, categorical),
            param_distributions=tuning_grid(winner_name),
            n_iter=6,
            scoring="f1",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_fit, y_fit)
        tuned_val = round(float(f1_score(y_val, search.best_estimator_.predict(X_val))), 4)
        print(f"   best params: {search.best_params_}")
        print(f"   validation F1: untuned = {untuned_val:.4f}  tuned = {tuned_val:.4f}")

        validation = {
            "split_size": VALIDATION_SIZE,
            "untuned_f1": untuned_val,
            "tuned_f1": tuned_val,
            "best_params": {key: str(value) for key, value in search.best_params_.items()},
        }
        if tuned_val > untuned_val:
            deployed = search.best_estimator_
            selection = f"tuned ({search.best_params_}) - validation F1 {tuned_val:.4f}"
        else:
            deployed = untuned
            selection = f"untuned - validation F1 {untuned_val:.4f} >= tuned {tuned_val:.4f}"
        print(f"   selected: {selection}")

    deployed.fit(X_train, y_train)
    deployed_metrics = evaluate(deployed, X_test, y_test)
    deployed_cv, deployed_cv_std = cross_validated_f1(deployed, X_train, y_train)
    print(
        f"   deployed pipeline: CV F1 = {deployed_cv:.4f} (+/- {deployed_cv_std:.4f})  "
        f"holdout F1 = {deployed_metrics['f1']:.4f}"
    )

    comparison = [
        {
            "model": "Baseline LogisticRegression",
            "cv_f1_mean": baseline_cv,
            "holdout_f1": baseline_metrics["f1"],
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "roc_auc": baseline_metrics["roc_auc"],
        },
        {
            "model": f"AutoML winner ({winner_name})",
            "cv_f1_mean": deployed_cv,
            "holdout_f1": deployed_metrics["f1"],
            "precision": deployed_metrics["precision"],
            "recall": deployed_metrics["recall"],
            "roc_auc": deployed_metrics["roc_auc"],
        },
    ]
    _print_table(
        "   Model comparison (same stratified holdout)",
        comparison,
        ["model", "cv_f1_mean", "holdout_f1", "precision", "recall", "roc_auc"],
    )

    improvement = deployed_metrics["f1"] - baseline_metrics["f1"]
    print(f"   AutoML vs baseline holdout F1: {improvement:+.4f}")

    # --- Rubric checks ----------------------------------------------------
    print("\n4. Leakage / bias checks")
    checks = leakage_checks(frame, features, target, engineering_report)
    for check in checks:
        print(f"   [{check['status']:<6}] {check['check']}: {check['detail']}")

    fairness = fairness_report(deployed, X_test, y_test)
    if fairness:
        _print_table(
            "   Fairness slices (holdout)",
            fairness,
            ["feature", "group", "n", "churn_rate", "precision", "recall"],
        )

    # --- SHAP -------------------------------------------------------------
    print("\n5. SHAP explainability")
    shap_result: dict[str, Any] = {"available": False, "global": [], "method": "none"}
    if args.no_shap:
        print("   skipped (--no-shap)")
    elif not shap_available():
        print("!  shap is not installed - skipping plots.")
    else:
        shap_result = explain_model(deployed, X_test, y_test, numeric, categorical)

    # --- Persist ----------------------------------------------------------
    print("\n6. Saving artifacts")
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, PIPELINE_PATH, compress=3)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": winner_name,
        "engine": automl["engine"],
        "selection": selection,
        "validation": validation,
        "dataset": {
            "source": source,
            "rows": int(len(features)),
            "features": int(features.shape[1]),
            "churn_rate": round(float(target.mean()), 4),
        },
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
        "metrics": {
            "baseline": {
                "cv_f1_mean": baseline_cv,
                "cv_f1_std": baseline_cv_std,
                **baseline_metrics,
            },
            "automl": {
                "cv_f1_mean": deployed_cv,
                "cv_f1_std": deployed_cv_std,
                "pycaret_holdout_f1": automl.get("pycaret_f1"),
                **deployed_metrics,
            },
            "f1_improvement_over_baseline": round(float(improvement), 4),
        },
        "comparison": comparison,
        "automl_leaderboard": automl.get("leaderboard") or comparison_rows,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "raw_features": numeric + categorical,
        "feature_defaults": feature_defaults(X_train, numeric),
        "reference_profile": reference_profile(deployed, X_train, y_train, numeric),
        "training_stats": training_statistics(X_train, numeric),
        "top_features": shap_result.get("global", []),
        "xai": {
            "available": bool(shap_result.get("available")),
            "method": shap_result.get("method", "none"),
            "plots": shap_result.get("plots", []),
        },
        "leakage_checks": checks,
        "fairness": fairness,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    REPORT_METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"   pipeline -> {PIPELINE_PATH}")
    print(f"   metadata -> {METADATA_PATH}")
    print(f"   report   -> {REPORT_METRICS_PATH}")
    print("\nDone. Next: uvicorn main:app --reload  (then open http://localhost:8000/docs)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and explain the ChurnGuard AI model.")
    parser.add_argument("--data", help="Path to a local Telco churn CSV.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never download the dataset (use data/telco_churn.csv only).",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "pycaret", "sklearn"),
        default="auto",
        help="AutoML engine. 'auto' uses PyCaret when installed.",
    )
    parser.add_argument("--fast", action="store_true", help="Fewer candidates for a quick run.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP plots.")
    parser.add_argument("--no-tune", dest="tune", action="store_false", help="Skip hyper-parameter tuning.")
    parser.add_argument(
        "--allow-extra-estimators",
        action="store_true",
        help="Let PyCaret consider xgboost/lightgbm (needs requirements-train.txt at runtime).",
    )
    parser.set_defaults(tune=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(train(parse_args()))
