"""Tests for the AutoML rubric pieces: Optuna tuning and the leakage probe."""

from __future__ import annotations

import numpy as np
import pandas as pd

import train


def _frame(rows: int = 60) -> pd.DataFrame:
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    return pd.DataFrame(
        {
            "tenure": [(index % 24) + 1 for index in range(rows)],
            "MonthlyCharges": [30.0 + (index % 50) for index in range(rows)],
            "TotalCharges": [
                float(((index % 24) + 1) * (30.0 + (index % 50))) for index in range(rows)
            ],
            "Contract": [contracts[index % 3] for index in range(rows)],
            "InternetService": [internet[index % 3] for index in range(rows)],
            "Churn": [1 if index % 3 == 0 else 0 for index in range(rows)],
        }
    )


def test_deliberate_leakage_probe_detects_a_label_copy() -> None:
    frame, _ = train.clean_dataset(_frame())
    features, target, numeric, categorical = train.split_features(frame)
    split = train.train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )
    X_train, X_test, y_train, y_test = split

    probe = train.deliberate_leakage_probe(
        X_train, y_train, X_test, y_test, numeric, categorical
    )

    assert probe["feature"] == train.LEAKY_PROBE_FEATURE
    assert probe["detected"] is True
    assert probe["dropped"] is True
    assert probe["leaky_f1"] == 1.0
    assert probe["clean_f1"] <= probe["leaky_f1"]


def test_optuna_returns_a_fitted_pipeline() -> None:
    frame, _ = train.clean_dataset(_frame())
    features, target, numeric, categorical = train.split_features(frame)

    result = train.optimize_logistic_regression(
        features,
        target,
        numeric,
        categorical,
        n_trials=2,
        timeout=20,
    )

    assert result is not None
    assert result["completed"] is True
    assert result["n_trials"] >= 1
    assert 0.0 <= result["cv_f1_mean"] <= 1.0
    assert "C" in result["best_params"]

    predictions = result["pipeline"].predict(features)
    assert len(predictions) == len(features)
    assert set(np.unique(predictions)) <= {0, 1}


def test_three_models_training_and_shap_generation(tmp_path, monkeypatch) -> None:
    frame, _ = train.clean_dataset(_frame(rows=80))
    features, target, numeric, categorical = train.split_features(frame)
    X_train, X_test, y_train, y_test = train.train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(train, "REPORTS_DIR", reports_dir)

    # 1. Logistic Regression
    lr = train.build_pipeline(
        train.LogisticRegression(max_iter=1000, random_state=42), numeric, categorical
    ).fit(X_train, y_train)
    lr_metrics = train.evaluate(lr, X_test, y_test)
    assert 0.0 <= lr_metrics["f1"] <= 1.0
    assert 0.0 <= lr_metrics["roc_auc"] <= 1.0

    # 2. Random Forest
    rf = train.build_pipeline(
        train.RandomForestClassifier(n_estimators=10, random_state=42), numeric, categorical
    ).fit(X_train, y_train)
    rf_metrics = train.evaluate(rf, X_test, y_test)
    assert 0.0 <= rf_metrics["f1"] <= 1.0
    assert 0.0 <= rf_metrics["roc_auc"] <= 1.0

    # 3. XGBoost
    xgb = train.build_pipeline(
        train.XGBClassifier(n_estimators=10, max_depth=3, eval_metric="logloss", random_state=42),
        numeric,
        categorical,
    ).fit(X_train, y_train)
    xgb_metrics = train.evaluate(xgb, X_test, y_test)
    assert 0.0 <= xgb_metrics["f1"] <= 1.0
    assert 0.0 <= xgb_metrics["roc_auc"] <= 1.0

    # Test SHAP plot generation for all 3 models
    for tag, pipe in (("lr", lr), ("rf", rf), ("xgb", xgb)):
        res = train.generate_model_shap_plots(pipe, X_test, y_test, numeric, categorical, tag, sample_size=20)
        assert res["available"] is True
        assert (reports_dir / f"shap_{tag}.png").exists()
        assert (reports_dir / f"shap_{tag}_force.png").exists()
