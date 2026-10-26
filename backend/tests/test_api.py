"""API contract tests for the ChurnGuard AI service.

They must pass with no trained artifact (CI has no model) *and* verify the
AutoML + SHAP code path using a tiny pipeline built on the fly.
"""

from __future__ import annotations

import csv
import importlib
import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import main as api
import train

VALID_PAYLOAD = {
    "tenure": 12,
    "monthly_charges": 89.5,
    "contract_type": 0,
    "support_calls": 3,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "model" in body


def test_predict_returns_frontend_contract(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"High Risk", "Low Risk"}
    assert set(body["feature_importance"]) == {
        "tenure",
        "monthly_charges",
        "contract_type",
        "support_calls",
    }
    assert sum(body["feature_importance"].values()) == pytest.approx(1.0, abs=0.001)
    assert len(body["top_factors"]) == 3
    assert body["explanation"]
    assert body["xai_method"] == "coefficients"
    assert body["model_name"]


def test_predict_accepts_optional_telco_fields(client: TestClient) -> None:
    payload = {
        **VALID_PAYLOAD,
        "contract": "Two year",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "total_charges": 1000.0,
    }
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_accepts_every_telco_field(client: TestClient) -> None:
    payload = {
        **VALID_PAYLOAD,
        "contract": "One year",
        "total_charges": 850.0,
        "gender": "Female",
        "senior_citizen": True,
        "partner": "Yes",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "Yes",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
    }
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_PAYLOAD, "tenure": -1},
        {**VALID_PAYLOAD, "monthly_charges": -5},
        {**VALID_PAYLOAD, "contract_type": 7},
        {**VALID_PAYLOAD, "support_calls": -2},
        {**VALID_PAYLOAD, "contract": "Weekly"},
        {**VALID_PAYLOAD, "internet_service": "Carrier pigeon"},
        {**VALID_PAYLOAD, "tech_support": "Maybe"},
        {**VALID_PAYLOAD, "senior_citizen": "perhaps"},
    ],
)
def test_predict_validates_input(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_drift_status_shape(client: TestClient) -> None:
    response = client.get("/drift-status")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["drift_detected"], bool)
    assert isinstance(body["drift_count"], int)


def test_model_info_reports_fallback(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["model_kind"] == "fallback"
    assert body["artifact_present"] is False
    assert body["xai_method"] == "coefficients"


def test_retrain_without_logs_returns_404(client: TestClient) -> None:
    if api.PREDICTIONS_CSV_PATH.exists():
        api.PREDICTIONS_CSV_PATH.unlink()
    response = client.post("/retrain")
    assert response.status_code == 404


def test_retrain_with_logs_succeeds(client: TestClient) -> None:
    api.PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with api.PREDICTIONS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "tenure",
                "monthly_charges",
                "contract_type",
                "support_calls",
                "churn_probability",
                "prediction",
                "model_name",
            ]
        )
        for index in range(10):
            writer.writerow(
                [
                    "2026-01-01T00:00:00+00:00",
                    10 + index,
                    50.0 + index,
                    index % 3,
                    index % 4,
                    0.5,
                    "High Risk" if index % 2 else "Low Risk",
                    "baseline-logistic-regression",
                ]
            )

    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "model retrained successfully"


def _sample_frame() -> pd.DataFrame:
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    rows = []
    for index in range(90):
        rows.append(
            {
                "tenure": (index % 70) + 1,
                "MonthlyCharges": 30.0 + (index % 40),
                "TotalCharges": float(((index % 70) + 1) * (30.0 + (index % 40))),
                "Contract": contracts[index % 3],
                "InternetService": internet[index % 3],
                "PaymentMethod": "Electronic check" if index % 2 else "Mailed check",
                "Churn": 1 if index % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def _write_artifacts(directory) -> tuple[pd.DataFrame, pd.Series, object, dict]:
    frame, _ = train.clean_dataset(_sample_frame())
    features, target, numeric, categorical = train.split_features(frame)

    pipeline = train.build_pipeline(
        RandomForestClassifier(n_estimators=5, random_state=42), numeric, categorical
    )
    pipeline.fit(features, target)

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "churn_pipeline.joblib")
    metadata = {
        "trained_at": "2026-01-01T00:00:00+00:00",
        "model_name": "RandomForestClassifier",
        "engine": "sklearn",
        "raw_features": numeric + categorical,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_defaults": train.feature_defaults(features, numeric),
        "reference_profile": train.reference_profile(pipeline, features, target, numeric),
        "training_stats": train.training_statistics(features, numeric),
        "top_features": [{"feature": "tenure", "importance": 0.5}],
        "metrics": {"automl": {"f1": 0.8}},
        "leakage_checks": [],
        "fairness": [],
    }
    (directory / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return features, target, pipeline, metadata


def test_automl_artifact_serves_shap_explanations(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    _write_artifacts(model_dir)
    monkeypatch.setenv("CHURNGUARD_MODEL_DIR", str(model_dir))

    reloaded = importlib.reload(api)
    with TestClient(reloaded.app) as client:
        info = client.get("/model-info").json()
        assert info["model_kind"] == "automl"
        assert info["xai_method"] == "shap"

        response = client.post(
            "/predict",
            json={
                "tenure": 2,
                "monthly_charges": 95.0,
                "contract_type": 0,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
            },
        )
        assert response.status_code == 200
        body = response.json()

    assert body["xai_method"] == "shap"
    assert body["model_name"] == "RandomForestClassifier"
    assert len(body["top_factors"]) == 3
    assert "SHAP drivers" in body["explanation"]
    assert sum(body["feature_importance"].values()) == pytest.approx(1.0, abs=0.01)
    # Every reported feature must be a real input column of the trained model.
    assert set(body["feature_importance"]) <= set(
        reloaded.app.state.metadata["raw_features"]
    )
    # Unspecified inputs are reported back so the caller knows what was assumed.
    assert set(body["assumed_features"]) == {"TotalCharges", "PaymentMethod"}

    with TestClient(reloaded.app) as client:
        retrain = client.post("/retrain")
    assert retrain.status_code == 409
    assert "train.py" in retrain.json()["error"]


# ---------------------------------------------------------------------------
# Serving-path guarantees: defaults, coherence, exact rebuilds
# ---------------------------------------------------------------------------
def test_profile_coherence_helper() -> None:
    without_internet = {
        "InternetService": "No",
        "PhoneService": "No",
        "OnlineSecurity": "Yes",
        "StreamingTV": "No",
        "MultipleLines": "Yes",
    }
    api._enforce_profile_coherence(without_internet)
    assert without_internet["OnlineSecurity"] == "No internet service"
    assert without_internet["StreamingTV"] == "No internet service"
    assert without_internet["MultipleLines"] == "No phone service"

    with_internet = {
        "InternetService": "DSL",
        "PhoneService": "Yes",
        "OnlineSecurity": "No internet service",
        "MultipleLines": "No phone service",
    }
    api._enforce_profile_coherence(with_internet)
    assert with_internet["OnlineSecurity"] == "No"
    assert with_internet["MultipleLines"] == "No"


def test_reference_profile_picks_the_row_nearest_the_base_rate() -> None:
    """The reference row is the observation closest to the portfolio churn rate."""
    features = pd.DataFrame(
        {"tenure": [3.0, 12.0, 60.0], "Contract": ["Month-to-month", "One year", "Two year"]}
    )
    target = pd.Series([1, 0, 0])

    class _StubPipeline:
        def predict_proba(self, frame):
            probabilities = np.array([0.10, 0.49, 0.90])
            return np.column_stack([1 - probabilities, probabilities])

    profile = train.reference_profile(_StubPipeline(), features, target, ["tenure"])
    assert profile["base_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert profile["predicted_probability"] == pytest.approx(0.49)
    assert profile["values"]["tenure"] == pytest.approx(12.0)
    assert profile["values"]["Contract"] == "One year"


def test_feature_row_falls_back_to_the_reference_profile(tmp_path) -> None:
    output = _write_artifacts(tmp_path / "models")
    features, _, _, metadata = output
    reference = metadata["reference_profile"]["values"]

    payload = api.ChurnRequest(tenure=5, monthly_charges=60.0, contract_type=1)
    row, assumed = api._feature_row(payload, metadata)

    assert row["Contract"].iloc[0] == "One year"
    assert row["tenure"].iloc[0] == pytest.approx(5.0)
    assert row["TotalCharges"].iloc[0] == pytest.approx(300.0)  # derived, not assumed
    for column in ("Partner", "PaperlessBilling", "PaymentMethod"):
        if column in features.columns:
            assert row[column].iloc[0] == reference[column]
    expected_assumed = set(features.columns) - {"tenure", "MonthlyCharges", "Contract"}
    assert set(assumed) == expected_assumed


def test_feature_row_rebuilds_an_observed_row_exactly(tmp_path) -> None:
    features, _, pipeline, metadata = _write_artifacts(tmp_path / "models")
    contract_codes = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    original = features.iloc[[4]]

    payload = api.ChurnRequest(
        tenure=float(original["tenure"].iloc[0]),
        monthly_charges=float(original["MonthlyCharges"].iloc[0]),
        contract_type=contract_codes[original["Contract"].iloc[0]],
        total_charges=float(original["TotalCharges"].iloc[0]),
        internet_service=original["InternetService"].iloc[0],
        payment_method=original["PaymentMethod"].iloc[0],
    )
    rebuilt, assumed = api._feature_row(payload, metadata)

    assert assumed == {}
    for column in features.columns:
        if column in metadata["numeric_features"]:
            assert float(rebuilt[column].iloc[0]) == pytest.approx(
                float(original[column].iloc[0])
            )
        else:
            assert rebuilt[column].iloc[0] == original[column].iloc[0]
    assert pipeline.predict_proba(rebuilt)[0, 1] == pytest.approx(
        pipeline.predict_proba(original)[0, 1]
    )
