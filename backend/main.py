"""ChurnGuard AI - FastAPI service.

Serves the model produced by ``train.py`` (AutoML winner exported as a portable
scikit-learn pipeline) together with per-request SHAP explanations, drift
monitoring and structured request logging.

``/predict`` keeps the original response contract (``churn_probability``,
``prediction``, ``feature_importance``) so the existing React frontend keeps
working, and adds ``top_factors`` / ``explanation`` / ``xai_method``.

If no trained artifact is present the service falls back to an in-process
synthetic baseline model, so the API (and the CI test suite) always boots.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.linear_model import LogisticRegression

from xai import (
    aggregate_to_raw_features,
    build_explainer,
    explain_matrix,
    normalized_absolute,
    top_factors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("churn-api")

BASE_DIR: Path = Path(__file__).resolve().parent


def _env_path(variable: str, default: Path) -> Path:
    raw = os.getenv(variable)
    return Path(raw).expanduser() if raw else default


MODEL_DIR: Path = _env_path("CHURNGUARD_MODEL_DIR", BASE_DIR / "models")
DATA_DIR: Path = _env_path("CHURNGUARD_DATA_DIR", BASE_DIR / "data")
PIPELINE_PATH: Path = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH: Path = MODEL_DIR / "model_metadata.json"
PREDICTIONS_CSV_PATH: Path = DATA_DIR / "predictions.csv"

LEGACY_FEATURES: tuple[str, str, str, str] = (
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_calls",
)
DRIFT_THRESHOLD: float = 2.0
CONTRACT_BY_CODE: dict[int, str] = {0: "Month-to-month", 1: "One year", 2: "Two year"}
#: These services only exist when the customer has internet / phone service.
INTERNET_DEPENDENT_FEATURES: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
#: Metadata uses the original Telco column names; the API uses snake_case.
DRIFT_FEATURE_ALIASES: dict[str, str] = {
    "tenure": "tenure",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
}


class ChurnRequest(BaseModel):
    """Input for a single prediction.

    ``tenure`` / ``monthly_charges`` / ``contract_type`` / ``support_calls`` are
    the original frontend fields and are always required. Every other Telco
    column is optional: whatever the caller leaves out is filled from the
    model's *reference profile* (a real training customer at the portfolio
    churn rate) and reported back in ``assumed_features``.
    """

    tenure: float
    monthly_charges: float
    contract_type: int
    support_calls: int = Field(default=0, description="Legacy field; unused by the Telco model.")
    total_charges: float | None = None
    contract: Literal["Month-to-month", "One year", "Two year"] | None = None
    internet_service: Literal["DSL", "Fiber optic", "No"] | None = None
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] | None = None
    gender: Literal["Female", "Male"] | None = None
    senior_citizen: bool | None = None
    partner: Literal["Yes", "No"] | None = None
    dependents: Literal["Yes", "No"] | None = None
    phone_service: Literal["Yes", "No"] | None = None
    multiple_lines: Literal["Yes", "No", "No phone service"] | None = None
    online_security: Literal["Yes", "No", "No internet service"] | None = None
    online_backup: Literal["Yes", "No", "No internet service"] | None = None
    device_protection: Literal["Yes", "No", "No internet service"] | None = None
    tech_support: Literal["Yes", "No", "No internet service"] | None = None
    streaming_tv: Literal["Yes", "No", "No internet service"] | None = None
    streaming_movies: Literal["Yes", "No", "No internet service"] | None = None
    paperless_billing: Literal["Yes", "No"] | None = None

    @field_validator("tenure")
    @classmethod
    def validate_tenure(cls, value: float) -> float:
        if value < 0:
            raise ValueError("tenure must be greater than or equal to 0")
        return value

    @field_validator("monthly_charges")
    @classmethod
    def validate_monthly_charges(cls, value: float) -> float:
        if value < 0:
            raise ValueError("monthly_charges must be greater than or equal to 0")
        return value

    @field_validator("contract_type")
    @classmethod
    def validate_contract_type(cls, value: int) -> int:
        if value not in (0, 1, 2):
            raise ValueError("contract_type must be one of: 0, 1, 2")
        return value

    @field_validator("support_calls")
    @classmethod
    def validate_support_calls(cls, value: int) -> int:
        if value < 0:
            raise ValueError("support_calls must be greater than or equal to 0")
        return value

    @field_validator("total_charges")
    @classmethod
    def validate_total_charges(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("total_charges must be greater than or equal to 0")
        return value


class Factor(BaseModel):
    feature: str
    contribution: float


class ChurnResponse(BaseModel):
    # ``model_name`` would otherwise collide with pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float
    prediction: Literal["High Risk", "Low Risk"]
    feature_importance: dict[str, float]
    top_factors: list[Factor] = Field(default_factory=list)
    explanation: str
    model_name: str
    xai_method: str
    assumed_features: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw model inputs the caller did not supply, filled from the reference profile.",
    )


class DriftStatusResponse(BaseModel):
    drift_detected: bool
    drift_count: int


class RetrainResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Fallback model (used when no AutoML artifact has been trained yet)
# ---------------------------------------------------------------------------
def _generate_synthetic_data(rows: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    tenure = rng.uniform(1, 72, size=rows)
    monthly_charges = rng.uniform(20, 130, size=rows)
    contract_type = rng.choice([0, 1, 2], size=rows, p=[0.5, 0.3, 0.2])
    support_calls = rng.poisson(lam=2.2, size=rows)

    logit = (
        -1.7
        - 0.03 * tenure
        + 0.025 * monthly_charges
        + 0.45 * support_calls
        + 1.05 * (contract_type == 0)
        + 0.35 * (contract_type == 1)
        + rng.normal(0, 0.45, size=rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = (rng.random(rows) < probability).astype(int)

    X = np.column_stack((tenure, monthly_charges, contract_type, support_calls)).astype(float)
    return X, churn.astype(int)


def _train_fallback_model() -> LogisticRegression:
    X, y = _generate_synthetic_data(rows=1000)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


def _coefficient_importance(model: LogisticRegression) -> dict[str, float]:
    raw = np.abs(model.coef_[0])
    total = float(raw.sum())
    normalized = np.zeros_like(raw) if total == 0 else raw / total
    return {
        name: round(float(value), 4) for name, value in zip(LEGACY_FEATURES, normalized)
    }


def _fallback_stats(X: np.ndarray) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for index, feature in enumerate(LEGACY_FEATURES):
        stats[feature] = (float(np.mean(X[:, index])), float(np.std(X[:, index])))
    return stats


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def _load_artifacts() -> tuple[Any | None, dict[str, Any] | None]:
    """Load the portable pipeline + metadata written by ``train.py``."""
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(
            "Could not load AutoML artifacts (%s: %s). Falling back to the baseline model. "
            "Re-run train.py in an environment that has every dependency of the winning model.",
            type(exc).__name__,
            exc,
        )
        return None, None
    return pipeline, metadata


def _explainer_for(pipeline: Any) -> Any | None:
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        n_features = len(preprocessor.get_feature_names_out())
    except Exception:
        return None
    return build_explainer(estimator, n_features)


def _metadata_stats(metadata: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for feature, values in (metadata or {}).get("training_stats", {}).items():
        alias = DRIFT_FEATURE_ALIASES.get(feature)
        if alias:
            stats[alias] = (float(values["mean"]), float(values["std"]))
    return stats


def _ensure_storage_paths() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature assembly + explanations
# ---------------------------------------------------------------------------
def _total_charges(payload: ChurnRequest) -> float:
    if payload.total_charges is not None:
        return float(payload.total_charges)
    # Telco's TotalCharges is (almost exactly) tenure x MonthlyCharges.
    return round(float(payload.tenure) * float(payload.monthly_charges), 2)


def _supplied_features(payload: ChurnRequest) -> dict[str, Any]:
    """Raw column -> value for every model input the caller actually sent."""
    supplied: dict[str, Any] = {
        "tenure": float(payload.tenure),
        "MonthlyCharges": float(payload.monthly_charges),
        "Contract": payload.contract or CONTRACT_BY_CODE[payload.contract_type],
    }
    if payload.total_charges is not None:
        supplied["TotalCharges"] = float(payload.total_charges)

    optional: dict[str, Any] = {
        "gender": payload.gender,
        "SeniorCitizen": (
            None if payload.senior_citizen is None else ("1" if payload.senior_citizen else "0")
        ),
        "Partner": payload.partner,
        "Dependents": payload.dependents,
        "PhoneService": payload.phone_service,
        "MultipleLines": payload.multiple_lines,
        "InternetService": payload.internet_service,
        "OnlineSecurity": payload.online_security,
        "OnlineBackup": payload.online_backup,
        "DeviceProtection": payload.device_protection,
        "TechSupport": payload.tech_support,
        "StreamingTV": payload.streaming_tv,
        "StreamingMovies": payload.streaming_movies,
        "PaperlessBilling": payload.paperless_billing,
        "PaymentMethod": payload.payment_method,
    }
    supplied.update({column: value for column, value in optional.items() if value is not None})
    return supplied


def _enforce_profile_coherence(row: dict[str, Any]) -> None:
    """Keep dependent services consistent with the parent service.

    The Telco data only contains ``No internet service`` when
    ``InternetService == "No"`` and ``No phone service`` when
    ``PhoneService == "No"``. A caller (or a stale default) can otherwise
    assemble a combination that never occurs, which the model would treat as an
    unseen category.
    """
    internet = str(row.get("InternetService", ""))
    for column in INTERNET_DEPENDENT_FEATURES:
        if column not in row:
            continue
        if internet == "No":
            row[column] = "No internet service"
        elif str(row[column]) == "No internet service":
            row[column] = "No"

    if "MultipleLines" in row:
        if str(row.get("PhoneService", "")) == "No":
            row["MultipleLines"] = "No phone service"
        elif str(row["MultipleLines"]) == "No phone service":
            row["MultipleLines"] = "No"


def _feature_row(
    payload: ChurnRequest, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one model row from the reference profile plus the caller's values.

    Returns the DataFrame *and* the raw features that were assumed, so the API
    can tell the caller exactly which inputs the explanation is conditioned on.
    """
    reference: dict[str, Any] = (
        (metadata.get("reference_profile") or {}).get("values")
        or metadata.get("feature_defaults")
        or {}
    )
    raw_features = list(metadata.get("raw_features") or reference.keys())
    if not raw_features:
        raise ValueError("Model metadata does not list the raw feature columns.")

    missing = [column for column in raw_features if column not in reference]
    if missing:
        raise ValueError(f"No reference value available for: {', '.join(missing)}")

    supplied = _supplied_features(payload)
    row: dict[str, Any] = {column: reference[column] for column in raw_features}
    row.update(supplied)
    if "TotalCharges" in row:
        row["TotalCharges"] = _total_charges(payload)
    _enforce_profile_coherence(row)

    # Training cast every categorical column to str; mirror that exactly here.
    for column in metadata.get("categorical_features", []):
        if column in row:
            row[column] = str(row[column])
    for column in metadata.get("numeric_features", []):
        if column in row:
            row[column] = float(row[column])

    assumed = {column: row[column] for column in raw_features if column not in supplied}
    return pd.DataFrame([row], columns=raw_features), assumed


def _drift_values(payload: ChurnRequest) -> dict[str, float]:
    return {
        "tenure": float(payload.tenure),
        "monthly_charges": float(payload.monthly_charges),
        "total_charges": float(_total_charges(payload)),
        "contract_type": float(payload.contract_type),
        "support_calls": float(payload.support_calls),
    }


def _check_drift(
    values: dict[str, float],
    baseline_stats: dict[str, tuple[float, float]],
) -> bool:
    drift_found = False
    for feature, value in values.items():
        if feature not in baseline_stats:
            continue
        mean, std = baseline_stats[feature]
        is_drift = (value != mean) if std == 0 else abs(value - mean) > DRIFT_THRESHOLD * std
        if is_drift:
            drift_found = True
            _log_event(
                "/predict",
                {"feature": feature, "value": value, "mean": round(mean, 4), "std": round(std, 4)},
                "drift_detected",
            )
    return drift_found


def _explain_text(factors: list[dict[str, Any]], method: str) -> str:
    if not factors:
        return "No explanation available for this model."
    parts = [
        f"{factor['feature']} ({factor['contribution']:+.3f})" for factor in factors
    ]
    lead = "SHAP drivers" if method == "shap" else "Model coefficients"
    return (
        f"{lead} for this customer: " + ", ".join(parts) + ". "
        "Positive values push the prediction toward churn, negative values away from it."
    )


def _explain_automl(
    pipeline: Any,
    explainer: Any | None,
    row: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], str]:
    """Per-request SHAP values aggregated back to the raw input features."""
    if explainer is None:
        return {}, [], "unavailable"
    preprocessor = pipeline.named_steps["preprocess"]
    matrix = np.asarray(preprocessor.transform(row), dtype=float)
    values = explain_matrix(explainer, matrix)[0]
    transformed_names = list(preprocessor.get_feature_names_out())
    contributions = aggregate_to_raw_features(
        values, transformed_names, metadata.get("raw_features") or []
    )
    factors = top_factors(contributions, limit=3)
    return normalized_absolute(contributions), factors, "shap"


# ---------------------------------------------------------------------------
# Logging / storage
# ---------------------------------------------------------------------------
def _log_event(endpoint: str, data: Any, prediction: Any) -> None:
    logger.info("%s - %s - %s", endpoint, data, prediction)


def _append_prediction_log(
    *,
    payload: ChurnRequest,
    churn_probability: float,
    prediction: str,
    model_name: str,
) -> None:
    file_exists = PREDICTIONS_CSV_PATH.exists()
    with PREDICTIONS_CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
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
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                payload.tenure,
                payload.monthly_charges,
                payload.contract_type,
                payload.support_calls,
                round(churn_probability, 4),
                prediction,
                model_name,
            ]
        )


def _load_retrain_data() -> tuple[np.ndarray, np.ndarray]:
    if not PREDICTIONS_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="No prediction data found for retraining")

    with PREDICTIONS_CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise HTTPException(status_code=400, detail="Prediction data is empty; retraining skipped")

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    for row in rows:
        try:
            X_rows.append(
                [
                    float(row["tenure"]),
                    float(row["monthly_charges"]),
                    float(row["contract_type"]),
                    float(row["support_calls"]),
                ]
            )
            y_rows.append(1 if row["prediction"] == "High Risk" else 0)
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid prediction log format: {exc}"
            ) from exc

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    if X.size == 0 or y.size == 0:
        raise HTTPException(status_code=400, detail="Prediction data is empty; retraining skipped")
    if np.unique(y).size < 2:
        raise HTTPException(
            status_code=400,
            detail="Retraining requires both classes; collect more diverse predictions",
        )
    return X, y


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_storage_paths()
    pipeline, metadata = _load_artifacts()

    if pipeline is not None and metadata is not None:
        app.state.kind = "automl"
        app.state.pipeline = pipeline
        app.state.metadata = metadata
        app.state.model_name = str(metadata.get("model_name", "automl"))
        app.state.explainer = _explainer_for(pipeline)
        app.state.baseline_stats = _metadata_stats(metadata)
        logger.info(
            "Loaded AutoML pipeline '%s' (trained %s) with %s explainer",
            app.state.model_name,
            metadata.get("trained_at", "unknown"),
            "a SHAP" if app.state.explainer is not None else "no",
        )
    else:
        logger.warning(
            "No AutoML artifact found in %s - serving the synthetic baseline model. "
            "Run 'python train.py' for the AutoML + SHAP model.",
            MODEL_DIR,
        )
        X, _ = _generate_synthetic_data(rows=1000)
        app.state.kind = "fallback"
        app.state.pipeline = None
        app.state.metadata = None
        app.state.model = _train_fallback_model()
        app.state.model_name = "baseline-logistic-regression"
        app.state.explainer = None
        app.state.baseline_stats = _fallback_stats(X)

    app.state.drift_count = 0
    app.state.drift_detected = False
    yield


app = FastAPI(
    title="ChurnGuard AI: AutoML + XAI + DevOps",
    version="2.0.0",
    description=(
        "Churn prediction served from an AutoML-selected pipeline, explained per request "
        "with SHAP, with drift monitoring and structured logging."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    _log_event(request.url.path, "incoming_request", "N/A")
    return await call_next(request)


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"message": "ChurnGuard AI API is running", "model": app.state.model_name}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest):
    try:
        values = _drift_values(payload)
        if _check_drift(values, app.state.baseline_stats):
            app.state.drift_count += 1
            app.state.drift_detected = True

        if app.state.kind == "automl":
            metadata = app.state.metadata
            row, assumed = _feature_row(payload, metadata)
            probability = float(app.state.pipeline.predict_proba(row)[0][1])
            importance, factors, method = _explain_automl(
                app.state.pipeline, app.state.explainer, row, metadata
            )
        else:
            assumed = {}
            model = app.state.model
            if model is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Model is not ready", "message": "Prediction failed"},
                )
            vector = np.array(
                [
                    [
                        float(payload.tenure),
                        float(payload.monthly_charges),
                        int(payload.contract_type),
                        int(payload.support_calls),
                    ]
                ],
                dtype=float,
            )
            probability = float(model.predict_proba(vector)[0][1])
            importance = _coefficient_importance(model)
            factors = top_factors(importance, limit=3)
            method = "coefficients"

        prediction = "High Risk" if probability >= 0.5 else "Low Risk"
        explanation = _explain_text(factors, method)
        _append_prediction_log(
            payload=payload,
            churn_probability=probability,
            prediction=prediction,
            model_name=app.state.model_name,
        )
        _log_event(
            "/predict",
            payload.model_dump(),
            {"label": prediction, "probability": round(probability, 4), "xai": method},
        )

        return ChurnResponse(
            churn_probability=round(probability, 4),
            prediction=prediction,
            feature_importance=importance,
            top_factors=[Factor(**factor) for factor in factors],
            explanation=explanation,
            model_name=app.state.model_name,
            xai_method=method,
            assumed_features=assumed,
        )
    except Exception as exc:
        logger.exception("Prediction error")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "message": "Prediction failed"},
        )


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Model card: metrics, SHAP importance, leakage checks, dataset details."""
    metadata = app.state.metadata or {}
    return {
        "model_kind": app.state.kind,
        "model_name": app.state.model_name,
        "xai_method": "shap" if app.state.explainer is not None else "coefficients",
        "artifact_present": PIPELINE_PATH.exists(),
        "trained_at": metadata.get("trained_at"),
        "engine": metadata.get("engine"),
        "dataset": metadata.get("dataset"),
        "metrics": metadata.get("metrics"),
        "top_features": metadata.get("top_features"),
        "reference_profile": metadata.get("reference_profile"),
        "feature_defaults": metadata.get("feature_defaults"),
        "leakage_checks": metadata.get("leakage_checks"),
        "fairness": metadata.get("fairness"),
        "versions": metadata.get("versions"),
    }


@app.get("/drift-status", response_model=DriftStatusResponse)
def drift_status() -> DriftStatusResponse:
    return DriftStatusResponse(
        drift_detected=bool(app.state.drift_detected),
        drift_count=int(app.state.drift_count),
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain_model():
    if app.state.kind == "automl":
        message = (
            "The AutoML model is retrained offline by train.py (versioned artifacts), "
            "not from the prediction log. Run 'python train.py' and redeploy."
        )
        return JSONResponse(
            status_code=409,
            content={"error": message, "message": message},
        )

    X, y = _load_retrain_data()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    app.state.model = model
    app.state.baseline_stats = {
        feature: (float(np.mean(X[:, index])), float(np.std(X[:, index])))
        for index, feature in enumerate(LEGACY_FEATURES)
    }
    app.state.drift_count = 0
    app.state.drift_detected = False
    _log_event("/retrain", {"samples": int(X.shape[0])}, "model_retrained")
    return RetrainResponse(status="model retrained successfully")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=10000)
