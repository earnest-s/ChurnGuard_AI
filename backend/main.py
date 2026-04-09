import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import csv
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, field_validator
from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("churn-api")

FEATURE_NAMES: tuple[str, str, str, str] = (
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_calls",
)
DRIFT_THRESHOLD: float = 2.0
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
PREDICTIONS_CSV_PATH: Path = DATA_DIR / "predictions.csv"
MODEL_VERSION: str = "v1"
MODEL_PATH: Path = MODELS_DIR / f"model_{MODEL_VERSION}.pkl"


class ChurnRequest(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: int
    support_calls: int

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


class FeatureImportance(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: float
    support_calls: float


class ChurnResponse(BaseModel):
    churn_probability: float
    prediction: Literal["High Risk", "Low Risk"]
    feature_importance: FeatureImportance


class DriftStatusResponse(BaseModel):
    drift_detected: bool
    drift_count: int


class RetrainResponse(BaseModel):
    status: str


def _generate_synthetic_data(rows: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    tenure = rng.uniform(1, 72, size=rows)
    monthly_charges = rng.uniform(20, 130, size=rows)
    contract_type = rng.choice([0, 1, 2], size=rows, p=[0.5, 0.3, 0.2])
    support_calls = rng.poisson(lam=2.2, size=rows)

    # Construct a signal with meaningful churn relationships.
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
    y = churn.astype(int)
    return X, y


def _train_model() -> LogisticRegression:
    X, y = _generate_synthetic_data(rows=1000)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


def _normalized_feature_importance(model: LogisticRegression) -> FeatureImportance:
    raw_coefficients: np.ndarray = np.abs(model.coef_[0])
    total: float = float(raw_coefficients.sum())

    if total == 0:
        normalized = np.zeros_like(raw_coefficients)
    else:
        normalized = raw_coefficients / total

    rounded_values: list[float] = [round(float(value), 4) for value in normalized]
    importance_map: dict[str, float] = dict(zip(FEATURE_NAMES, rounded_values))
    return FeatureImportance(**importance_map)


def _log_event(endpoint: str, data: Any, prediction: Any) -> None:
    logger.info("%s - %s - %s", endpoint, data, prediction)


def _ensure_storage_paths() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_baseline_stats(X: np.ndarray) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for idx, feature in enumerate(FEATURE_NAMES):
        mean = float(np.mean(X[:, idx]))
        std = float(np.std(X[:, idx]))
        stats[feature] = (mean, std)
    return stats


def _load_or_train_model() -> LogisticRegression:
    if MODEL_PATH.exists():
        loaded_model = joblib.load(MODEL_PATH)
        if isinstance(loaded_model, LogisticRegression):
            _log_event("/startup", "model_loaded", str(MODEL_PATH))
            return loaded_model

    model = _train_model()
    joblib.dump(model, MODEL_PATH)
    _log_event("/startup", "model_trained_saved", str(MODEL_PATH))
    return model


def _append_prediction_log(
    *,
    payload: ChurnRequest,
    churn_probability: float,
    prediction: str,
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
            ]
        )


def _check_drift(payload: ChurnRequest, baseline_stats: dict[str, tuple[float, float]]) -> bool:
    incoming_values: dict[str, float] = {
        "tenure": payload.tenure,
        "monthly_charges": payload.monthly_charges,
        "contract_type": float(payload.contract_type),
        "support_calls": float(payload.support_calls),
    }

    drift_found = False
    for feature, value in incoming_values.items():
        mean, std = baseline_stats[feature]
        if std == 0:
            is_drift = value != mean
        else:
            is_drift = abs(value - mean) > DRIFT_THRESHOLD * std

        if is_drift:
            drift_found = True
            _log_event(
                "/predict",
                {"feature": feature, "value": value, "mean": round(mean, 4), "std": round(std, 4)},
                "drift_detected",
            )

    return drift_found


def _load_training_data_from_predictions() -> tuple[np.ndarray, np.ndarray]:
    if not PREDICTIONS_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="No prediction data found for retraining")

    rows: list[dict[str, str]] = []
    with PREDICTIONS_CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader]

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
            raise HTTPException(status_code=400, detail=f"Invalid prediction log format: {exc}") from exc

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    if X.size == 0 or y.size == 0:
        raise HTTPException(status_code=400, detail="Prediction data is empty; retraining skipped")

    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        raise HTTPException(
            status_code=400,
            detail="Retraining requires both classes; collect more diverse predictions",
        )

    return X, y


app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model: LogisticRegression | None = None
app.state.baseline_stats: dict[str, tuple[float, float]] = {}
app.state.drift_count = 0
app.state.drift_detected = False


@app.on_event("startup")
def startup_event() -> None:
    _ensure_storage_paths()
    baseline_X, _ = _generate_synthetic_data(rows=1000)
    app.state.baseline_stats = _build_baseline_stats(baseline_X)
    app.state.model = _load_or_train_model()
    app.state.drift_count = 0
    app.state.drift_detected = False
    _log_event("/startup", "baseline_ready", "N/A")


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    endpoint: str = request.url.path
    _log_event(endpoint, "incoming_request", "N/A")
    response: Response = await call_next(request)
    return response


@app.get("/")
def healthcheck() -> dict[str, str]:
    _log_event("/", "healthcheck", "N/A")
    return {"message": "Churn API Running"}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    baseline_stats: dict[str, tuple[float, float]] = app.state.baseline_stats
    if _check_drift(payload, baseline_stats):
        app.state.drift_count += 1
        app.state.drift_detected = True

    payload_data: dict[str, Any] = payload.model_dump()
    _log_event("/predict", payload_data, "pending")

    input_vector = np.array(
        [
            [
                payload.tenure,
                payload.monthly_charges,
                payload.contract_type,
                payload.support_calls,
            ]
        ],
        dtype=float,
    )
    churn_probability = float(model.predict_proba(input_vector)[0][1])
    prediction = "High Risk" if churn_probability >= 0.5 else "Low Risk"
    feature_importance = _normalized_feature_importance(model)
    _append_prediction_log(payload=payload, churn_probability=churn_probability, prediction=prediction)

    _log_event(
        "/predict",
        payload_data,
        {"label": prediction, "probability": round(churn_probability, 4)},
    )

    return ChurnResponse(
        churn_probability=round(churn_probability, 4),
        prediction=prediction,
        feature_importance=feature_importance,
    )


@app.get("/drift-status", response_model=DriftStatusResponse)
def drift_status() -> DriftStatusResponse:
    return DriftStatusResponse(
        drift_detected=bool(app.state.drift_detected),
        drift_count=int(app.state.drift_count),
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain_model() -> RetrainResponse:
    X, y = _load_training_data_from_predictions()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    app.state.model = model
    app.state.baseline_stats = _build_baseline_stats(X)
    app.state.drift_count = 0
    app.state.drift_detected = False
    _log_event("/retrain", {"samples": int(X.shape[0])}, "model_retrained")

    return RetrainResponse(status="model retrained successfully")
