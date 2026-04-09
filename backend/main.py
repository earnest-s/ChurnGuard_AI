import logging
from typing import Any, Awaitable, Callable, Literal

import numpy as np
from fastapi import FastAPI, Request
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


app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model = _train_model()


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    endpoint: str = request.url.path
    logger.info("%s - data=%s - prediction=%s", endpoint, "incoming_request", "N/A")
    response: Response = await call_next(request)
    return response


@app.get("/")
def healthcheck() -> dict[str, str]:
    logger.info("%s - data=%s - prediction=%s", "/", "healthcheck", "N/A")
    return {"message": "Churn API Running"}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
    model: LogisticRegression = app.state.model
    payload_data: dict[str, Any] = payload.model_dump()
    logger.info("%s - data=%s - prediction=%s", "/predict", payload_data, "pending")

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

    logger.info(
        "%s - data=%s - prediction=%s",
        "/predict",
        payload_data,
        {"label": prediction, "probability": round(churn_probability, 4)},
    )

    return ChurnResponse(
        churn_probability=round(churn_probability, 4),
        prediction=prediction,
        feature_importance=feature_importance,
    )
