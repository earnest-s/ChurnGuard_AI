from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression


class ChurnRequest(BaseModel):
    tenure: float = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    contract_type: int = Field(..., ge=0, le=2)
    support_calls: int = Field(..., ge=0)


class ChurnResponse(BaseModel):
    churn_probability: float
    prediction: Literal["High Risk", "Low Risk"]


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


app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model = _train_model()


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"message": "Churn API Running"}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
    model: LogisticRegression = app.state.model

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

    return ChurnResponse(
        churn_probability=round(churn_probability, 4),
        prediction=prediction,
    )
