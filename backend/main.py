from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
FEATURE_COLUMNS = ["tenure", "monthly_charges", "contract_type", "support_calls"]


class ChurnRequest(BaseModel):
    tenure: float = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    contract_type: int = Field(..., ge=0, le=2)
    support_calls: float = Field(..., ge=0)


class ChurnResponse(BaseModel):
    churn_probability: float
    prediction: Literal["High Risk", "Low Risk"]


def _generate_synthetic_data(rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, size=rows)
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

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "support_calls": support_calls,
            "churn": churn,
        }
    )


def _build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("contract_ohe", OneHotEncoder(handle_unknown="ignore"), ["contract_type"])
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def _train_and_save_model(path: Path) -> Pipeline:
    data = _generate_synthetic_data(rows=1000)
    X = data[FEATURE_COLUMNS]
    y = data["churn"]

    pipeline = _build_model_pipeline()
    pipeline.fit(X, y)

    joblib.dump(pipeline, path)
    return pipeline


def _load_or_create_model(path: Path) -> Pipeline:
    if path.exists():
        return joblib.load(path)
    return _train_and_save_model(path)


app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model = None


@app.on_event("startup")
def startup_event() -> None:
    app.state.model = _load_or_create_model(MODEL_PATH)


@app.get("/")
def healthcheck() -> str:
    return "Churn API Running"


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
    model: Optional[Pipeline] = app.state.model
    if model is None:
        model = _load_or_create_model(MODEL_PATH)
        app.state.model = model

    input_df = pd.DataFrame([payload.model_dump()])
    churn_probability = float(model.predict_proba(input_df)[0][1])
    prediction = "High Risk" if churn_probability >= 0.5 else "Low Risk"

    return ChurnResponse(
        churn_probability=round(churn_probability, 4),
        prediction=prediction,
    )
