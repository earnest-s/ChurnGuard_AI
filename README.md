# ChurnGuard AI

A full-stack customer-churn platform that demonstrates three coursework areas in a
single deployable repository:

| Area | Implementation |
| --- | --- |
| **AutoML** | `backend/train.py` compares candidate pipelines with PyCaret `compare_models` (stratified CV, F1 as the primary metric), tunes the winner, and exports the winning estimator as a portable scikit-learn pipeline. A built-in scikit-learn comparison is used when PyCaret is not installed. |
| **Explainable AI** | SHAP (`TreeExplainer` / `LinearExplainer`) both offline (`backend/reports/shap_*.png`) and online: every `/predict` call returns the signed drivers for *that* customer. |
| **DevOps** | Multi-stage, non-root Docker image with a healthcheck; GitHub Actions runs flake8 + pytest in a blocking `ci` job before the Azure Container Instances deployment; `GET /drift-status`, `GET /model-info` and a prediction log provide monitoring. |
| MLOps | Drift detection (z-score vs. training statistics), model metadata/versioning, structured request logging, offline retraining. |

```
React + Vite (frontend/)
      │  POST /predict
      ▼
FastAPI (backend/main.py) ──serves──▶ models/churn_pipeline.joblib
      │                                     ▲
      │ SHAP per request                    │ exported by
      ▼                                     │
top_factors + explanation          backend/train.py (PyCaret AutoML)
```

---

## Quick start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/docs>.

Without a trained artifact the API boots with a small built-in baseline model, so it
is always demoable. To serve the real AutoML + SHAP model:

```bash
cd backend
pip install -r requirements-train.txt   # PyCaret, matplotlib, xgboost
python train.py                         # ~2-5 min, downloads the Telco dataset
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173
```

`frontend/.env` holds `VITE_API_URL` (defaults to `http://localhost:8000`).

---

## Training (`backend/train.py`)

```bash
python train.py                        # full run: AutoML + tuning + SHAP
python train.py --fast                 # fewer candidates, quick sanity run
python train.py --engine sklearn       # no PyCaret, local model comparison
python train.py --no-download          # use data/telco_churn.csv only
python train.py --allow-extra-estimators   # also try xgboost/lightgbm
```

The script prints, in order:

1. **Dataset summary** - Telco Customer Churn (7 043 rows, 20 columns), identifier
   dropped, 11 blank `TotalCharges` rows removed.
2. **Baseline** - `LogisticRegression` pipeline, 5-fold stratified CV F1 + hold-out F1.
3. **AutoML comparison** - PyCaret `compare_models(sort="F1")`, then a small
   randomized hyper-parameter search on the winner.
4. **Leakage / bias checks** - computed, not asserted: identifier removal, target not
   in features, `TotalCharges ≈ tenure × MonthlyCharges` redundancy test, duplicate
   rows, fit/transform isolation, class balance, plus per-group recall on `gender`
   and `SeniorCitizen`.
5. **SHAP explainability** - global bar chart, beeswarm, and a waterfall for the
   highest-risk customer.
6. **Artifact export**.

| Artifact | Purpose |
| --- | --- |
| `backend/models/churn_pipeline.joblib` | Portable scikit-learn pipeline served by the API (no PyCaret at runtime). |
| `backend/models/model_metadata.json` | Model card: metrics, reference profile, training statistics, global SHAP importance, leakage checks, fairness slices. |
| `backend/models/reference_rows.joblib` | Training rows used to marginalise request fields the caller omits. |
| `backend/reports/shap_summary_bar.png` | Global feature importance (report figure). |
| `backend/reports/shap_beeswarm.png` | Effect direction and spread per feature. |
| `backend/reports/shap_waterfall_high_risk.png` | Single-customer explanation. |
| `backend/reports/metrics.json` | Machine-readable copy of the metadata. |

**Design note.** PyCaret is a training-time dependency only. The winning estimator is
re-instantiated as a plain scikit-learn pipeline (`preprocess` + `model`) and refitted
on the same split, so the container needs only scikit-learn + SHAP, cold starts are
fast, and SHAP attribution is exact. This also keeps the Docker image ~10x smaller
than an image with the full AutoML stack.

---

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Healthcheck (used by the Docker `HEALTHCHECK`). |
| `POST` | `/predict` | Churn probability + prediction + per-customer SHAP drivers. |
| `GET` | `/model-info` | Model card: metrics, SHAP top features, leakage checks, fairness slices. |
| `GET` | `/drift-status` | Whether recent requests drifted from the training distribution. |
| `POST` | `/retrain` | Online retrain from the prediction log (fallback model only). |

Request (only `tenure`, `monthly_charges`, `contract_type` and `support_calls` are
required; `support_calls` is the legacy frontend field):

```json
{
  "tenure": 12,
  "monthly_charges": 89.5,
  "contract_type": 0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check",
  "paperless_billing": "Yes",
  "tech_support": "No"
}
```

`contract_type` maps to `Contract` (`0 → Month-to-month`, `1 → One year`,
`2 → Two year`). Every other Telco column (`gender`, `senior_citizen`, `partner`,
`dependents`, `phone_service`, `multiple_lines`, `internet_service`,
`online_security`, `online_backup`, `device_protection`, `tech_support`,
`streaming_tv`, `streaming_movies`, `paperless_billing`, `payment_method`) is
optional and type-checked against the exact categories seen in training.

Omitted fields are **marginalised over look-alike customers** rather than filled from
a single made-up row: the service scores the 100 training customers closest to the
request in tenure and monthly charges (distance in standard deviations), with the
supplied values applied to every one of them, and averages the results. That
estimates `P(churn | supplied inputs)` and avoids the classic failure where a sparse
request silently inherits one hidden customer archetype. Two cheaper strategies are
demonstrably worse on the Telco data: per-column medians/modes describe a high-risk
archetype (month-to-month + fiber + electronic check predicts ≈0.40 against a 0.27 base
rate), and a single global "typical customer" predicts 0.02 for a brand-new
$95/month customer whose real look-alikes churn at 0.71.

The response reports what was inferred: `assumed_features` lists the inputs you did
not supply (derived columns such as `TotalCharges` are recomputed, not inferred, so
they are not listed) and `aggregation` says whether the score came from
`neighbour_average` (with the number of rows) or from one row (`single_row`, used when
every non-derived input is supplied). If the `reference_rows.joblib` artifact is
absent, the service falls back to one real training customer at the portfolio churn
rate (metadata `reference_profile`).

`total_charges` is derived as `tenure × monthly_charges` when not supplied, and
`No internet service` / `No phone service` values are kept consistent with the
parent service. `support_calls` is accepted for backwards compatibility but is not
part of the Telco dataset, so the AutoML model ignores it.

Response:

```json
{
  "churn_probability": 0.7312,
  "prediction": "High Risk",
  "feature_importance": { "Contract": 0.41, "tenure": 0.22, "InternetService": 0.15 },
  "top_factors": [
    { "feature": "Contract", "contribution": 0.83 },
    { "feature": "tenure", "contribution": -0.45 },
    { "feature": "MonthlyCharges", "contribution": 0.21 }
  ],
  "explanation": "SHAP drivers averaged over 100 similar customers: Contract (+0.830), tenure (-0.450), MonthlyCharges (+0.210). Positive values push the prediction toward churn, negative values away from it.",
  "model_name": "LogisticRegression",
  "xai_method": "shap",
  "assumed_features": ["InternetService", "PaperlessBilling"],
  "aggregation": { "method": "neighbour_average", "rows": 100 }
}
```

`feature_importance` keeps the original response key so the existing frontend works
unchanged; `top_factors`, `explanation`, `model_name`, `xai_method`,
`assumed_features` and `aggregation` are additive.

---

## Tests and lint

```bash
cd backend
pip install -r requirements-dev.txt
flake8 . --count --statistics
pytest -q
```

The suite is hermetic (no network, no pre-trained artifact): it covers the response
contract, input validation, drift/retrain endpoints, the SHAP helper layer
(list / 2-D / 3-D SHAP value normalisation) and the AutoML serving path (a tiny
pipeline is built on the fly).

## Docker

```bash
docker build -t churn-api ./backend
docker run -p 8000:8000 churn-api
curl http://localhost:8000/model-info
```

The image is multi-stage:

* **builder** - `python:3.10-slim` + `build-essential`, installs pinned
  `requirements.txt` into `/opt/venv`.
* **runtime** - fresh `python:3.10-slim`, copies only the virtualenv and the app,
  runs as uid 1001, exposes a Python-based `HEALTHCHECK` (no `curl` needed, so no
  extra apt packages).

## CI/CD

`.github/workflows/deploy.yml`:

* `ci` job (every push + PR): installs `requirements-dev.txt`, runs `flake8`
  (blocking - a style error fails the build) and `pytest`.
* `deploy` job (`needs: ci`, skipped on PRs): builds the image, pushes it to ACR,
  recreates the Azure Container Instance, and writes the public API URL to the
  workflow summary.

Required repository secrets: `ACR_NAME`, `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`.

## Project structure

```
backend/
  main.py                    FastAPI app (AutoML + SHAP serving, drift, retrain)
  train.py                   AutoML comparison, checks, SHAP plots, artifact export
  xai.py                     SHAP helpers (shape normalisation, aggregation)
  conftest.py                pytest path/env setup
  tests/                     API + SHAP helper tests
  Dockerfile                 multi-stage runtime image
  .flake8                    static analysis config
  requirements*.txt          runtime / train / dev dependency sets
  models/                    exported pipeline + model card
  reports/                   SHAP figures + metrics.json
frontend/                    React + Vite single-page UI
docs/REPORT.md               report pack mapped to the rubric
.github/workflows/deploy.yml CI (lint + tests) then CD (ACR + ACI)
```

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `No AutoML artifact found ... serving the synthetic baseline model` | Run `python train.py` (or commit `models/churn_pipeline.joblib`). |
| `POST /retrain` returns 409 | Expected: with the AutoML model, retraining is offline (`train.py`) to avoid training on the model's own predictions. |
| `Could not load AutoML artifacts (...)` | The winning estimator needs a package missing from the runtime image. `pip install -r requirements-train.txt`, or retrain with `--engine sklearn`. |
| `pip install pycaret` resolution error | PyCaret 3.2 pins `scikit-learn<1.3` / `pandas<2.0`. Use `requirements-train.txt` (PyCaret 3.3.2 + scikit-learn 1.4.2 + pandas 2.1.4). |
