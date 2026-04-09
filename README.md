# ChurnGuard AI

Full-stack churn prediction project with a FastAPI backend and React frontend.

## Project Structure

- `backend/`: FastAPI API and logistic regression model training/loading
- `frontend/`: React + Vite single-page app for predictions

## Backend Setup (FastAPI + scikit-learn)

This project supports `uv`-based setup.

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:

- `GET /` -> `"Churn API Running"`
- `POST /predict`

Example request payload:

```json
{
  "tenure": 12,
  "monthly_charges": 89.5,
  "contract_type": 0,
  "support_calls": 3
}
```

## Frontend Setup (React)

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` and sends requests to `http://localhost:8000/predict`.
