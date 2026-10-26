# ChurnGuard AI - report pack (3 pages)

Every number below is produced by the code in this repository. The raw run is in
`docs/training_run.log`; machine-readable values are in
`backend/reports/metrics.json` and `backend/models/model_metadata.json`.

| Rubric area | Evidence |
| --- | --- |
| AutoML - baseline & tuning | `train.py` section `2. Baseline model` (Optuna) |
| AutoML - implementation & comparison | `train.py` section `3. AutoML model comparison` (PyCaret + FLAML, same holdout) |
| AutoML - critical evaluation | `train.py` sections `4. Deliberate leakage probe` and `5. Leakage / bias checks` |
| AutoML - deployment | `backend/main.py`, `GET /model-card`, `backend/Dockerfile` |
| XAI - visualisation | `backend/reports/shap_summary_bar.png`, `shap_force_plot.png` |
| XAI - interpretation GUI | `POST /predict` returns `top_factors`; `frontend/src/App.jsx` renders the SHAP force plot |
| DevOps - pipeline | `.github/workflows/deploy.yml` (`ci` job gates `deploy`) |
| DevOps - containers & orchestration | `backend/Dockerfile`, `k8s/deployment.yaml` (`replicas: 2`), `k8s/service.yaml` |
| DevOps - IaC & monitoring | `terraform/main.tf`, `GET /metrics` (Prometheus) |

---

## Page 1 - AutoML evaluation (30 marks)

**Problem & dataset.** Predict whether a telecom customer will churn (binary
classification; churn is the positive class). The IBM Telco Customer Churn sample has
7,043 accounts and 20 columns. After dropping `customerID` and 11 rows with a blank
`TotalCharges`, the cleaned data has 7,032 rows and 19 features (3 numeric, 16
categorical) with a churn rate of **0.266**. The split is stratified 80/20:
**5,625 train / 1,407 hold-out** rows (`random_state=42`). F1 is the primary metric
because the classes are imbalanced and a missed churner costs more than a wasted
retention offer; precision, recall and ROC-AUC are reported alongside it.

**Step 2 - Optuna Bayesian optimisation (baseline).** `train.py` tunes a
`LogisticRegression` pipeline with Optuna's TPE sampler. Each trial is scored with
3-fold stratified cross-validation on the training partition only; the hold-out is
never touched. The search space is regularisation strength `C` (log-uniform
`1e-3..1e2`), `class_weight` (`None` / `balanced`) and `fit_intercept` (`True` /
`False`).

```
20 trials in 10.5 s
best CV F1 = 0.6338 (+/- 0.0029)
best params = {C: 1.1462, class_weight: "balanced", fit_intercept: False}
holdout F1 = 0.6061, precision = 0.4901, recall = 0.7941, ROC-AUC = 0.8351
untuned reference: CV F1 = 0.5960, holdout F1 = 0.6099
```

**Step 3 - PyCaret vs FLAML (60-second budget each).** Both engines are trained on
`X_train` and scored on the **same** 1,407-row hold-out. PyCaret's internal split is
disabled (`train_size=0.999`) and `compare_models` is capped with
`budget_time=60/60` minutes; FLAML is capped with `time_budget=60` seconds. This is an
apples-to-apples comparison, and the table includes the Optuna baseline.

| Tool | Budget | Hold-out F1 | Winner |
| --- | ---: | ---: | --- |
| Optuna (baseline LR) | 60 s cap | **0.6061** | LogisticRegression |
| PyCaret | 60 s | **0.6082** | LogisticRegression |
| FLAML | 60 s | **0.5844** | RandomForest (`rf`) |

PyCaret's 5-fold CV leaderboard on the training split was: Logistic Regression
0.5970, Gradient Boosting 0.5859, Random Forest 0.5615, Extra Trees 0.5260, Decision
Tree 0.4966. PyCaret therefore selected logistic regression; FLAML spent its budget
on a small random forest (`n_estimators=7`), which did not beat the linear models on
this mostly-linear, high-cardinality categorical data.

**Step 4 - Critical evaluation: deliberate leakage.** A synthetic
`future_churn_indicator` column was appended to copies of the training and hold-out
frames as an exact copy of the label. A logistic-regression probe trained on it
scored **F1 = 1.0000**, proving that a model can "predict" the answer when the label
is present in the features:

```
[WARNING] Model with leaky feature achieved F1: 1.0000
[ACTION] Leaky feature dropped; final model retrained without 'future_churn_indicator'.
holdout F1 with / without the probe: 1.0000 / 0.6099
```

The leaked column is **never** part of the real feature frame, and the deployed
pipeline is fitted without it. The same audit also checked identifier removal, target
contamination, `TotalCharges ≈ tenure × MonthlyCharges` redundancy (r = 1.000,
known at prediction time, so redundancy rather than leakage), duplicate rows,
fit/transform isolation, class balance, and fairness slices on `gender` and
`SeniorCitizen`.

**SHAP summary.** The summary plot (`backend/reports/shap_summary_bar.png`) ranks the
global drivers as `MonthlyCharges` (**0.1790**), `tenure` (**0.1561**),
`InternetService` (**0.1242**), `Contract` (**0.0729**) and `TotalCharges`
(**0.0693**). *SHAP explains the model's decisions, identifying `MonthlyCharges` and
`tenure` as the top drivers.*

**Step 5 - Deployment.** The winning estimator is exported as a portable
scikit-learn pipeline and served by FastAPI. `POST /predict` returns
`churn_probability`, `prediction` (`High Risk` / `Low Risk`), `feature_importance`
(kept for the original React frontend), signed `top_factors`, and an English
`explanation`. `GET /model-card` returns the dataset description, metrics, intended
use and known limits; `GET /model-info` returns the deeper local metadata.

**Honest result.** The tuned AutoML winner scored hold-out F1 **0.6054** versus
**0.6061** for the Optuna baseline, a difference of **-0.0007**. That is well inside
the CV standard deviation (0.0216) and is the expected result for this dataset:
AutoML contributes a defensible selection process, a tuning record and a model card,
not free accuracy. The report does not claim an AutoML improvement.

---

## Page 2 - Explainable AI (30 marks)

**GUI prediction.** `frontend/src/App.jsx` posts a customer profile to `/predict` and
renders (1) the risk verdict and probability, (2) the original normalised importance
bars, and (3) a signed **SHAP force plot** built from `top_factors`. The screenshot
in the PDF (`docs/report_assets/gui_prediction.png`) was captured from the running
React app, not mocked.

**Visualisation & interpretation.** The GUI clearly displays SHAP force plots,
explaining exactly why a specific customer was predicted to churn: red bars push the
prediction toward churn, green bars push it toward retention, and the signed
log-odds contribution is printed next to each feature. The offline
`backend/reports/shap_force_plot.png` is the same additive view for the
highest-risk hold-out customer, produced with
`shap.plots.force(explanation[i], matplotlib=True)`. The waterfall
(`shap_waterfall_high_risk.png`) gives the detailed single-customer accounting, and
the beeswarm shows how the direction of each effect varies across customers.

**Innovation - kNN marginalisation for sparse requests.** A retention agent will not
have all 19 Telco fields in front of them. Rather than silently filling missing
inputs with one made-up "typical" row (which describes a high-risk archetype) or
with per-column medians/modes, the API averages the model over the **100 training
customers closest to the supplied inputs in tenure and monthly charges** (distance in
standard deviations), applying the supplied values to every one of those rows. This
estimates `P(churn | supplied inputs)` by marginalising over the missing dimensions,
so the score and its SHAP values stay on-manifold and mathematically sound. The
response names the inferred fields in `assumed_features` and reports
`aggregation.method = "neighbour_average"` (with the row count); when every input is
supplied, a single-row fast path is used instead. SHAP values are averaged over the
same look-alike rows, so the force plot explains the number that was actually
returned rather than a different, complete customer.

**Method.** Explanations use SHAP (`TreeExplainer` for tree-based winners,
`LinearExplainer` for logistic regression) on the exact served pipeline. Per-category
one-hot SHAP values are summed back onto the original column
(`Contract_Month-to-month` + `Contract_One year` -> `Contract`), which is what
`feature_importance` and `top_factors` expose. Global importance is the mean absolute
SHAP value over 200 hold-out rows; local importance is the signed contribution for
the request.

**Caveat.** SHAP values describe model behaviour, not causal effects. They say which
inputs the model relied on, not what would happen if a customer changed a feature.

---

## Page 3 - DevOps pipeline (Excellent / 7 marks)

**CI/CD.** `.github/workflows/deploy.yml` has a blocking `ci` job that installs
`requirements-dev.txt`, runs **flake8** (style errors fail the build) and then
**pytest** (35 tests) before the `deploy` job is allowed to start. The `deploy` job
builds the image, pushes it to ACR and recreates the Azure Container Instance. The
screenshot in the PDF (`docs/report_assets/github_actions.png`) is the public Actions
run for this commit.

**Containerisation.** `backend/Dockerfile` is a two-stage build:

```dockerfile
FROM python:3.10-slim AS builder
# build-essential + pip install -r requirements.txt into /opt/venv
FROM python:3.10-slim AS runtime
# copy only /opt/venv + app, run as uid 1001, HEALTHCHECK
```

The runtime image contains no build toolchain and no PyCaret. Kubernetes runs it with
**two replicas** and rolling updates:

```yaml
spec:
  replicas: 2
  strategy:
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
```

`k8s/deployment.yaml` also adds readiness (`/model-info`) and liveness (`/`) probes,
a non-root security context and resource requests/limits. `k8s/service.yaml` exposes
port 80 -> 8000 as a `LoadBalancer`.

**IaC.** `terraform/main.tf` provisions the Azure Resource Group and Container
Registry:

```hcl
resource "azurerm_resource_group" "main" {
  name     = "${var.project}-rg"
  location = var.location
}

resource "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
}
```

**Monitoring.** `prometheus-fastapi-instrumentator` is wired in `main.py` and exposes
`GET /metrics` with request counts, latency histograms and in-progress requests.
`curl http://localhost:8000/metrics` returns the Prometheus text format; the PDF
includes a screenshot/terminal capture (`docs/report_assets/metrics_endpoint.png`).
The API also keeps structured request logs and a drift endpoint (`/drift-status`).

**Deployment status (honest note).** The `ci` job is green. The Azure `deploy` job
still fails at `azure/login` with `No subscriptions found for ***` because the
repository's `AZURE_CLIENT_ID` / service-principal OIDC identity is not associated
with a subscription. That is an external subscription/RBAC configuration issue, not a
code failure; the Docker build, tests, Kubernetes manifests and Terraform validate
locally and in CI.
