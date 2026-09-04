# Fraud Detection — End-to-End ML System

[![CI](https://github.com/848vania/FraudDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/848vania/FraudDetection/actions/workflows/ci.yml)
[![Docker](https://github.com/848vania/FraudDetection/actions/workflows/docker.yml/badge.svg)](https://github.com/848vania/FraudDetection/actions/workflows/docker.yml)

A project implementing a complete, production-shaped machine learning system for scoring financial transaction fraud risk — from synthetic data generation through training, a served REST API, a monitoring dashboard, drift detection, and automated retraining, all tracked with MLflow and containerized with Docker.

The goal of this project is to demonstrate MLOps practices end-to-end, not just a model in a notebook: experiment tracking, a model registry with champion/candidate aliasing, cost-aware threshold selection, prediction logging, drift monitoring, a retraining trigger/pipeline, and a deployable API + dashboard.

> **Note:** the dataset is synthetically generated for demonstration purposes. This model is **not** trained on real financial data and should not be used for real-world fraud decisions.

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Model performance](#model-performance)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the API](#using-the-api)
- [ML pipeline](#ml-pipeline)
- [Monitoring & retraining](#monitoring--retraining)
- [Testing](#testing)
- [Limitations](#limitations)
- [Possible extensions](#possible-extensions)

## What it does

- **Scores transactions in real time** — a FastAPI service takes a transaction payload and returns a fraud probability, a risk band (low/medium/high), and an approve/review decision based on a cost-optimized threshold.
- **Tracks every experiment** — training runs, metrics, and model artifacts are logged to MLflow (`mlflow.db` + `mlruns/`), with models registered under aliases (`champion` / `candidate`) in the MLflow Model Registry.
- **Picks a threshold economically, not arbitrarily** — instead of defaulting to 0.5, the decision threshold is chosen by minimizing *expected cost*, where a missed fraud case and a false alarm are weighted differently (configurable false-negative/false-positive costs).
- **Explains individual predictions** — for linear models, each prediction ships with the top contributing features and their direction of effect (`increases_risk` / `decreases_risk`), computed from preprocessed feature values × model coefficients.
- **Logs every prediction** to a SQLite table for downstream monitoring (volume, latency, review rate, high-risk rate, risk-band distribution).
- **Detects drift** between a reference dataset and current traffic, using relative mean shift for numeric features and total variation distance for categorical features and risk-band distributions.
- **Decides when to retrain** — a rules-based trigger checks drift, live monitoring metrics, and labeled performance, and a retraining pipeline trains a candidate model, compares it against the current champion (PR-AUC improvement, expected cost, minimum recall), and promotes it only if it clears the bar.
- **Ships a Streamlit dashboard** with pages for making predictions, reviewing model performance, watching monitoring metrics, and inspecting drift/retraining reports.
- **Runs anywhere via Docker Compose** — API, dashboard, and MLflow UI as three containers on a shared network.

## Architecture

```
                       +------------------------+
                       |      MLflow (5000)     |
                       | tracking + registry UI |
                       |  mlflow.db + mlruns/   |
                       +-----------+------------+
                                   ^
                                   | shared files (dev-time UI,
                                   | not on the live serving path)
                                   |
  +------------------+   POST /predict   +------------------+
  | Streamlit (8501) | ----------------> |  FastAPI (8000)  |
  |   frontend/      | <---------------- |    app/api/      |
  +--------+---------+   JSON response   +--------+---------+
           |                                      |
           | reads                                | loads at startup,
           v                                      | logs each prediction
  +------------------+                    +--------+---------+
  |  data/results/   |                    | artifacts/models/ |
  | (eval, drift &   |                    |    *.joblib +      |
  |  retrain reports)|                    | fraud_predictions.db |
  +------------------+                    +-------------------+
```

FastAPI exposes `/predict`, `/model/info`, `/monitoring/summary`, and `/health`. The Streamlit dashboard's Predict/Monitoring pages call the API, while its Model Performance and Drift & Retraining pages read `data/results/*.json` directly.

The **offline pipeline** (`scripts/*.py`, wrapped by the `Makefile`) generates synthetic data, builds features, trains and evaluates models with MLflow, registers them, and produces the drift/monitoring reports the API and dashboard read.

## Model performance

Baseline model (`logistic_regression`, `configs/baseline_logistic.yaml`), evaluated on the held-out test split at the expected-cost-minimizing threshold:

| Metric | Value |
|---|---|
| ROC-AUC | 0.965 |
| PR-AUC | 0.524 |
| Selected threshold | 0.48 |
| Precision | 0.197 |
| Recall (fraud capture rate) | 0.948 |
| Review rate | 14.7% |

Threshold selection minimizes `false_negative_cost × FN + false_positive_cost × FP` (default: missing a fraud case costs 20× a false alarm), which is why the optimal operating point favors high recall over precision — deliberately catching nearly 95% of fraud at the cost of flagging ~15% of legitimate transactions for review, rather than optimizing for accuracy in isolation. Full threshold sweep: [`data/results/threshold_report.json`](data/results/threshold_report.json).

## Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| ML / data | scikit-learn, pandas, NumPy, SciPy |
| Experiment tracking & registry | MLflow (SQLite backend store, local artifact store), skops model serialization |
| Persistence | SQLAlchemy + SQLite (prediction log) |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

## Project structure

```
app/
├── api/            FastAPI app and routers (predict, model info, monitoring, health)
├── core/            Settings (pydantic-settings) and Pydantic request/response schemas
├── data/             Ingestion, validation, profiling, synthetic data generation, splitting
├── database/         SQLAlchemy engine/session, prediction log CRUD and model
├── features/         Feature schema and sklearn preprocessing (ColumnTransformer)
├── models/            Train, evaluate, threshold selection, predict, explain, MLflow registry/tracking, retrain
└── monitoring/         Prediction logger, monitoring metrics, drift detection, retraining trigger logic

frontend/
├── streamlit_app.py    Dashboard entrypoint
├── pages/                Predict / Model Performance / Monitoring / Drift & Retraining
└── utils.py              API client + local results-file helpers

scripts/                 CLI entrypoints (one per pipeline stage) — see Makefile
configs/                  YAML configs per model/experiment (data paths, hyperparameters, thresholds, registry settings)
docker/                   Dockerfile.api, Dockerfile.frontend, Dockerfile.mlflow
tests/                    pytest suite mirroring the app/ structure
notebooks/                EDA, model experimentation, monitoring analysis
```

## Getting started

### Option A — Docker Compose (recommended)

Brings up the API, dashboard, and MLflow UI together, using the model already trained and committed in this repo (`artifacts/models/baseline_logistic.joblib`).

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

| Service | URL |
|---|---|
| API | http://localhost:8000 (docs at `/docs`) |
| Streamlit dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

Persistent state (`data/`, `artifacts/`, `mlflow.db`, `mlruns/`, `fraud_predictions.db`) is bind-mounted from the repo root, so anything produced by the containers is visible on the host and vice versa.

### Option B — Local Python environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
make install                  # or: pip install -r requirements.txt

make api                      # FastAPI on http://localhost:8000
make frontend                 # Streamlit on http://localhost:8501 (in another terminal)
make mlflow-ui                # MLflow UI on http://localhost:5000 (optional)
```

## Using the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "transaction_amount": 249.99,
        "transaction_hour": 2,
        "merchant_category": "electronics",
        "customer_tenure_days": 30,
        "num_transactions_24h": 6,
        "num_failed_transactions_24h": 2,
        "avg_transaction_amount_30d": 55.0,
        "is_foreign_transaction": true,
        "device_type": "mobile",
        "country": "RO",
        "previous_chargebacks": 1,
        "account_age_days": 45,
        "risk_score_external": 0.72
      }'
```

```json
{
  "fraud_probability": 0.81,
  "risk_band": "high",
  "decision": "review",
  "threshold_used": 0.48,
  "model_name": "baseline_logistic",
  "latency_ms": 12.4
}
```

Other endpoints: `GET /model/info` (registered model metadata and test metrics), `GET /monitoring/summary` (live prediction volume/latency/review rate), `GET /health`.

## ML pipeline

Every stage is a standalone CLI script under `scripts/`, wrapped by `Makefile` targets so the pipeline reads as a sequence of commands:

```bash
make data              # generate synthetic transactions
make validate           # schema/quality validation
make features             # build feature set
make splits                 # train / valid / test split
make reference               # snapshot a reference dataset for drift comparison
make train                    # train + log to MLflow
make evaluate                   # threshold selection + test metrics
make register                     # register the model in MLflow, tag as champion

make drift                        # compare current vs. reference data
make trigger-retrain                # decide whether to retrain
make retrain                          # train a candidate, compare vs. champion, promote if it wins
```

Each config file under `configs/` (e.g. [`baseline_logistic.yaml`](configs/baseline_logistic.yaml), [`retraining_logistic.yaml`](configs/retraining_logistic.yaml)) fully controls one run: data paths, model hyperparameters, the cost-based threshold objective, and registry/promotion rules — so a new model variant is a new config file, not a code change.

## Monitoring & retraining

- **Drift detection** ([`app/monitoring/drift.py`](app/monitoring/drift.py)) compares a reference dataset against current traffic: relative mean shift for numeric features, total variation distance for categorical features, and the same for the prediction and risk-band distributions.
- **Operational monitoring** ([`app/monitoring/metrics.py`](app/monitoring/metrics.py)) summarizes the live prediction log: volume, average latency, review rate, high-risk rate.
- **The retraining trigger** ([`app/monitoring/retraining.py`](app/monitoring/retraining.py)) combines drift, monitoring, and (if available) labeled performance signals into a single decision with explicit reasons (e.g. `overall_drift_detected`, `review_rate_too_high`, `recall_below_threshold`).
- **The retraining pipeline** ([`app/models/retrain.py`](app/models/retrain.py)) only promotes a candidate model to `champion` if it beats the current model by the margins configured under `promotion:` in the retraining config (minimum PR-AUC improvement, maximum expected-cost increase, minimum recall) — it never promotes blindly.

## Testing & CI

```bash
make test          # or: pytest -v
ruff check .        # lint (config in pyproject.toml)
```

The suite (`tests/`) mirrors `app/`'s structure — data validation, feature building, training, thresholding, the MLflow registry/tracking wrappers, prediction, the API layer, the database layer, monitoring, drift, and the retraining pipeline.

Every push and pull request to `main` runs automatically via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): install dependencies, lint with [ruff](https://docs.astral.sh/ruff/), then run the full pytest suite. A separate workflow ([`.github/workflows/docker.yml`](.github/workflows/docker.yml)) builds all three Docker images on every push/PR to catch a broken Dockerfile before it reaches `main`.

## Limitations

- The dataset is synthetic; no real customer or transaction data is used anywhere in this repo.
- Fairness and bias auditing are not implemented.
- Inference currently loads the model from the local `artifacts/` joblib file rather than the MLflow registry by default (see [`app/models/predict.py`](app/models/predict.py)) — the registry-loading path exists and is exercised by training/registration, but isn't yet the default for serving.

## Possible extensions

- Switch live inference to load from the MLflow registry (`models:/fraud-risk-model@champion`) instead of the local artifact, so promoting a new champion doesn't require redeploying the API.
- Add a gradient-boosted model (XGBoost/LightGBM) as a second candidate alongside logistic regression.
- Schedule the retraining trigger/pipeline (currently manual/CLI) with a scheduler such as cron or Airflow.