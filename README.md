# BizSentinel

E-commerce business intelligence platform with anomaly detection, customer segmentation, and churn prediction.

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-127-passing)
![Coverage](https://img.shields.io/badge/coverage-77%25-yellow)

## Overview

BizSentinel processes e-commerce transaction data (Olist public dataset, ~100k orders) through three ML modules to produce actionable business intelligence: it flags anomalous transactions and customer behavior, segments customers into business-meaningful groups (champions, at-risk, lost, etc.), and predicts churn probability for every customer. All outputs are served through a REST API, an interactive Dash dashboard, and an MCP server for LLM-based querying.

The three ML modules share a common feature engineering pipeline (RFM + behavioral features) and run sequentially: anomaly scores and segment labels are fed as features into the churn model. This creates a dependency chain where each module builds on the outputs of the previous ones. The supervised churn model uses differential privacy (ε ≤ 2) to protect individual customer records.

From an MLOps perspective, BizSentinel uses Kedro for pipeline DAG management with MLflow experiment tracking, Prefect for flow orchestration (training, inference, monitoring), and containerized deployment via Docker Compose with GitHub Actions CI/CD. Privacy controls (HMAC-SHA256 pseudonymization at ingestion, DP-SGD at training) are built in, not bolted on. SQLAlchemy provides the serving-layer database, and all scoring outputs are stored as parquet files in a layered data catalog.

## Architecture

```
                   Olist CSVs (Kaggle)
                           │
                           ▼
┌──────────────────────────────────────────────┐
│              Kedro Pipeline                  │
│                                              │
│  ┌──────────┐   ┌──────────────┐   ┌──────┐  │
│  │Preprocess │──▶│Feature Eng.  │──▶│Train │  │
│  │· Clean    │   │· RFM         │   │· IF  │  │
│  │· Validate │   │· Reviews     │   │· KMeans  │
│  │· Pseudo-  │   │· Delivery    │   │· LGBM │  │
│  │  nymize   │   │· Payments    │   │· DP   │  │
│  └──────────┘   └──────────────┘   └──────┘  │
│         │              │               │      │
│         ▼              ▼               ▼      │
│         parquet       parquet        parquet  │
└──────────────────────────────────────────────┘
              │                    │
              ▼                    ▼
       MLflow Tracking      Prefect Flows
       (experiments,        (training,
        model registry)      inference,
                             monitoring)
              │
              ▼
┌──────────────────────────────┐
│   SQLite DB (serving layer)  │
│  ┌─────────────────────────┐ │
│  │ anomaly_scores          │ │
│  │ segment_assignments     │ │
│  │ churn_scores            │ │
│  │ alerts                  │ │
│  └─────────────────────────┘ │
└──────────────────────────────┘
              │
    ┌─────────┼────────────┐
    ▼         ▼            ▼
┌──────┐ ┌────────┐ ┌──────────┐
│FastAPI│ │Dash    │ │FastMCP   │
│:8000  │ │:8050   │ │:8080     │
└──────┘ └────────┘ └──────────┘
```

## ML Modules

| Module | Algorithm | Primary Metric | Privacy |
|--------|-----------|---------------|---------|
| Anomaly Detection | Isolation Forest + SHAP | AUC-PR | Pseudonymization |
| Segmentation | K-Means (tuned via silhouette score) | Silhouette Score | Pseudonymization |
| Churn Scoring | LightGBM + SHAP (+ DP baseline) | ROC-AUC | Differential Privacy (ε ≤ 2) |

## Tech Stack

| Category | Technology | Minimum Version |
|----------|-----------|-----------------|
| Language | Python | 3.11 |
| ML / Data | scikit-learn, LightGBM, SHAP, pandas, NumPy | ≥1.5.0 / ≥4.3.0 / ≥0.46.0 |
| Pipeline | Kedro | ≥0.19.5 |
| Tracking | MLflow | ≥2.13.2 |
| Orchestration | Prefect | ≥2.19.0 |
| Privacy | diffprivlib | ≥0.2.1 |
| API | FastAPI + Uvicorn | ≥0.111.0 |
| Dashboard | Dash + Plotly | ≥2.17.0 |
| MCP | FastMCP | ≥0.1.0 |
| Database | SQLAlchemy + SQLite | ≥2.0.30 |
| Auth | python-jose (JWT) | ≥3.3.0 |
| Infrastructure | Docker, Docker Compose | |
| CI/CD | GitHub Actions | |

## Project Structure

```
src/biz_sentinel/
├── domain/models.py              # Pydantic data contracts (Raw → Pseudonymized → ScoredCustomer)
├── pipelines/
│   ├── preprocessing/            # Data cleaning, validation, pseudonymization
│   │   ├── nodes.py              # clean_orders, clean_customers, pseudonymize_customers, build_transactions
│   │   └── pipeline.py           # Kedro pipeline definition
│   ├── feature_engineering/      # RFM, review, delivery, payment features
│   │   ├── nodes.py              # compute_rfm, assemble_feature_matrix, etc.
│   │   └── pipeline.py
│   ├── training/                 # All three ML modules
│   │   ├── nodes.py              # Isolation Forest (anomaly) + SHAP
│   │   ├── segmentation_nodes.py # K-Means clustering + segment labeling
│   │   ├── churn_nodes.py        # LightGBM + DP logistic regression baseline + SHAP
│   │   └── pipeline.py           # Orchestrates all three modules
│   └── inference/                # Placeholder (future batch scoring)
├── flows/
│   ├── training_flow.py          # Prefect flow: preprocessing → FE → training
│   ├── inference_flow.py         # Prefect flow: champion model → scoring → DB
│   └── monitoring_flow.py        # Prefect flow: data drift + score distribution
├── privacy/pseudonymizer.py      # HMAC-SHA256 pseudonymization utilities
├── serving/
│   ├── api/main.py               # FastAPI with JWT auth (4 endpoints)
│   ├── api/database.py           # SQLAlchemy ORM models
│   ├── api/schemas.py            # Pydantic response models
│   ├── dashboard/app.py          # Dash (3 tabs: Overview, Anomalies, Segments)
│   └── mcp/server.py             # FastMCP (4 tools + Ollama integration)
├── scripts/load_scores_to_db.py  # Pipeline outputs → SQLite
├── pipeline_registry.py          # Kedro __default__ pipeline wiring
└── settings.py
├── conf/base/
│   ├── catalog.yml               # Data catalog (CSV → parquet layers)
│   ├── parameters.yml            # All pipeline parameters
│   └── logging.yml
├── docker/
│   ├── Dockerfile                # Multi-stage build
│   ├── docker-compose.yml        # Production services
│   └── docker-compose.dev.yml    # Dev overrides
├── tests/
│   ├── pipelines/                # 93 tests across 5 test files
│   ├── serving/                  # 37 tests (API + MCP)
│   └── flows/                    # 11 tests
└── notebooks/
    └── 01_olist_eda.ipynb
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/anomalyco/biz-sentinel.git
cd biz-sentinel
uv sync

# 2. Download Olist data
# https://www.kaggle.com/olistbr/brazilian-ecommerce
# Place CSVs in data/01_raw/

# 3. Configure environment
cp .env.example .env
# Edit .env: set HMAC_SALT, SECRET_KEY, DATABASE_URL

# 4. Run the full pipeline
kedro run

# 5. Load scores to database
uv run python -m biz_sentinel.scripts.load_scores_to_db

# 6. Start serving (Docker)
docker compose -f docker/docker-compose.yml up
# Or individually:
#   API:       uv run uvicorn biz_sentinel.serving.api.main:app --port 8000
#   Dashboard: uv run python -m biz_sentinel.serving.dashboard.app
#   MCP:       uv run python -m biz_sentinel.serving.mcp.server

# 7. Access
# API:        http://localhost:8000/docs
# Dashboard:  http://localhost:8050
# MCP:        http://localhost:8080
```

## Development

```bash
# Tests
uv run pytest

# Type checking
uv run pyright src/

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# All quality checks in order
ruff check src/ && pyright src/ && pytest
```

## Privacy Design

- **Pseudonymization at ingestion**: All customer IDs are replaced with HMAC-SHA256 hashes before entering the pipeline. Salt is stored separately from the database.
- **Differential privacy at training**: The churn scoring module (LightGBM supervised model) has a DP baseline using diffprivlib Logistic Regression with ε ≤ 2 (validated at evaluation). The anomaly and segmentation modules are unsupervised and use pseudonymization only.
- **API security**: JWT authentication on all endpoints except `/health`. MCP tools return scores and pseudonymized hashes only — no raw customer data is exposed to the LLM or API clients.
- **Full details**: [docs/privacy_design.md](docs/privacy_design.md)

## Deployment Options

| Mode | Implementation | Status |
|------|---------------|--------|
| Batch inference via Prefect flow | `biz_sentinel/flows/training_flow.py` | ✓ Implemented |
| REST API (FastAPI) | `biz_sentinel/serving/api/main.py` | ✓ Implemented |
| Interactive dashboard (Dash) | `biz_sentinel/serving/dashboard/app.py` | ✓ Implemented |
| MCP server + Ollama | `biz_sentinel/serving/mcp/server.py` | ✓ Implemented |

Real-time transaction scoring is stubbed (returns placeholder). Cloud deployment (DigitalOcean) and CD automation are configured (GitHub Actions builds and pushes Docker images) but require manual first-deploy setup.

## Test Coverage

Current: **127 tests** (unit only), **~77% coverage**. Run with:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Coverage reports are generated as HTML in `htmlcov/` and uploaded as artifacts in CI.

## Portfolio Context

This project demonstrates end-to-end ML engineering: data pipeline construction (Kedro), experiment tracking (MLflow), flow orchestration (Prefect), privacy-preserving ML (diffprivlib), model interpretability (SHAP), API design (FastAPI + JWT), interactive visualization (Dash), and LLM tool integration (FastMCP). It is not a production system — the inference pipeline is partially stubbed, there is no real-time scoring, no cloud deployment is active, and no federated learning is implemented. These are natural extension points. What is implemented works end-to-end on the Olist dataset from raw CSV files to a running multi-service application.

## License

MIT
