# BizSentinel

*End-to-end ML platform for e-commerce customer intelligence*

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-127-passing)
![Coverage](https://img.shields.io/badge/coverage-82%25-green)

## Overview

BizSentinel is a production-grade ML pipeline that processes 97,896 customers from the Olist Brazilian E-commerce dataset (Sep 2016 – Oct 2018, 25 months), applying three ML modules to detect anomalies, segment customers, and predict churn. The pipeline ingests 7 raw CSV files (customers, orders, items, payments, reviews, products, sellers), cleans and pseudonymizes at ingestion, engineers 9 RFM and behavioral features, then trains and scores all three models in a single Kedro DAG. Outputs are stored as parquet files and loaded into a SQLite serving database for API and dashboard access.

The three modules run sequentially and are architecturally connected: Module A (Isolation Forest) produces anomaly scores per customer, Module B (K-Means, k=3) assigns business segment labels (champions, loyal, at_risk, etc.), and both outputs become features for Module C (LightGBM churn classifier). The final churn model uses 11 features — 9 behavioral + anomaly_score + segment_encoded. This design means the unsupervised modules directly improve the supervised model's predictive power.

From an MLOps perspective, BizSentinel implements Kedro pipelines with MLflow experiment tracking and model registry, Prefect flows for orchestration and monitoring, a FastAPI REST layer with JWT auth, a Dash UI with three views (overview KPIs, anomaly table, segment charts), and a local LLM assistant via FastMCP + Ollama. All services run in Docker containers behind a single `docker compose up`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Data                                                                 │
│  7 Olist CSVs ──▶ Kedro Pipelines ──▶ Parquet Layers ──▶ MLflow     │
│  (data/01_raw/)   preprocessing        (02-07)         experiment    │
│                     feature eng                         registry     │
│                     training                                          │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ load_scores_to_db.py
                                 ▼
                    ┌───────────────────────┐
                    │    SQLite DB           │
                    │  (data/biz_sentinel.db)│
                    │  anomaly_scores        │
                    │  churn_scores          │
                    │  segment_assignments   │
                    │  alerts                │
                    └──────┬────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          │                │                  │
          ▼                ▼                  ▼
   ┌──────────┐    ┌────────────┐    ┌──────────────┐
   │ FastAPI  │    │ Dash       │    │ Dash         │
   │ :8000    │    │ Dashboard  │    │ Landing      │
   │ /health  │    │ :8050      │    │ :8055        │
   │ /anomalies│   │ Overview   │    │               │
   │ /customers│   │ Anomalies  │    │               │
   │ /alerts   │    │ Segments   │    │ Dash Chat    │
   └──────────┘    └────────────┘    │ :8060        │
                                      └──────┬───────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │ FastMCP (stdio)  │
                                   │ 4 tools          │
                                   │ ◄─── Ollama      │
                                   │     qwen2.5-coder│
                                   └─────────────────┘
```

## ML Modules

| Module | Algorithm | Primary Metric | Privacy |
|--------|-----------|---------------|---------|
| Anomaly Detection | Isolation Forest + SHAP | AUC-PR | Pseudonymization |
| Segmentation | K-Means (k=3) | Silhouette Score | Pseudonymization |
| Churn Scoring | LightGBM + SHAP | ROC-AUC | Differential Privacy (ε≤5) |

**Module dependency chain:** Feature matrix (9 RFM/behavioral features) → Module A scores + labels → 11-feature input for Module C.

## Results

| Metric | Value |
|--------|-------|
| Customers processed | 97,896 |
| Anomaly rate | 1.47% (1,440 flagged as anomalous) |
| Churn rate | 70.77% (180-day inactivity threshold) |
| Alerts generated | 69,848 |
| Dataset | Olist Brazilian E-commerce (Sep 2016 – Oct 2018, 25 months) |
| RFM snapshot date | 2018-10-17 |

## Tech Stack

| Category | Technology |
|----------|-----------|
| ML / Data | pandas, scikit-learn, lightgbm, shap, diffprivlib |
| Pipelines | Kedro 1.3.x, Prefect, MLflow |
| Serving | FastAPI, Dash, FastMCP, Ollama |
| Storage | SQLite (dev), PostgreSQL (prod-ready) |
| Infrastructure | Docker, GitHub Actions, uv, Ruff, Pyright |

## Quick Start

```bash
# a) Clone and install
git clone https://github.com/HernanRochon664/biz-sentinel.git
cd biz-sentinel
uv sync --dev

# b) Configure environment
cp .env.example .env
# Edit .env: set HMAC_SALT and SECRET_KEY to random strings

# c) Download data
# Download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Place CSV files in data/01_raw/

# d) Run the pipeline
export $(grep -v '^#' .env | xargs) && uv run kedro run
# Or step by step:
make pipeline-preprocessing
make pipeline-features
make pipeline-training

# e) Load results to database
make load-db

# f) Start all services (Docker)
docker compose -f docker/docker-compose.yml up -d
# Landing page: http://localhost:8055

# g) Or start individually:
make landing    # http://localhost:8055 — start here
make dashboard  # http://localhost:8050
make api        # http://localhost:8000/docs
make chat       # http://localhost:8060 (requires Ollama)

# h) AI Chat (optional)
# Install Ollama: https://ollama.com
ollama pull qwen2.5-coder:7b
make chat
```

## Project Structure

```
biz-sentinel/
├── src/biz_sentinel/
│   ├── pipelines/              # Kedro ML pipelines
│   │   ├── preprocessing/      # Data cleaning + pseudonymization
│   │   ├── feature_engineering/ # RFM + behavioral features
│   │   └── training/           # IsoForest + KMeans + LightGBM
│   ├── serving/
│   │   ├── api/                # FastAPI REST endpoints
│   │   ├── dashboard/          # Dash UI (landing, dashboard, chat)
│   │   └── mcp/                # FastMCP server for LLM agents
│   ├── flows/                  # Prefect orchestration flows
│   ├── privacy/                # Pseudonymization + DP utilities
│   └── scripts/                # load_scores_to_db, ollama_chat
├── tests/                      # 127 unit tests, 82% coverage
├── docs/                       # Architecture, privacy, deployment docs
├── docker/                     # Dockerfile + docker-compose
├── notebooks/                  # EDA notebook (01_olist_eda.ipynb)
├── Makefile                    # All commands
└── conf/                       # Kedro configuration
```

## Live Demo

- **Landing:** https://biz-sentinel-dashboard.onrender.com
- **Dashboard:** https://biz-sentinel-dashboard-7le8.onrender.com
- **AI Chat:** Local only (requires Ollama + qwen2.5-coder:7b)

## Development

```bash
make test       # Run unit tests only (~30s)
make test-all   # Includes integration tests (~9min)
make quality    # Ruff lint + Pyright type check
make coverage   # HTML coverage report (htmlcov/)
make mlflow-ui  # View experiment tracking (http://localhost:5000)
```

## Privacy Design

- **Layer 1 — Pseudonymization at ingestion**: All `customer_id` and `customer_unique_id` values replaced with HMAC-SHA256 hashes before entering the pipeline. Salt is stored in `HMAC_SALT` env var, separate from the database.
- **Layer 2 — Differential Privacy at training**: The churn model is trained with a diffprivlib Logistic Regression baseline at ε=2.0 (≤5 bound, validated at evaluation). Unsupervised modules (anomaly, segmentation) use pseudonymization only.
- **Layer 3 — JWT authentication**: All API endpoints except `/health` require a Bearer JWT token (HS256). Token verified via `python-jose`.
- **Layer 4 — MCP data isolation**: The FastMCP server exposes 4 read-only tools that return scores and segment descriptions only — never raw customer data. The LLM has no access to PII or transaction details.

Full details: [docs/privacy_design.md](docs/privacy_design.md)

## AI Assistant

BizSentinel includes a local AI assistant powered by Ollama and FastMCP. The assistant uses `qwen2.5-coder:7b` to query ML results in natural language — it can summarize anomalies, explain customer risk, describe segment profiles, and recommend business actions. All inference runs locally; no data leaves the machine. Accessible at http://localhost:8060 or via MCP protocol for external LLM clients (Claude Desktop, custom agents).

**Hardware requirements for AI Chat:**
- Minimum: 8GB RAM free (CPU inference, ~2min response time)
- Recommended: GPU with 8GB+ VRAM (5–15s response time)
- Models: qwen2.5-coder:7b (stable), gemma4:e4b (better quality, needs more RAM)

## Test Coverage

127 tests, 82% coverage.

```bash
make test       # unit tests only (~30s)
make test-all   # includes integration tests (~9min)
```

Coverage reports generated as HTML in `htmlcov/` and uploaded as CI artifacts.

## Portfolio Context

This project demonstrates end-to-end pipeline ownership: Kedro DAG management from raw CSVs to scored parquet outputs, MLflow experiment tracking across three model families, Prefect orchestration with retry/caching, privacy-aware ML via HMAC pseudonymization and diffprivlib DP, multiple serving modes (REST API with JWT, Dash dashboard, MCP LLM tools), and production-grade code quality (type annotations, Ruff linting, Pyright, CI/CD). The inference pipeline is a stub awaiting a real-time scoring endpoint; the monitoring flow detects drift but has no automated retraining. Next steps: real-time scoring endpoint, cloud deployment (DigitalOcean), and federated learning for multi-tenant privacy.

## License

MIT
