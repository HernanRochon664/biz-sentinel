# BizSentinel System Architecture

## 1. System Components Diagram

```
┌──────────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│   Data Sources       │    │  Kedro Pipelines   │    │   MLflow         │
│                      │───▶│                    │───▶│                  │
│  Olist CSV Files     │    │  Preprocessing     │    │  Experiment      │
│  (Historical)        │    │  Feature Engineering│    │  Tracking        │
│                      │    │  Training          │    │  Model Registry  │
│                      │    │  Inference         │    │                  │
└──────────────────────┘    └────────────────────┘    └──────────────────┘
                                      │                         │
                                      ▼                         ▼
                            ┌────────────────────┐    ┌──────────────────┐
                            │   SQLite/PostgreSQL│    │   Prefect        │
                            │                    │◀──▶│                  │
                            │  Raw Data          │    │  Flow Orchestration│
                            │  Features          │    │                  │
                            │  Scores            │    │  Training Flow   │
                            │  Alerts            │    │  Inference Flow  │
                            │  Model Metadata    │    │  Monitoring Flow │
                            └────────────────────┘    └──────────────────┘
                                      ▲                         │
                                      │                         ▼
                            ┌────────────────────┐    ┌──────────────────┐
                            │   Serving Layer    │    │   LLM Integration│
                            │                    │    │                  │
                            │  FastAPI (API)     │    │  Ollama          │
                            │  Dash (Dashboard)  │    │  FastMCP Server  │
                            │                    │    │                  │
                            └────────────────────┘    └──────────────────┘
                                      ▲                         │
                                      │                         ▼
                            ┌────────────────────┐    ┌──────────────────┐
                            │   CI/CD Pipeline   │    │   Deployment     │
                            │                    │    │                  │
                            │  GitHub Actions    │───▶│  Docker Registry │
                            │                    │    │  DigitalOcean    │
                            └────────────────────┘    └──────────────────┘
```

## 2. Kedro Pipeline Structure

BizSentinel implements four distinct Kedro pipelines that process data from raw CSV files to actionable business insights:

1. **Preprocessing Pipeline**
   - Input: Raw Olist CSV files
   - Process: Data cleaning, validation, pseudonymization
   - Output: Cleaned, privacy-safe data tables

2. **Feature Engineering Pipeline**
   - Input: Cleaned data tables
   - Process: RFM calculation, temporal features, behavioral metrics
   - Output: Feature matrix with engineered features per customer

3. **Training Pipeline**
   - Input: Feature matrix
   - Process: Model training for anomaly detection, segmentation, and churn scoring
   - Output: Trained models registered in MLflow

4. **Inference Pipeline**
   - Input: Latest features + champion models from MLflow
   - Process: Score generation for all three modules
   - Output: Anomaly scores, segment labels, and risk scores stored in database

## 3. Prefect Flows

### Training Flow
- **Trigger**: Volume threshold reached OR manual trigger
- **Tasks**: 
  - Data validation and preprocessing
  - Feature engineering
  - Model training with cross-validation
  - Model registration in MLflow
- **Retry Policy**: 3 retries with exponential backoff
- **Failure Handling**: Alert administrators, rollback to previous model

### Inference Flow
- **Trigger**: Daily schedule (02:00 UTC)
- **Tasks**: 
  - Feature extraction from latest data
  - Model loading from MLflow registry
  - Scoring for all three modules
  - Results storage in database
- **Retry Policy**: 2 retries with 30-minute intervals
- **Failure Handling**: Log errors, notify team, maintain previous scores

### Monitoring Flow
- **Trigger**: Weekly schedule (Sunday 03:00 UTC)
- **Tasks**: 
  - Data drift detection
  - Model performance evaluation
  - Feature importance stability check
  - Alert generation for significant changes
- **Retry Policy**: 1 retry within 24 hours
- **Failure Handling**: Send critical alert to team lead

## 4. MLflow Integration

### Experiment Tracking
- **Naming Convention**: `{module_name}` (experiment name) with runs identified by `{timestamp}_{data_version}`
- **Logged Items**:
  - Parameters: Model hyperparameters, privacy settings
  - Metrics: AUC-PR, ROC-AUC, F1 Score, Silhouette Score
  - Artifacts: Model pickles, SHAP plots, feature importance charts

### Model Management
- **Stages**: Staging → Production → Archived
- **Promotion Logic**: Manual approval for MVP; automated A/B testing planned for future
- **Versioning**: Automatic version increment on successful training runs

## 5. Storage Schema

### Database Tables
1. **customers**: Pseudonymized customer records
2. **transactions**: Order-level data with privacy-safe identifiers
3. **features**: Computed feature matrix per customer per week
4. **anomaly_scores**: Module A outputs (transaction/customer anomaly scores)
5. **segment_labels**: Module B outputs (customer cluster assignments)
6. **risk_scores**: Module C outputs (churn/default probability scores)
7. **alerts**: Triggered alerts with threshold, score, and timestamp
8. **exchange_rates**: Currency conversion rates (if needed for normalization)
9. **model_metadata**: Champion model info (mirrors MLflow for quick access)

## 6. CI/CD Pipeline

### GitHub Actions Workflows
- **ci.yml**:
  - Trigger: On every pull request
  - Actions: Ruff linting, Pyright type checking, Pytest unit/integration tests
  - Artifacts: Test coverage report, code quality metrics

- **cd.yml**:
  - Trigger: On merge to main branch
  - Actions: Docker image build, push to registry, deploy to DigitalOcean
  - Artifacts: Deployed application version, infrastructure state

## 7. Docker Compose Services

### Development Profile
- app: Kedro + Prefect worker container
- mlflow: MLflow tracking server
- dashboard: Dash application for visualization
- api: FastAPI server for REST endpoints
- mcp: FastMCP server for LLM agent integration
- db: SQLite database (development only)

### Production Profile
- All services above with:
- db: PostgreSQL database (replaces SQLite)
- ollama: Ollama service for local LLM inference

## 8. Directory Structure

```
biz-sentinel/
├── .github/workflows/
├── conf/                    # Kedro configuration files
├── data/                    # Local data (gitignored)
├── docs/                    # Documentation files
├── notebooks/               # Exploratory notebooks
├── src/biz_sentinel/
│   ├── pipelines/
│   │   ├── preprocessing/   # Data cleaning and validation
│   │   ├── feature_engineering/  # Feature computation
│   │   ├── training/        # Model training workflows
│   │   └── inference/       # Scoring pipelines
│   ├── domain/              # Pydantic models and data contracts
│   ├── privacy/             # Pseudonymization and DP utilities
│   ├── serving/
│   │   ├── api/             # FastAPI server code
│   │   ├── dashboard/       # Dash application
│   │   └── mcp/             # FastMCP server implementation
│   └── monitoring/          # Monitoring and alerting code
├── tests/                   # Unit and integration tests
├── docker/                  # Docker configurations
├── deployment/              # Deployment scripts and infrastructure
├── README.md
├── PROJECT.md
├── STACK.md
├── pyproject.toml
└── .env.example
```