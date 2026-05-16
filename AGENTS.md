# BizSentinel Agent Instructions

## Repository Overview
BizSentinel is an ML Engineering portfolio project for SMEs focused on business intelligence and anomaly detection with privacy-preserving techniques.

Key components:
- Anomaly Detection (unsupervised)
- Customer Segmentation (unsupervised)
- Churn/Risk Scoring (supervised with differential privacy)

## Tech Stack Commands
```bash
# Install dependencies (using uv for speed)
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Linting and formatting (Ruff)
ruff check src/
ruff format src/

# Type checking (Pyright)
pyright src/

# Testing with coverage
pytest --cov=src --cov-report=term-missing

# Run all code quality checks in order
ruff check src/ && pyright src/ && pytest
```

## Expected Project Structure
Based on documentation, the project should follow this structure:
- `src/` - Main source code
- `notebooks/` - Exploratory analysis (Jupyter notebooks only here)
- `conf/` - Kedro configuration files
- `data/` - Local data storage (gitignored)
- `docs/` - Documentation

## Framework Guidelines

### Kedro Pipeline Development
- Separate data, processing, and configuration concerns
- No Jupyter notebooks in production pipeline components
- Use Kedro's declarative pipeline approach

### MLflow Integration
- Track experiments with MLflow for model versioning
- Log parameters, metrics, and artifacts consistently

### Prefect Orchestration
- Use Prefect flows for complex orchestration needs
- Monitor runs through Prefect's observability features

## Privacy Requirements
- Apply differential privacy using diffprivlib for all supervised models
- Use pseudonymization during data ingestion
- Validate privacy parameters (epsilon ≤ 5) during model evaluation

## Docker Workflow
- All services must be containerized
- Use docker-compose for local development
- Single `docker-compose up` should run the entire system locally

## Quality Checks Order
1. Lint with Ruff: `ruff check src/`
2. Format with Ruff: `ruff format src/`
3. Type check with Pyright: `pyright src/`
4. Test with pytest: `pytest --cov=src`

Follow this exact sequence before any commit.

## Branch/PR Workflow
Default branch is `main`. Follow standard GitHub flow for features.

## Python Version
Project requires Python >=3.11 (as specified in pyproject.toml)