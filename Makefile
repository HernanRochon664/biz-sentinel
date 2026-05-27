PYTHON = uv run python
KEDRO = export $$(grep -v '^\#' .env | xargs) && uv run kedro
ENV_EXPORT = export $$(grep -v '^\#' .env | xargs)

.DEFAULT_GOAL := help

help:
	@echo "\033[1;32mAvailable targets:\033[0m"
	@echo "\033[1;32m  install\033[0m                Install dependencies"
	@echo "\033[1;32m  install-playwright\033[0m      Install Playwright browser"
	@echo "\033[1;32m  download-data\033[0m           Download Olist dataset from Kaggle"
	@echo "\033[1;32m  pipeline-preprocessing\033[0m  Run preprocessing Kedro pipeline"
	@echo "\033[1;32m  pipeline-features\033[0m       Run feature engineering Kedro pipeline"
	@echo "\033[1;32m  pipeline-training\033[0m       Run training Kedro pipeline"
	@echo "\033[1;32m  pipeline-all\033[0m            Run full Kedro pipeline"
	@echo "\033[1;32m  load-db\033[0m                 Load scores to database"
	@echo "\033[1;32m  api\033[0m                     Start FastAPI server on :8000"
	@echo "\033[1;32m  dashboard\033[0m               Start Dash dashboard on :8050"
	@echo "\033[1;32m  landing\033[0m                 Start landing page on :8055"
	@echo "\033[1;32m  chat\033[0m                    Start AI chat on :8060"
	@echo "\033[1;32m  mcp\033[0m                     Start MCP server (stdio)"
	@echo "\033[1;32m  test\033[0m                    Run unit tests (no integration)"
	@echo "\033[1;32m  test-all\033[0m                Run all tests"
	@echo "\033[1;32m  test-integration\033[0m        Run integration tests only"
	@echo "\033[1;32m  lint\033[0m                    Lint with Ruff"
	@echo "\033[1;32m  format\033[0m                  Format with Ruff"
	@echo "\033[1;32m  typecheck\033[0m               Type check with Pyright"
	@echo "\033[1;32m  quality\033[0m                 Run lint + typecheck"
	@echo "\033[1;32m  coverage\033[0m                Generate HTML coverage report"
	@echo "\033[1;32m  mlflow-ui\033[0m               Start MLflow UI on :5000"
	@echo "\033[1;32m  setup-env\033[0m               Create .env from template"

install:
	uv sync --dev
	@echo "✓ Dependencies installed"

install-playwright:
	uv run playwright install chromium
	@echo "✓ Playwright installed"

# --- Data ---
download-data:
	@echo "Download Olist dataset from Kaggle:"
	@echo "  kaggle datasets download -d olistbr/brazilian-ecommerce"
	@echo "  unzip brazilian-ecommerce.zip -d data/01_raw/"
	@echo "(Manual step — requires Kaggle API key)"

# --- Pipeline ---
pipeline-preprocessing:
	$(KEDRO) run --pipeline preprocessing

pipeline-features:
	$(KEDRO) run --pipeline feature_engineering

pipeline-training:
	$(KEDRO) run --pipeline training

pipeline-all:
	$(KEDRO) run
	@echo "✓ Full pipeline complete"

load-db:
	$(ENV_EXPORT) && $(PYTHON) -m biz_sentinel.scripts.load_scores_to_db
	@echo "✓ Scores loaded to database"

# --- Serving ---
api:
	@echo "Starting BizSentinel API on http://localhost:8000"
	$(ENV_EXPORT) && $(PYTHON) -m uvicorn biz_sentinel.serving.api.main:app \
		--host 0.0.0.0 --port 8000 --reload

dashboard:
	@echo "Starting Dashboard on http://localhost:8050"
	$(ENV_EXPORT) && $(PYTHON) -m biz_sentinel.serving.dashboard.app

landing:
	@echo "Starting Landing page on http://localhost:8055"
	$(ENV_EXPORT) && $(PYTHON) -m biz_sentinel.serving.dashboard.landing

chat:
	@echo "Starting AI Chat on http://localhost:8060"
	$(ENV_EXPORT) && $(PYTHON) -m biz_sentinel.serving.dashboard.chat

mcp:
	@echo "Starting MCP server (stdio)"
	$(ENV_EXPORT) && $(PYTHON) -m biz_sentinel.serving.mcp.server stdio

# --- Dev tools ---
test:
	uv run pytest tests/ -v -m "not integration"

test-all:
	uv run pytest tests/ -v

test-integration:
	uv run pytest tests/ -v -m "integration"

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

typecheck:
	uv run pyright src/

quality: lint typecheck
	@echo "✓ All quality checks passed"

coverage:
	uv run pytest tests/ -m "not integration" --cov=src --cov-report=html
	@echo "✓ Coverage report in htmlcov/"

# --- MLflow ---
mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000"
	$(ENV_EXPORT) && uv run mlflow ui \
		--backend-store-uri sqlite:///data/mlflow.db \
		--host 0.0.0.0 --port 5000

# --- Setup ---
setup-env:
	cp .env.example .env
	@echo "✓ .env created — edit it and set HMAC_SALT and SECRET_KEY"

.PHONY: help install install-playwright download-data \
	pipeline-preprocessing pipeline-features pipeline-training \
	pipeline-all load-db api dashboard landing chat mcp \
	test test-all test-integration lint format typecheck \
	quality coverage mlflow-ui setup-env
