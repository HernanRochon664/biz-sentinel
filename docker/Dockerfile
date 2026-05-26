# Stage 1 — builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel

# Stage 2 — runtime
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash biz_sentinel

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    kedro>=1.3.0 \
    kedro-datasets>=3.0.0 \
    prefect>=2.19.0 \
    mlflow>=2.13.2 \
    fastapi>=0.111.0 \
    uvicorn>=0.30.1 \
    dash>=2.17.0 \
    plotly>=5.22.0 \
    fastmcp>=0.1.0 \
    lightgbm>=4.3.0 \
    shap>=0.46.0 \
    ollama>=0.4.0 \
    diffprivlib>=0.6.0

COPY conf/ ./conf/
COPY src/ ./src/

RUN mkdir -p data logs && \
    chown -R biz_sentinel:biz_sentinel /app

USER biz_sentinel

EXPOSE 8000 8050 8055 8060

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1