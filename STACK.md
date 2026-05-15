# BizSentinel Technology Stack

## 1. Philosophy

The BizSentinel stack emphasizes reproducibility, observability, and privacy-first engineering while maintaining portfolio-demonstrable artifacts. We chose tools that balance production-readiness with learning visibility, ensuring every architectural decision can be clearly explained and justified in professional contexts. Reproducibility is enforced through Docker-first development and pinned dependencies. Observability comes from integrated experiment tracking and pipeline monitoring. Privacy is embedded through differential privacy libraries and pseudonymization techniques applied by default. Each tool serves a specific business goal while demonstrating modern ML engineering practices.

## 2. Tool Table

### Data & ML
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| pandas | Data Processing | 2.2.3 | Industry standard for data manipulation with excellent ecosystem integration | Polars, DuckDB (considered for performance but pandas offers better portfolio demonstration) |
| numpy | Numerical Computing | 2.1.3 | Fundamental package for scientific computing and array operations | Built-in Python lists (insufficient for numerical operations) |
| scikit-learn | ML Framework | 1.5.2 | Comprehensive, well-documented library with consistent APIs for traditional ML | XGBoost standalone (integrated via sklearn interface) |
| lightgbm | Gradient Boosting | 4.5.0 | High-performance gradient boosting with excellent categorical handling and speed | XGBoost (LightGBM faster for our dataset size) |
| shap | Model Interpretability | 0.46.0 | Model-agnostic explanations with strong theoretical foundations | LIME (SHAP provides better consistency) |
| diffprivlib | Differential Privacy | 0.7.1 | IBM's library offering rigorous differential privacy guarantees | TensorFlow Privacy (focused on deep learning) |
| rapidfuzz | Fuzzy Matching | 3.10.0 | Fast fuzzy string matching for entity resolution tasks | difflib (slower), fuzzywuzzy (less maintained) |

### Pipeline & Orchestration
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| Kedro | Pipeline Framework | 0.19.9 | Declarative pipelines with clear separation of data, processing, and configuration | Airflow (more complex for portfolio scope) |
| Prefect | Orchestration | 3.1.6 | Modern orchestration with intuitive API and excellent observability | Airflow (Prefect easier to demonstrate) |
| MLflow | Experiment Tracking | 2.17.2 | Standard experiment tracking with model registry capabilities | Weights & Biases (MLflow more open-source friendly) |

### Storage
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| SQLite | Development Database | 3.42.0 | Lightweight file-based database ideal for development | PostgreSQL (used in production, not MVP) |
| PostgreSQL | Production Database | 16.4 | Robust relational database with strong privacy controls | MySQL (PostgreSQL better for analytics workloads) |

### Serving
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| FastAPI | REST API | 0.115.4 | Modern, fast web framework with automatic documentation | Flask (FastAPI provides better async support) |
| Dash | Interactive Dashboard | 2.18.1 | Python-first interactive visualization framework | Streamlit (Dash better for analytical dashboards) |
| FastMCP | MCP Server | 0.1.0 | Enables LLM agent integration with ML pipeline outputs | Custom API (FastMCP provides standardized agent interface) |
| Ollama | Local LLM | 0.3.14 | Local LLM inference without external dependencies | OpenAI API (Ollama ensures privacy-first approach) |

### Privacy & Security
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| diffprivlib | Differential Privacy | 0.7.1 | Rigorous differential privacy implementation | PySyft (diffprivlib simpler for portfolio demonstration) |
| python-jose | JWT Authentication | 3.3.0 | Standard JWT implementation for API security | Custom token systems (insecure alternatives) |
| hashlib/hmac | Pseudonymization | Built-in | Cryptographically secure hashing for data anonymization | Custom hashing (insecure) |

### Infrastructure
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| Docker + Docker Compose | Containerization | 27.3.1 + 2.29.7 | Reproducible environments with clear deployment paths | Virtual environments (insufficient for ops demonstration) |
| GitHub Actions | CI/CD | N/A | Integrated CI/CD with GitHub repositories | Jenkins (GitHub Actions simpler for portfolio project) |
| DigitalOcean | Cloud Provider | N/A | Cost-effective cloud platform with good documentation | AWS (DigitalOcean simpler setup) |

### Dev Tooling
| Tool | Category | Version | Why Chosen | What It Replaces/Alternative Considered |
|------|----------|---------|------------|----------------------------------------|
| uv | Package Manager | 0.5.10 | Ultra-fast Python package installer and resolver | pip + conda (uv provides faster installation) |
| Ruff | Linting/Formatting | 0.7.2 | Extremely fast Python linter with broad rule coverage | flake8 + black (separate tools, Ruff combines both) |
| Pyright | Type Checking | 1.1.391 | Microsoft's static type checker with excellent Python support | mypy (Pyright faster and more accurate) |
| pytest + pytest-cov | Testing | 8.3.3 + 6.0.1 | Standard testing framework with coverage analysis | unittest (pytest more pythonic) |

## 3. Architecture Constraints

1. **Python Version**: All components require Python >=3.11 to leverage modern language features and maintain compatibility with latest packages.
2. **Type Annotation**: All ML code must be fully type-annotated and pass Pyright basic mode checks to ensure code quality and maintainability.
3. **Production Pipeline**: No Jupyter notebooks allowed in production pipeline components (kedro pipelines); notebooks only permitted in `/notebooks/` directory for exploratory analysis.
4. **Docker-first Development**: Every service must be containerized and runnable through docker-compose for local development and deployment consistency.

## 4. Version Pinning Strategy

All Python dependencies are pinned to exact versions in pyproject.toml to ensure reproducible builds and prevent unexpected behavior due to upstream package changes. Version updates will occur through systematic review cycles:
1. Periodic assessment of security vulnerabilities using `pip-audit`
2. Quarterly review of major package updates for feature enhancements
3. Integration testing before applying any version upgrades to validate compatibility across the entire stack

This strategy balances stability with maintainability while ensuring security updates can be applied promptly.