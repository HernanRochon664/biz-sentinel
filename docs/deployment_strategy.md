# BizSentinel Deployment Strategy

## 1. Overview

| Mode | Primary Use Case | Implementation Priority | Status |
|------|------------------|-------------------------|--------|
| Batch Inference | Daily automated scoring | Phase 4 | MVP |
| REST API | Real-time integration with external systems | Phase 5 | Post-MVP |
| Dash Dashboard | Interactive exploration and alerts for business users | Phase 5 | Post-MVP |
| MCP Server | LLM agent integration via natural language | Phase 6 | Advanced |

## 2. Mode 1: Batch Inference (Prefect Flow)

- **Triggered**: daily schedule via Prefect
- **Process**: load latest data → run Kedro inference pipeline → write scores to DB → trigger alerts if thresholds exceeded
- **Output**: scored records in anomaly_scores, segment_labels, and risk_scores tables
- **Infrastructure**: runs in Docker container on DigitalOcean droplet

## 3. Mode 2: REST API (FastAPI)

### Endpoints
- `POST /score/transaction` — score a single transaction
- `GET /customers/{id}/risk` — get latest churn score for a customer
- `GET /anomalies/recent` — list recent flagged anomalies
- `GET /health` — health check

### Security
- **Authentication**: JWT bearer token
- **Rate limiting**: basic (no Redis at MVP, in-memory)
- **Docs**: auto-generated via FastAPI /docs

## 4. Mode 3: Dash Dashboard

### Pages
- **Overview**: KPI cards (anomaly rate today, high-risk customers count, top segment)
- **Anomalies**: table of recent anomalies with score and explanation (SHAP top features)
- **Segments**: scatter plot of customer clusters (PCA-reduced), segment profiles
- **Risk**: ranked list of at-risk customers with predicted churn probability

### Implementation
- **Data source**: reads from SQLite/PostgreSQL
- **Refresh**: manual button + automatic every 24h

## 5. Mode 4: MCP Server (FastMCP + Ollama)

### What is MCP?
MCP (Model Control Protocol) is an open protocol for tool-integrated AI agents. It allows language models to interact with software systems through structured tools, enabling natural language queries against business systems.

### Tools Exposed
- `get_anomaly_summary(days: int)` → summary of recent anomalies
- `get_customer_risk(customer_id: str)` → churn score + top SHAP features
- `explain_alert(alert_id: str)` → natural language explanation of a specific alert
- `get_segment_profile(segment_id: int)` → description of a customer segment

### LLM Integration
- **Backend**: Ollama (local)
- **Recommended models**: 
  - Llama 3.1 8B (minimum requirement)
  - Llama 3.3 70B (preferred with sufficient VRAM)

### Hardware Considerations
- **8B models**: Run on laptops with 8GB+ RAM
- **70B models**: Require 12GB+ VRAM GPU

### Example Interaction
> User: "Which customers are at high risk of churning this week?"
> 
> LLM Agent: "I'll check our customer risk scores for this week. [Calling get_customer_risk tool]... Here are the top 5 customers at risk of churning with their risk probabilities: [Lists customers with probabilities]"

### Privacy Note
MCP tools only return scores and explanations, never raw customer data, maintaining compliance with privacy requirements.

## 6. Infrastructure Decision: DigitalOcean

### Why DigitalOcean?
- **Cost-effective** for portfolio projects
- **Docker-native** environment
- **Simple setup** process

### Service Distribution
- **Droplet**: Hosts Docker containers for all services
- **Container 1**: Prefect flow for batch inference
- **Container 2**: FastAPI REST service
- **Container 3**: Dash dashboard application
- **Container 4**: MCP server with Ollama

### Cost Estimate
Rough monthly estimate for minimal DigitalOcean droplet: $10-15/month

## 7. Rollout Order

1. **Phase 4** → Batch Inference only (MVP)
2. **Phase 5** → REST API + Dashboard (simultaneously)
3. **Phase 6** → MCP server integration

**Note**: All four modes share the same underlying Kedro pipeline and MLflow models, ensuring consistency across deployment options.