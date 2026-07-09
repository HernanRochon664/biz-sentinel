"""BizSentinel FastAPI application.

Exposes ML model outputs (anomaly scores, churn probabilities,
customer segments) via a REST API with JWT authentication.
"""

import os
import warnings
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt  # type: ignore[import-untyped]
from sqlalchemy.orm import Session

from biz_sentinel.serving.api.database import (
    AlertRecord,
    AnomalyScoreRecord,
    ChurnScoreRecord,
    SegmentRecord,
    get_db,
    get_engine,
    init_db,
)
from biz_sentinel.serving.api.schemas import (
    AlertResponse,
    AnomalySummary,
    CustomerRiskSummary,
    HealthResponse,
)

ALGORITHM = "HS256"
VERSION = "0.1.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

SECRET_KEY = os.getenv("SECRET_KEY")
if ENVIRONMENT == "production" and not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable must be set in production. "
        "Export SECRET_KEY=<your-secret> before starting the server."
    )
if not SECRET_KEY:
    SECRET_KEY = "dev_secret_change_in_production"
    warnings.warn(
        "SECRET_KEY not set. Using insecure default for development. Set SECRET_KEY in production.",
        stacklevel=2,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(get_engine())
    yield


app = FastAPI(
    title="BizSentinel API",
    description="Business anomaly detection and customer intelligence platform",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> str:
    """Verify JWT bearer token. Returns subject claim."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        subject: str = payload.get("sub", "")
        if not subject:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return subject
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from exc


# --- Routes ---


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Health check — no auth required."""
    return HealthResponse(status="ok", version=VERSION, environment=ENVIRONMENT)


@app.get("/anomalies/recent", response_model=AnomalySummary, tags=["anomalies"])
def get_anomaly_summary(
    days: int = 7,
    db: Session = Depends(get_db),
    _: str = Depends(verify_token),
) -> AnomalySummary:
    """Get summary statistics of recent anomalies."""
    records = db.query(AnomalyScoreRecord).all()
    total = len(records)
    anomaly_count = sum(1 for r in records if r.anomaly_flag == "anomalous")  # type: ignore[misc]
    suspicious_count = sum(1 for r in records if r.anomaly_flag == "suspicious")  # type: ignore[misc]
    return AnomalySummary(
        total_customers_scored=total,
        anomaly_count=anomaly_count,
        suspicious_count=suspicious_count,
        normal_count=total - anomaly_count - suspicious_count,
        anomaly_rate=anomaly_count / total if total > 0 else 0.0,
        period_days=days,
    )


@app.get("/customers/{customer_hash}/risk", response_model=CustomerRiskSummary, tags=["customers"])
def get_customer_risk(
    customer_hash: str,
    db: Session = Depends(get_db),
    _: str = Depends(verify_token),
) -> CustomerRiskSummary:
    """Get combined risk profile for a specific customer."""
    anomaly = (
        db.query(AnomalyScoreRecord)
        .filter(AnomalyScoreRecord.customer_hash == customer_hash)
        .first()
    )
    churn = (
        db.query(ChurnScoreRecord).filter(ChurnScoreRecord.customer_hash == customer_hash).first()
    )
    segment = db.query(SegmentRecord).filter(SegmentRecord.customer_hash == customer_hash).first()

    if not anomaly and not churn:
        raise HTTPException(status_code=404, detail=f"No scores found for customer {customer_hash}")

    return CustomerRiskSummary(
        customer_hash=customer_hash,
        anomaly_score=anomaly.anomaly_score if anomaly else None,
        anomaly_flag=anomaly.anomaly_flag if anomaly else None,
        churn_probability=churn.churn_probability if churn else None,
        predicted_churn=churn.predicted_churn if churn else None,
        segment_label=segment.segment_label if segment else None,
    )


@app.get("/alerts", response_model=list[AlertResponse], tags=["alerts"])
def get_alerts(
    resolved: bool = False,
    limit: int = 50,
    db: Session = Depends(get_db),
    _: str = Depends(verify_token),
) -> list[AlertResponse]:
    """Get recent alerts, optionally filtered by resolution status."""
    query = db.query(AlertRecord).filter(AlertRecord.is_resolved == resolved)
    records = query.order_by(AlertRecord.created_at.desc()).limit(limit).all()
    return [AlertResponse.model_validate(r) for r in records]


@app.post("/score/transaction", tags=["scoring"])
def score_transaction(
    _: str = Depends(verify_token),
) -> dict:
    """Placeholder for real-time transaction scoring.
    Full implementation in Phase 8 (inference pipeline integration).
    """
    return {"detail": "Real-time scoring not yet implemented. Use batch inference."}
