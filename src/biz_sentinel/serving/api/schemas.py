"""Pydantic response schemas for the BizSentinel API.

Separate from domain/models.py — these are API-layer contracts,
optimized for JSON serialization, not ML pipeline contracts.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class AnomalyScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_hash: str
    anomaly_score: float
    is_anomaly: bool
    anomaly_flag: str
    scored_at: datetime | None = None


class ChurnScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_hash: str
    churn_probability: float
    predicted_churn: bool
    scored_at: datetime | None = None


class CustomerRiskSummary(BaseModel):
    """Combined risk view — anomaly + churn in one response."""

    customer_hash: str
    anomaly_score: float | None = None
    anomaly_flag: str | None = None
    churn_probability: float | None = None
    predicted_churn: bool | None = None
    segment_label: str | None = None


class AnomalySummary(BaseModel):
    """Summary stats for recent anomalies."""

    total_customers_scored: int
    anomaly_count: int
    suspicious_count: int
    normal_count: int
    anomaly_rate: float
    period_days: int


class SegmentProfileResponse(BaseModel):
    segment_label: str
    customer_count: int
    avg_recency_days: float
    avg_frequency: float
    avg_monetary_total: float
    avg_review_score: float


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_hash: str
    alert_type: str
    score: float
    threshold: float
    created_at: datetime
    is_resolved: bool
