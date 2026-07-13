"""Pydantic enums and models for BizSentinel pipeline contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AnomalyFlag(StrEnum):
    """Enumeration of anomaly classification flags."""

    normal = "normal"
    suspicious = "suspicious"
    anomalous = "anomalous"


class SegmentLabel(StrEnum):
    """Enumeration of customer segment labels."""

    champions = "champions"
    loyal = "loyal"
    at_risk = "at_risk"
    hibernating = "hibernating"
    lost = "lost"
    new_customers = "new_customers"


class CustomerFeatures(BaseModel):
    """Feature vector for ML algorithms - output of feature engineering pipeline.

    This model represents the analytical layer with engineered features ready
    for machine learning algorithms.
    """

    customer_hash: str
    snapshot_date: datetime

    # RFM features
    recency_days: float = Field(..., description="Days since last purchase")
    frequency: int = Field(..., description="Number of orders")
    monetary_total: float = Field(..., description="Total spend in BRL")
    monetary_avg: float = Field(..., description="Average order value")

    # Behavioral features
    avg_review_score: float
    review_count: int
    payment_installments_avg: float
    unique_product_categories: int
    avg_delivery_days: float
    late_delivery_rate: float = Field(..., description="Fraction of orders delivered late")

    # Derived from other modules (nullable - filled progressively)
    anomaly_score: float | None = None
    segment_label: SegmentLabel | None = None

    model_config = ConfigDict(from_attributes=True)
