"""Pydantic data models for BizSentinel pipeline contracts.

This module defines the two-layer contract pattern:
1. Raw ingestion models: Direct mapping to source data schemas
2. Normalized analytical models: Processed data ready for ML algorithms

All models use Pydantic v2 syntax with complete type annotations.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderStatus(StrEnum):
    """Enumeration of possible order statuses."""
    delivered = "delivered"
    shipped = "shipped"
    invoiced = "invoiced"
    canceled = "canceled"
    unavailable = "unavailable"
    processing = "processing"
    created = "created"
    approved = "approved"


class PaymentType(StrEnum):
    """Enumeration of payment types."""
    credit_card = "credit_card"
    boleto = "boleto"
    voucher = "voucher"
    debit_card = "debit_card"


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


class RawCustomer(BaseModel):
    """Raw customer data model matching Olist olist_customers_dataset.csv schema.
    
    This model represents the initial ingestion layer directly from source data.
    """
    customer_id: str
    customer_unique_id: str
    customer_zip_code_prefix: str
    customer_city: str
    customer_state: str

    model_config = ConfigDict(from_attributes=True)


class RawOrder(BaseModel):
    """Raw order data model matching Olist olist_orders_dataset.csv schema.
    
    This model represents the initial ingestion layer directly from source data.
    """
    order_id: str
    customer_id: str
    order_status: OrderStatus
    order_purchase_timestamp: datetime
    order_approved_at: datetime | None = None
    order_delivered_customer_date: datetime | None = None
    order_estimated_delivery_date: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RawOrderItem(BaseModel):
    """Raw order item data model matching Olist olist_order_items_dataset.csv schema.
    
    This model represents the initial ingestion layer directly from source data.
    """
    order_id: str
    order_item_id: int
    product_id: str
    seller_id: str
    price: float
    freight_value: float

    @field_validator('price', 'freight_value')
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        """Validate that price and freight_value are non-negative."""
        if v < 0:
            raise ValueError(f'{info.field_name} must be >= 0')
        return v

    model_config = ConfigDict(from_attributes=True)


class RawPayment(BaseModel):
    """Raw payment data model matching Olist olist_order_payments_dataset.csv schema.
    
    This model represents the initial ingestion layer directly from source data.
    """
    order_id: str
    payment_sequential: int
    payment_type: PaymentType
    payment_installments: int
    payment_value: float

    @field_validator('payment_value')
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Validate that payment_value is non-negative."""
        if v < 0:
            raise ValueError('payment_value must be >= 0')
        return v

    model_config = ConfigDict(from_attributes=True)


class RawReview(BaseModel):
    """Raw review data model matching Olist olist_order_reviews_dataset.csv schema.
    
    This model represents the initial ingestion layer directly from source data.
    """
    review_id: str
    order_id: str
    review_score: int
    review_creation_date: datetime
    review_answer_timestamp: datetime | None = None

    @field_validator('review_score')
    @classmethod
    def validate_review_score_range(cls, v: int) -> int:
        """Validate that review_score is between 1 and 5 inclusive."""
        if not 1 <= v <= 5:
            raise ValueError('review_score must be between 1 and 5 inclusive')
        return v

    model_config = ConfigDict(from_attributes=True)


class PseudonymizedCustomer(BaseModel):
    """Pseudonymized customer data model with privacy-preserving identifiers.
    
    This model represents the privacy-safe layer where customer_id is replaced
    with an HMAC hash to prevent direct identification while maintaining
    referential integrity.
    """
    customer_hash: str = Field(..., description="HMAC-SHA256 hash of customer_id")
    customer_city: str
    customer_state: str
    # Note: zip code intentionally omitted (too identifying)

    @classmethod
    def from_raw(cls, raw: RawCustomer, salt: str) -> PseudonymizedCustomer:
        """Create a pseudonymized customer from raw data.
        
        Args:
            raw: Raw customer data
            salt: Salt value for HMAC hashing
            
        Returns:
            PseudonymizedCustomer instance
        """
        # Compute HMAC-SHA256 hash of customer_id
        customer_hash = hmac.new(
            salt.encode(),
            raw.customer_id.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return cls(
            customer_hash=customer_hash,
            customer_city=raw.customer_city,
            customer_state=raw.customer_state
        )

    model_config = ConfigDict(from_attributes=True)


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


class ScoredCustomer(BaseModel):
    """Final output model with scores and flags for business decisions.
    
    This model represents the final output layer that gets written to the
    database and served to business users.
    """
    customer_hash: str
    scored_at: datetime
    anomaly_score: float
    anomaly_flag: AnomalyFlag
    segment_label: SegmentLabel
    churn_probability: float
    shap_top_features: dict[str, float] = Field(
            ..., description="Top 5 feature names to SHAP values")
    model_config = ConfigDict(from_attributes=True)
