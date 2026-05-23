"""Node functions for the feature engineering pipeline.

This module contains pure functions that compute RFM, review, delivery,
and payment features from transaction data. These functions must be stateless
with clear input/output contracts.
"""

import pandas as pd


def compute_rfm(transactions_clean: pd.DataFrame, rfm_snapshot_date: str) -> pd.DataFrame:
    if not rfm_snapshot_date or pd.isna(rfm_snapshot_date):
        raise ValueError("rfm_snapshot_date must be a non-empty date string")

    snapshot_dt = pd.to_datetime(rfm_snapshot_date)

    grouped = (
        transactions_clean.groupby("customer_id")
        .agg(
            last_order=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary_total=("payment_value", "sum"),
        )
        .reset_index()
    )

    grouped["recency_days"] = (snapshot_dt - grouped["last_order"]).dt.days
    grouped["monetary_avg"] = grouped["monetary_total"] / grouped["frequency"]
    result = grouped[["customer_id", "recency_days", "frequency", "monetary_total", "monetary_avg"]]

    return pd.DataFrame(result)


def compute_review_features(reviews_clean: pd.DataFrame,
                            orders_clean: pd.DataFrame) -> pd.DataFrame:
    reviews_with_customers = pd.merge(
        reviews_clean, orders_clean[["order_id", "customer_id"]], on="order_id", how="inner"
    )

    grouped = (
        reviews_with_customers.groupby("customer_id")
        .agg(avg_review_score=("review_score", "mean"), review_count=("review_score", "count"))
        .reset_index()
    )

    return grouped


def compute_delivery_features(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    delivered = transactions_clean.dropna(subset=["order_delivered_customer_date"]).copy()

    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.days

    mask_has_estimate = delivered["order_estimated_delivery_date"].notna()
    delivered["estimated_days"] = float("nan")
    delivered.loc[mask_has_estimate, "estimated_days"] = (
        delivered.loc[mask_has_estimate, "order_estimated_delivery_date"]
        - delivered.loc[mask_has_estimate, "order_purchase_timestamp"]
    ).dt.days

    delivered["is_late"] = pd.NA
    delivered.loc[mask_has_estimate, "is_late"] = (
        delivered.loc[mask_has_estimate, "delivery_days"]
        > delivered.loc[mask_has_estimate, "estimated_days"]
    )

    grouped = (
        delivered.groupby("customer_id")
        .agg(avg_delivery_days=("delivery_days", "mean"), late_delivery_rate=("is_late", "mean"))
        .reset_index()
    )

    late_rate = grouped["late_delivery_rate"]
    grouped["late_delivery_rate"] = late_rate.infer_objects(copy=False).fillna(0.0).astype(float)

    return grouped


def compute_payment_features(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        transactions_clean.groupby("customer_id")
        .agg(payment_installments_avg=("payment_installments", "mean"))
        .reset_index()
    )

    grouped["unique_product_categories"] = 0

    cols = ["customer_id", "payment_installments_avg", "unique_product_categories"]
    return pd.DataFrame(grouped[cols])


def assemble_feature_matrix(
    rfm_features: pd.DataFrame,
    review_features: pd.DataFrame,
    delivery_features: pd.DataFrame,
    payment_features: pd.DataFrame,
    rfm_snapshot_date: str,
) -> pd.DataFrame:
    if not rfm_snapshot_date or pd.isna(rfm_snapshot_date):
        raise ValueError("rfm_snapshot_date must be a non-empty date string")

    result = rfm_features.copy()

    result = pd.merge(result, review_features, on="customer_id", how="left")
    result = pd.merge(result, delivery_features, on="customer_id", how="left")
    result = pd.merge(result, payment_features, on="customer_id", how="left")

    result["avg_review_score"] = result["avg_review_score"].fillna(3.0)
    result["review_count"] = result["review_count"].fillna(0)
    result["avg_delivery_days"] = result["avg_delivery_days"].fillna(0.0)
    result["late_delivery_rate"] = result["late_delivery_rate"].fillna(0.0)
    result["payment_installments_avg"] = result["payment_installments_avg"].fillna(1.0)
    result["unique_product_categories"] = result["unique_product_categories"].fillna(0)

    result["snapshot_date"] = pd.to_datetime(rfm_snapshot_date)

    result["anomaly_score"] = None
    result["segment_label"] = None

    result = result.rename(columns={"customer_id": "customer_hash"})

    required_columns = [
        "customer_hash",
        "snapshot_date",
        "recency_days",
        "frequency",
        "monetary_total",
        "monetary_avg",
        "avg_review_score",
        "review_count",
        "payment_installments_avg",
        "unique_product_categories",
        "avg_delivery_days",
        "late_delivery_rate",
        "anomaly_score",
        "segment_label",
    ]

    missing_columns = set(required_columns) - set(result.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in feature matrix: {missing_columns}")

    return pd.DataFrame(result[required_columns])
