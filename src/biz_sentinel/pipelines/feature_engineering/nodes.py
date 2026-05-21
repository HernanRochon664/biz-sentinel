"""Node functions for the feature engineering pipeline.

This module contains pure functions that compute RFM, review, delivery,
and payment features from transaction data. These functions must be stateless
with clear input/output contracts.
"""

import pandas as pd


def compute_rfm(transactions: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary metrics per customer.

    Aggregates transaction data to create RFM features for each customer.

    Args:
        transactions: Transactions DataFrame with columns:
            customer_id, order_id, order_purchase_timestamp, payment_value
        snapshot_date: ISO format date string (e.g., "2018-10-01") used as
            the reference point for computing recency

    Returns:
        DataFrame with columns:
            [customer_id, recency_days, frequency, monetary_total, monetary_avg]

    Raises:
        ValueError: If snapshot_date is null or empty

    Notes:
        - Recency is calculated as (snapshot_date - most_recent_order).days
        - Frequency counts distinct order_ids per customer
        - Monetary_total sums all payment values per customer
        - The calling pipeline must ensure the correct join key is used;
          pseudonymization happens in preprocessing but the join key here
          comes from transactions which may still use customer_id
    """
    if not snapshot_date or pd.isna(snapshot_date):
        raise ValueError("snapshot_date must be a non-empty date string")

    snapshot_dt = pd.to_datetime(snapshot_date)

    grouped = (
        transactions.groupby("customer_id")
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


def compute_review_features(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Compute review-related features per customer.

    Aggregates review scores and counts by customer_id.

    Args:
        reviews: Reviews DataFrame with columns:
            order_id, review_score
        orders: Orders DataFrame with columns:
            order_id, customer_id

    Returns:
        DataFrame with columns:
            [customer_id, avg_review_score, review_count]

    Notes:
        - Only customers with reviews are included in the output
        - The left join from all customers (for customers with no reviews)
          is handled in assemble_feature_matrix, not here
    """
    reviews_with_customers = pd.merge(
        reviews, orders[["order_id", "customer_id"]], on="order_id", how="inner"
    )

    grouped = (
        reviews_with_customers.groupby("customer_id")
        .agg(avg_review_score=("review_score", "mean"), review_count=("review_score", "count"))
        .reset_index()
    )

    return grouped


def compute_delivery_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute delivery performance features per customer.

    Calculates average delivery time and late delivery rate.

    Args:
        transactions: Transactions DataFrame with columns:
            customer_id, order_purchase_timestamp, order_delivered_customer_date,
            order_estimated_delivery_date

    Returns:
        DataFrame with columns:
            [customer_id, avg_delivery_days, late_delivery_rate]

    Notes:
        - Only rows with non-null order_delivered_customer_date are considered
        - Late delivery is when actual delivery exceeds estimated delivery
          (only calculated where order_estimated_delivery_date is not null)
        - Customers with no delivered orders receive 0.0 for both metrics
    """
    delivered = transactions.dropna(subset=["order_delivered_customer_date"]).copy()

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


def compute_payment_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute payment-related features per customer.

    Aggregates payment installment patterns by customer.

    Args:
        transactions: Transactions DataFrame with columns:
            customer_id, payment_installments

    Returns:
        DataFrame with columns:
            [customer_id, payment_installments_avg, unique_product_categories]

    Notes:
        - unique_product_categories is a placeholder (set to 0) because
          product category data comes from a separate table not joined here.
          TODO: Populate when product data is joined in a later pipeline stage.
    """
    grouped = (
        transactions.groupby("customer_id")
        .agg(payment_installments_avg=("payment_installments", "mean"))
        .reset_index()
    )

    grouped["unique_product_categories"] = 0

    cols = ["customer_id", "payment_installments_avg", "unique_product_categories"]
    return pd.DataFrame(grouped[cols])


def assemble_feature_matrix(
    rfm: pd.DataFrame,
    review_features: pd.DataFrame,
    delivery_features: pd.DataFrame,
    payment_features: pd.DataFrame,
    snapshot_date: str,
) -> pd.DataFrame:
    """Assemble final feature matrix by joining all feature DataFrames.

    Combines RFM, review, delivery, and payment features into a single
    feature matrix matching the CustomerFeatures schema.

    Args:
        rfm: RFM features DataFrame from compute_rfm
        review_features: Review features DataFrame from compute_review_features
        delivery_features: Delivery features DataFrame from compute_delivery_features
        payment_features: Payment features DataFrame from compute_payment_features
        snapshot_date: ISO format date string for the feature snapshot

    Returns:
        DataFrame with columns matching CustomerFeatures schema:
        [customer_hash, snapshot_date, recency_days, frequency, monetary_total,
         monetary_avg, avg_review_score, review_count, payment_installments_avg,
         unique_product_categories, avg_delivery_days, late_delivery_rate,
         anomaly_score, segment_label]

    Raises:
        ValueError: If any required columns are missing after assembly

    Notes:
        - Starts from rfm as base (all customers with purchases)
        - Left joins other features to preserve all customers
        - Nulls are filled with sensible defaults:
          avg_review_score -> 3.0, review_count -> 0,
          avg_delivery_days -> 0.0, late_delivery_rate -> 0.0,
          payment_installments_avg -> 1.0, unique_product_categories -> 0
        - customer_id is renamed to customer_hash at this stage, assuming
          pseudonymization has already occurred upstream. The column is
          renamed to match the CustomerFeatures contract.
    """
    if not snapshot_date or pd.isna(snapshot_date):
        raise ValueError("snapshot_date must be a non-empty date string")

    result = rfm.copy()

    result = pd.merge(result, review_features, on="customer_id", how="left")
    result = pd.merge(result, delivery_features, on="customer_id", how="left")
    result = pd.merge(result, payment_features, on="customer_id", how="left")

    result["avg_review_score"] = result["avg_review_score"].fillna(3.0)
    result["review_count"] = result["review_count"].fillna(0)
    result["avg_delivery_days"] = result["avg_delivery_days"].fillna(0.0)
    result["late_delivery_rate"] = result["late_delivery_rate"].fillna(0.0)
    result["payment_installments_avg"] = result["payment_installments_avg"].fillna(1.0)
    result["unique_product_categories"] = result["unique_product_categories"].fillna(0)

    result["snapshot_date"] = pd.to_datetime(snapshot_date)

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
