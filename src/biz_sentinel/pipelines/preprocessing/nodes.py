"""Node functions for the preprocessing pipeline.

This module contains pure functions that transform raw data into cleaned,
normalized formats ready for feature engineering. These functions must be
stateless with clear input/output contracts.
"""

import os

import pandas as pd

from biz_sentinel.privacy.pseudonymizer import pseudonymize_dataframe


def clean_orders(olist_orders_raw: pd.DataFrame, valid_statuses: list[str]) -> pd.DataFrame:
    # Filter to valid order statuses
    cleaned = olist_orders_raw[olist_orders_raw["order_status"].isin(valid_statuses)].copy()

    # Drop rows where order_purchase_timestamp is null
    cleaned = cleaned.dropna(subset=["order_purchase_timestamp"])  # type: ignore[call-overload]

    # Parse datetime columns
    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in datetime_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")

    # Drop duplicate order_id rows (keep first)
    cleaned = cleaned.drop_duplicates(subset=["order_id"], keep="first")

    return cleaned


def clean_customers(olist_customers_raw: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with null customer_id or customer_unique_id
    cleaned = olist_customers_raw.dropna(subset=["customer_id", "customer_unique_id"]).copy()

    # Strip whitespace from city and state
    cleaned["customer_city"] = cleaned["customer_city"].str.strip()
    cleaned["customer_state"] = cleaned["customer_state"].str.strip()

    # Uppercase state (Brazilian state codes are 2-letter uppercase)
    cleaned["customer_state"] = cleaned["customer_state"].str.upper()

    # Drop duplicates on customer_unique_id (keep last - most recent record)
    cleaned = cleaned.drop_duplicates(subset=["customer_unique_id"], keep="last")

    return cleaned


def clean_reviews(olist_order_reviews_raw: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with null review_score
    cleaned = olist_order_reviews_raw.dropna(subset=["review_score"]).copy()

    # Filter: review_score between 1 and 5
    cleaned = cleaned[(cleaned["review_score"] >= 1) & (cleaned["review_score"] <= 5)]

    # Parse datetime columns
    datetime_columns = ["review_creation_date", "review_answer_timestamp"]

    for col in datetime_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")

    # Drop duplicates on review_id
    cleaned = cleaned.drop_duplicates(subset=["review_id"])  # type: ignore[call-overload]

    return cleaned


def clean_payments(olist_order_payments_raw: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with payment_value <= 0
    cleaned = olist_order_payments_raw[olist_order_payments_raw["payment_value"] > 0].copy()

    # Drop rows with null payment_type
    cleaned = cleaned.dropna(subset=["payment_type"])  # type: ignore[call-overload]

    return cleaned


def pseudonymize_customers(customers_clean: pd.DataFrame, parameters: dict | None = None) -> pd.DataFrame:
    hmac_salt = os.getenv("HMAC_SALT")
    if not hmac_salt:
        raise ValueError("HMAC_SALT environment variable must be set for pseudonymization")

    result_df = pseudonymize_dataframe(
        customers_clean, id_column="customer_id", salt=hmac_salt, drop_original=True
    )

    if "customer_unique_id" in result_df.columns:
        result_df = pseudonymize_dataframe(
            result_df, id_column="customer_unique_id", salt=hmac_salt, drop_original=True
        )

    return result_df


def build_transactions(
    orders_clean: pd.DataFrame, olist_order_items_raw: pd.DataFrame, payments_clean: pd.DataFrame
) -> pd.DataFrame:
    transactions = pd.merge(olist_order_items_raw, orders_clean, on="order_id", how="left")

    payment_agg = (
        payments_clean.groupby("order_id")
        .agg({"payment_value": "sum", "payment_installments": "max"})
        .reset_index()
    )

    transactions = pd.merge(transactions, payment_agg, on="order_id", how="left")

    # Select and reorder relevant columns
    column_order = [
        "order_id",
        "customer_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
        "payment_value",
        "payment_installments",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    # Only select columns that exist in the DataFrame
    existing_columns = [col for col in column_order if col in transactions.columns]
    transactions = transactions[existing_columns]

    return transactions  # type: ignore[return-value]
