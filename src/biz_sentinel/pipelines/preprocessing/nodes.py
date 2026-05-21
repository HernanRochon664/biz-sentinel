"""Node functions for the preprocessing pipeline.

This module contains pure functions that transform raw data into cleaned,
normalized formats ready for feature engineering. These functions must be
stateless with clear input/output contracts.
"""

import os

import pandas as pd

from biz_sentinel.privacy.pseudonymizer import pseudonymize_dataframe


def clean_orders(orders: pd.DataFrame, valid_statuses: list[str]) -> pd.DataFrame:
    """Clean raw orders data.

    Filters orders by status, drops invalid timestamps, parses datetime columns,
    and removes duplicates.

    Args:
        orders: Raw orders DataFrame with columns:
            order_id, customer_id, order_status, order_purchase_timestamp,
            order_approved_at, order_delivered_customer_date,
            order_estimated_delivery_date
        valid_statuses: List of acceptable order statuses

    Returns:
        Cleaned orders DataFrame with:
        - Only orders with valid statuses
        - No null purchase timestamps
        - Parsed datetime columns
        - Duplicate order_ids removed (keeping first occurrence)

    Notes:
        - Rows with null order_purchase_timestamp are dropped as they're
          essential for temporal analysis
        - Duplicate order_id rows are removed to maintain primary key integrity
    """
    # Filter to valid order statuses
    cleaned = orders[orders["order_status"].isin(valid_statuses)].copy()

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


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customers data.

    Removes invalid records, cleans geographical data, and removes duplicates.

    Args:
        customers: Raw customers DataFrame with columns:
            customer_id, customer_unique_id, customer_zip_code_prefix,
            customer_city, customer_state

    Returns:
        Cleaned customers DataFrame with:
        - No null customer_id or customer_unique_id
        - Stripped city/state strings
        - Uppercase 2-letter state codes
        - Duplicate customer_unique_id removed (keeping last occurrence)

    Notes:
        - Brazilian state codes are standardized to uppercase 2-letter format
        - Duplicates are resolved by keeping the last record (most recent)
    """
    # Drop rows with null customer_id or customer_unique_id
    cleaned = customers.dropna(subset=["customer_id", "customer_unique_id"]).copy()

    # Strip whitespace from city and state
    cleaned["customer_city"] = cleaned["customer_city"].str.strip()
    cleaned["customer_state"] = cleaned["customer_state"].str.strip()

    # Uppercase state (Brazilian state codes are 2-letter uppercase)
    cleaned["customer_state"] = cleaned["customer_state"].str.upper()

    # Drop duplicates on customer_unique_id (keep last - most recent record)
    cleaned = cleaned.drop_duplicates(subset=["customer_unique_id"], keep="last")

    return cleaned


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean raw reviews data.

    Removes invalid reviews and parses datetime columns.

    Args:
        reviews: Raw reviews DataFrame with columns:
            review_id, order_id, review_score, review_creation_date,
            review_answer_timestamp

    Returns:
        Cleaned reviews DataFrame with:
        - No null review scores
        - Review scores between 1-5
        - Parsed datetime columns
        - Duplicate review_ids removed

    Notes:
        - Review scores are validated to be between 1-5 as per platform policy
        - Duplicates are removed to maintain primary key integrity
    """
    # Drop rows with null review_score
    cleaned = reviews.dropna(subset=["review_score"]).copy()

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


def clean_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Clean raw payments data.

    Removes invalid payments with zero/negative values or missing types.

    Args:
        payments: Raw payments DataFrame with columns:
            order_id, payment_sequential, payment_type, payment_installments,
            payment_value

    Returns:
        Cleaned payments DataFrame with:
        - Payment values > 0
        - Non-null payment types

    Notes:
        - Payments with zero/negative values are removed as they indicate
          data entry errors
        - Payments with null types are removed as they're unprocessable
    """
    # Drop rows with payment_value <= 0
    cleaned = payments[payments["payment_value"] > 0].copy()

    # Drop rows with null payment_type
    cleaned = cleaned.dropna(subset=["payment_type"])  # type: ignore[call-overload]

    return cleaned


def pseudonymize_customers(customers: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Pseudonymize customer identifiers using HMAC-SHA256.

    Replaces identifiable customer fields with cryptographically secure hashes
    to preserve privacy while maintaining referential integrity.

    Args:
        customers: Cleaned customers DataFrame
        parameters: Pipeline parameters dictionary (unused, salt comes from env)

    Returns:
        Pseudonymized customers DataFrame with:
        - customer_id_hash replacing customer_id
        - customer_unique_id_hash replacing customer_unique_id

    Notes:
        - The hmac_salt comes from environment variables rather than
          parameters.yml because it's a security-sensitive secret
        - Both customer_id and customer_unique_id are pseudonymized separately
          to maintain referential integrity in different contexts
    """
    # Get hmac_salt from environment variables (NOT from parameters.yml - it's a secret)
    hmac_salt = os.getenv("HMAC_SALT")
    if not hmac_salt:
        raise ValueError("HMAC_SALT environment variable must be set for pseudonymization")

    # Pseudonymize customer_id column
    result_df = pseudonymize_dataframe(
        customers, id_column="customer_id", salt=hmac_salt, drop_original=True
    )

    # Pseudonymize customer_unique_id column (separately)
    # It might be gone after first pseudonymization
    if "customer_unique_id" in result_df.columns:
        result_df = pseudonymize_dataframe(
            result_df, id_column="customer_unique_id", salt=hmac_salt, drop_original=True
        )

    return result_df


def build_transactions(
    orders: pd.DataFrame, order_items: pd.DataFrame, payments: pd.DataFrame
) -> pd.DataFrame:
    """Build transaction-level dataset by joining orders, items, and payments.

    Creates one row per order_item with complete transaction information
    including pricing, delivery, and payment details.

    Args:
        orders: Cleaned orders DataFrame
        order_items: Raw order items DataFrame with columns:
            order_id, order_item_id, product_id, seller_id, price, freight_value
        payments: Cleaned payments DataFrame

    Returns:
        Transactions DataFrame with one row per order_item and columns:
        - order_id, customer_id, product_id, seller_id, price, freight_value
        - payment_value (summed across payment methods)
        - payment_installments (maximum installment count)
        - order_status, order_purchase_timestamp, order_delivered_customer_date
        - order_estimated_delivery_date

    Notes:
        - Left join from orders to preserve all order_items
        - Payments are aggregated (summed) because one order might have multiple
          payment methods
        - Installment counts are maxed because we're interested in the highest
          installment plan used for the order
    """
    # Join orders + order_items on order_id (left join from orders)
    transactions = pd.merge(order_items, orders, on="order_id", how="left")

    # Aggregate payments: sum payment_value, max installments
    payment_agg = (
        payments.groupby("order_id")
        .agg({"payment_value": "sum", "payment_installments": "max"})
        .reset_index()
    )

    # Join with payments on order_id
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
