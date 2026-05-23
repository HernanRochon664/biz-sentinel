"""Kedro pipeline for preprocessing raw data.

This pipeline orchestrates the cleaning and normalization of raw data sources:
- Orders dataset filtering and timestamp parsing
- Customers data deduplication and standardization
- Reviews validation and cleaning
- Payments data sanitization
- Customer pseudonymization for privacy protection
- Transaction dataset construction via dataset joins

All processing follows BizSentinel's privacy-first principles with deterministic
pseudonymization of customer identifiers.
"""

from kedro.pipeline import Pipeline, node, pipeline  # type: ignore[import-untyped]

from biz_sentinel.pipelines.preprocessing.nodes import (
    build_transactions,
    clean_customers,
    clean_orders,
    clean_payments,
    clean_reviews,
    pseudonymize_customers,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_orders,
                inputs=["olist_orders_raw", "params:preprocessing.valid_order_statuses"],
                outputs="orders_clean",
                name="clean_orders_node",
                tags=["preprocessing"],
            ),
            node(
                func=clean_customers,
                inputs="olist_customers_raw",
                outputs="customers_clean",
                name="clean_customers_node",
                tags=["preprocessing"],
            ),
            node(
                func=clean_reviews,
                inputs="olist_order_reviews_raw",
                outputs="reviews_clean",
                name="clean_reviews_node",
                tags=["preprocessing"],
            ),
            node(
                func=clean_payments,
                inputs="olist_order_payments_raw",
                outputs="payments_clean",
                name="clean_payments_node",
                tags=["preprocessing"],
            ),
            node(
                func=pseudonymize_customers,
                inputs=["customers_clean", "parameters"],
                outputs="customers_pseudonymized",
                name="pseudonymize_customers_node",
                tags=["preprocessing", "privacy"],
            ),
            node(
                func=build_transactions,
                inputs=["orders_clean", "olist_order_items_raw", "payments_clean"],
                outputs="transactions_clean",
                name="build_transactions_node",
                tags=["preprocessing"],
            ),
        ]
    )
