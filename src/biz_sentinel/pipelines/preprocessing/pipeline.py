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

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create the preprocessing pipeline.
    
    This pipeline transforms raw datasets into clean, normalized intermediate
    representations. The pipeline handles data quality issues like duplicates,
    invalid values, and type mismatches while preserving referential integrity.
    
    The pipeline applies privacy-preserving transformations by pseudonymizing
    customer identifiers using HMAC-SHA256. The cryptographic salt is loaded
    from environment variables rather than configuration files to maintain security.
    
    Args:
        kwargs: Optional keyword arguments for the pipeline.
        
    Returns:
        A Kedro Pipeline object.
    """
    return pipeline([
        # Clean raw orders data
        node(  # type: ignore[arg-type]
            func="biz_sentinel.pipelines.preprocessing.nodes.clean_orders",
            inputs=["olist_orders_raw", "params:preprocessing.valid_order_statuses"],
            outputs="orders_clean",
            name="clean_orders_node",
            tags=["preprocessing"]
        ),

        # Clean raw customers data
        node(  # type: ignore[arg-type]
            func="biz_sentinel.pipelines.preprocessing.nodes.clean_customers",
            inputs="olist_customers_raw",
            outputs="customers_clean",
            name="clean_customers_node",
            tags=["preprocessing"]
        ),

        # Clean raw reviews data
        node(  # type: ignore[arg-type]
            func="biz_sentinel.pipelines.preprocessing.nodes.clean_reviews",
            inputs="olist_order_reviews_raw",
            outputs="reviews_clean",
            name="clean_reviews_node",
            tags=["preprocessing"]
        ),

        # Clean raw payments data
        node(  # type: ignore[arg-type]
            func="biz_sentinel.pipelines.preprocessing.nodes.clean_payments",
            inputs="olist_order_payments_raw",
            outputs="payments_clean",
            name="clean_payments_node",
            tags=["preprocessing"]
        ),

        # Pseudonymize customer identifiers
        # Note: hmac_salt is NOT a catalog input - it will be injected via env var in the runner
        node(  # type: ignore[arg-type]
            func="biz_sentinel.pipelines.preprocessing.nodes.pseudonymize_customers",
            inputs="customers_clean",
            outputs="customers_pseudonymized",
            name="pseudonymize_customers_node",
            tags=["preprocessing", "privacy"]
        ),

        # Build transaction dataset
        node(  # type: ignore[arg-type]
            func="biz_sentinel.pipelines.preprocessing.nodes.build_transactions",
            inputs=["orders_clean", "olist_order_items_raw", "payments_clean"],
            outputs="transactions_clean",
            name="build_transactions_node",
            tags=["preprocessing"]
        )
    ])