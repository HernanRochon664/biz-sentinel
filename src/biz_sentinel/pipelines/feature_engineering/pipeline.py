"""Kedro pipeline for feature engineering.

This pipeline computes customer-level features from preprocessed transaction data:
- RFM (Recency, Frequency, Monetary) features
- Review-related features (average score, review count)
- Delivery performance features (avg delivery days, late delivery rate)
- Payment features (installment patterns)
- Final feature matrix assembly

The feature_engineering pipeline depends on outputs from the preprocessing pipeline.
It must be run after preprocessing. The pipeline_registry.py __default__ pipeline
handles this ordering.

"""

from kedro.pipeline import Pipeline, node, pipeline  # type: ignore[import-untyped]


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline.

    This pipeline transforms preprocessed transaction data into customer-level
    feature vectors suitable for ML models. Each node computes a specific
    feature category, and the final node assembles them into a unified matrix.

    Args:
        kwargs: Optional keyword arguments for the pipeline.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.feature_engineering.nodes.compute_rfm",  # type: ignore[reportArgumentType]
                inputs=["transactions_clean", "params:feature_engineering.rfm_snapshot_date"],
                outputs="rfm_features",
                name="compute_rfm_node",
                tags=["feature_engineering", "rfm"],
            ),
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.feature_engineering.nodes.compute_review_features",  # type: ignore[reportArgumentType]
                inputs=["reviews_clean", "orders_clean"],
                outputs="review_features",
                name="compute_review_features_node",
                tags=["feature_engineering", "reviews"],
            ),
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.feature_engineering.nodes.compute_delivery_features",  # type: ignore[reportArgumentType]
                inputs="transactions_clean",
                outputs="delivery_features",
                name="compute_delivery_features_node",
                tags=["feature_engineering", "delivery"],
            ),
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.feature_engineering.nodes.compute_payment_features",  # type: ignore[reportArgumentType]
                inputs="transactions_clean",
                outputs="payment_features",
                name="compute_payment_features_node",
                tags=["feature_engineering", "payments"],
            ),
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.feature_engineering.nodes.assemble_feature_matrix",  # type: ignore[reportArgumentType]
                inputs=[
                    "rfm_features",
                    "review_features",
                    "delivery_features",
                    "payment_features",
                    "params:feature_engineering.rfm_snapshot_date",
                ],
                outputs="feature_matrix",
                name="assemble_feature_matrix_node",
                tags=["feature_engineering", "assembly"],
            ),
        ]
    )
