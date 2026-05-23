from kedro.pipeline import Pipeline, node, pipeline  # type: ignore[import-untyped]

from biz_sentinel.pipelines.feature_engineering.nodes import (
    assemble_feature_matrix,
    compute_delivery_features,
    compute_payment_features,
    compute_review_features,
    compute_rfm,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compute_rfm,
                inputs=["transactions_clean", "params:feature_engineering.rfm_snapshot_date"],
                outputs="rfm_features",
                name="compute_rfm_node",
                tags=["feature_engineering", "rfm"],
            ),
            node(
                func=compute_review_features,
                inputs=["reviews_clean", "orders_clean"],
                outputs="review_features",
                name="compute_review_features_node",
                tags=["feature_engineering", "reviews"],
            ),
            node(
                func=compute_delivery_features,
                inputs="transactions_clean",
                outputs="delivery_features",
                name="compute_delivery_features_node",
                tags=["feature_engineering", "delivery"],
            ),
            node(
                func=compute_payment_features,
                inputs="transactions_clean",
                outputs="payment_features",
                name="compute_payment_features_node",
                tags=["feature_engineering", "payments"],
            ),
            node(
                func=assemble_feature_matrix,
                inputs=[
                    "rfm_features",
                    "review_features",
                    "delivery_features",
                    "payment_features",
                    "params:rfm_snapshot_date",
                ],
                outputs="feature_matrix",
                name="assemble_feature_matrix_node",
                tags=["feature_engineering", "assembly"],
            ),
        ]
    )
