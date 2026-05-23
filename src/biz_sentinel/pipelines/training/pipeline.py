from kedro.pipeline import Pipeline, node, pipeline  # type: ignore[import-untyped]

from biz_sentinel.pipelines.training.churn_nodes import (
    build_churn_labels,
    compute_churn_shap,
    prepare_churn_features,
    score_churn,
    train_churn_model,
    train_churn_model_with_dp,
)
from biz_sentinel.pipelines.training.nodes import (
    compute_anomaly_shap,
    prepare_anomaly_features,
    score_anomalies,
    train_isolation_forest,
)
from biz_sentinel.pipelines.training.segmentation_nodes import (
    assign_segments,
    compute_segment_profiles,
    find_optimal_k,
    prepare_segmentation_features,
    train_kmeans,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the anomaly detection training pipeline.

    This pipeline:
    1. Extracts anomaly-specific features from the feature matrix
    2. Trains an Isolation Forest model with configurable contamination
    3. Scores all customers for anomaly detection
    4. Computes SHAP values for interpretability

    Args:
        kwargs: Optional keyword arguments for the pipeline.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                func=prepare_anomaly_features,
                inputs="feature_matrix",
                outputs=["anomaly_feature_matrix", "customer_hashes"],
                name="prepare_anomaly_features_node",
                tags=["training", "anomaly_detection", "preprocessing"],
            ),
            node(
                func=train_isolation_forest,
                inputs=[
                    "anomaly_feature_matrix",
                    "params:training.anomaly_contamination",
                    "params:training.random_state",
                ],
                outputs=["anomaly_model", "anomaly_training_metrics"],
                name="train_isolation_forest_node",
                tags=["training", "anomaly_detection", "mlflow"],
            ),
            node(
                func=score_anomalies,
                inputs=[
                    "anomaly_model",
                    "anomaly_feature_matrix",
                    "customer_hashes",
                    "params:inference.anomaly_threshold",
                ],
                outputs="anomaly_scores",
                name="score_anomalies_node",
                tags=["training", "anomaly_detection", "scoring"],
            ),
            node(
                func=compute_anomaly_shap,
                inputs=["anomaly_model", "anomaly_feature_matrix", "customer_hashes"],
                outputs="anomaly_shap_values",
                name="compute_anomaly_shap_node",
                tags=["training", "anomaly_detection", "explainability"],
            ),
            node(  # type: ignore[arg-type]
                func=prepare_segmentation_features,
                inputs="feature_matrix",
                outputs=["segmentation_feature_matrix", "seg_customer_hashes", "feature_scaler"],
                name="prepare_segmentation_features_node",
                tags=["training", "segmentation", "preprocessing"],
            ),
            node(  # type: ignore[arg-type]
                func=find_optimal_k,
                inputs=[
                    "segmentation_feature_matrix",
                    "params:feature_engineering.n_clusters_min",
                    "params:feature_engineering.n_clusters_max",
                    "params:training.random_state",
                ],
                outputs="k_analysis_results",
                name="find_optimal_k_node",
                tags=["training", "segmentation", "analysis"],
            ),
            # NOTE: Replace n_clusters with optimal k from find_optimal_k before production.
            # Currently uses n_clusters_min as reasonable default.
            node(  # type: ignore[arg-type]
                func=train_kmeans,
                inputs=[
                    "segmentation_feature_matrix",
                    "params:feature_engineering.n_clusters_min",
                    "params:training.random_state",
                ],
                outputs=["segmentation_model", "segmentation_metrics"],
                name="train_kmeans_node",
                tags=["training", "segmentation", "mlflow"],
            ),
            node(  # type: ignore[arg-type]
                func=assign_segments,
                inputs=[
                    "segmentation_model",
                    "segmentation_feature_matrix",
                    "seg_customer_hashes",
                    "feature_matrix",
                ],
                outputs="segment_assignments",
                name="assign_segments_node",
                tags=["training", "segmentation", "scoring"],
            ),
            node(  # type: ignore[arg-type]
                func=compute_segment_profiles,
                inputs=["feature_matrix", "segment_assignments"],
                outputs="segment_profiles",
                name="compute_segment_profiles_node",
                tags=["training", "segmentation", "analysis"],
            ),
            node(  # type: ignore[arg-type]
                func=build_churn_labels,
                inputs="feature_matrix",
                outputs="feature_matrix_with_labels",
                name="build_churn_labels_node",
                tags=["training", "churn", "labeling"],
            ),
            node(  # type: ignore[arg-type]
                func=prepare_churn_features,
                inputs=[
                    "feature_matrix_with_labels",
                    "anomaly_scores",
                    "segment_assignments",
                ],
                outputs=["churn_X", "churn_y", "churn_customer_hashes"],
                name="prepare_churn_features_node",
                tags=["training", "churn", "preprocessing"],
            ),
            node(  # type: ignore[arg-type]
                func=train_churn_model,
                inputs=[
                    "churn_X",
                    "churn_y",
                    "params:training.lgbm_n_estimators",
                    "params:training.lgbm_learning_rate",
                    "params:training.random_state",
                ],
                outputs=["churn_model", "churn_metrics"],
                name="train_churn_model_node",
                tags=["training", "churn", "mlflow"],
            ),
            node(  # type: ignore[arg-type]
                func=train_churn_model_with_dp,
                inputs=[
                    "churn_X",
                    "churn_y",
                    "params:training.dp_epsilon",
                    "params:training.dp_delta",
                    "params:training.random_state",
                ],
                outputs=["churn_dp_model", "churn_dp_metrics"],
                name="train_churn_model_with_dp_node",
                tags=["training", "churn", "privacy", "mlflow"],
            ),
            node(  # type: ignore[arg-type]
                func=score_churn,
                inputs=[
                    "churn_model",
                    "churn_X",
                    "churn_customer_hashes",
                    "params:inference.churn_risk_threshold",
                ],
                outputs="churn_scores",
                name="score_churn_node",
                tags=["training", "churn", "scoring"],
            ),
            node(  # type: ignore[arg-type]
                func=compute_churn_shap,
                inputs=["churn_model", "churn_X", "churn_customer_hashes"],
                outputs="churn_shap_values",
                name="compute_churn_shap_node",
                tags=["training", "churn", "explainability"],
            ),
        ]
    )
