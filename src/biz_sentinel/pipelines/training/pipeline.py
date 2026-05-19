"""Kedro pipeline for training anomaly detection models (Module A).

This pipeline trains an Isolation Forest model for anomaly detection and produces
scored output with SHAP-based explanations.

Note: anomaly_model and anomaly_training_metrics are in-memory datasets (MemoryDataset).
They are intermediate outputs within the same pipeline run and do not require catalog entries.
The model is passed directly to downstream nodes, and metrics are logged to MLflow.
"""

from kedro.pipeline import Pipeline, node, pipeline  # type: ignore[import-untyped]


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
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.training.nodes.prepare_anomaly_features",
                inputs="feature_matrix",
                outputs=["anomaly_feature_matrix", "customer_hashes"],
                name="prepare_anomaly_features_node",
                tags=["training", "anomaly_detection", "preprocessing"],
            ),
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.training.nodes.train_isolation_forest",
                inputs=[
                    "anomaly_feature_matrix",
                    "params:training.anomaly_contamination",
                    "params:training.random_state",
                ],
                outputs=["anomaly_model", "anomaly_training_metrics"],
                name="train_isolation_forest_node",
                tags=["training", "anomaly_detection", "mlflow"],
            ),
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.training.nodes.score_anomalies",
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
            node(  # type: ignore[arg-type]
                func="biz_sentinel.pipelines.training.nodes.compute_anomaly_shap",
                inputs=["anomaly_model", "anomaly_feature_matrix", "customer_hashes"],
                outputs="anomaly_shap_values",
                name="compute_anomaly_shap_node",
                tags=["training", "anomaly_detection", "explainability"],
            ),
        ]
    )
