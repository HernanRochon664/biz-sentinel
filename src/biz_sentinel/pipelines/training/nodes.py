"""Node functions for the training pipeline.

This module contains functions for training anomaly detection models using
Isolation Forest with MLflow experiment tracking.
"""

import json
import warnings

import pandas as pd
from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

from biz_sentinel.domain.models import AnomalyFlag

ANOMALY_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "avg_review_score",
    "review_count",
    "payment_installments_avg",
    "avg_delivery_days",
    "late_delivery_rate",
]


def prepare_anomaly_features(
    feature_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract and validate anomaly detection features from feature matrix.

    Args:
        feature_matrix: DataFrame with customer features including ANOMALY_FEATURES
            and customer_hash column.

    Returns:
        Tuple of (X, customer_hashes) where X is the feature matrix and
        customer_hashes is a Series of customer identifiers.

    Raises:
        ValueError: If required features are missing or contain null values.
    """
    missing_features = [f for f in ANOMALY_FEATURES if f not in feature_matrix.columns]
    if missing_features:
        raise ValueError(f"Missing required features for anomaly detection: {missing_features}")

    X = feature_matrix[ANOMALY_FEATURES].copy()
    customer_hashes = feature_matrix["customer_hash"]

    null_columns = X.columns[X.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Null values detected in feature columns: {null_columns}")

    return X, customer_hashes  # type: ignore[return-value]


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float,
    random_state: int,
    experiment_name: str = "biz_sentinel/anomaly_detection",
) -> tuple[object, dict[str, float]]:
    """Train Isolation Forest model for anomaly detection.

    MLflow logging is a side effect of this function. The MLflow tracking URI
    must be configured before calling this function (via MLFLOW_TRACKING_URI
    env var or mlflow.set_tracking_uri()).

    Args:
        X: Feature matrix with anomaly detection features.
        contamination: Expected proportion of anomalies in the dataset.
        random_state: Random seed for reproducibility.
        experiment_name: MLflow experiment name for tracking.

    Returns:
        Tuple of (model, metrics) where model is the trained IsolationForest
        and metrics is a dictionary with anomaly detection metrics.
    """
    import mlflow  # type: ignore[import-untyped]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="isolation_forest_training"):
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(ANOMALY_FEATURES))
        mlflow.log_param("features", str(ANOMALY_FEATURES))
        mlflow.log_param("n_samples", len(X))

        model = IsolationForest(
            contamination=contamination,  # type: ignore[arg-type]
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        model.fit(X)

        raw_scores = model.score_samples(X)
        normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        anomaly_scores = 1.0 - normalized_scores

        predictions = model.predict(X)
        anomaly_rate = float((predictions == -1).mean())
        avg_anomaly_score = float(anomaly_scores.mean())

        metrics = {
            "anomaly_rate": anomaly_rate,
            "avg_anomaly_score": avg_anomaly_score,
            "n_anomalies": int((predictions == -1).sum()),
            "contamination_param": contamination,
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(  # type: ignore[attr-defined]
            model,
            artifact_path="isolation_forest",
            registered_model_name="biz_sentinel_anomaly_detector",
        )

    return model, metrics


def score_anomalies(
    model: object,
    X: pd.DataFrame,
    customer_hashes: pd.Series,
    anomaly_threshold: float,
) -> pd.DataFrame:
    """Score customers for anomalies using trained Isolation Forest model.

    Args:
        model: Trained IsolationForest model.
        X: Feature matrix for scoring.
        customer_hashes: Customer identifiers corresponding to X rows.
        anomaly_threshold: Threshold for classifying anomalies (0-1 scale).

    Returns:
        DataFrame with columns: [customer_hash, anomaly_score, is_anomaly, anomaly_flag]

    Raises:
        TypeError: If model does not have required methods (score_samples, predict).
    """
    if not hasattr(model, "score_samples") or not hasattr(model, "predict"):
        raise TypeError("Model must have 'score_samples' and 'predict' methods for anomaly scoring")

    raw_scores = model.score_samples(X)  # type: ignore[attr-defined]
    normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    anomaly_scores = 1.0 - normalized

    predictions = model.predict(X)  # type: ignore[attr-defined]

    result = pd.DataFrame(
        {
            "customer_hash": customer_hashes.values,
            "anomaly_score": anomaly_scores,
            "is_anomaly": predictions == -1,
        }
    )

    def map_flag(score: float) -> str:
        if score >= anomaly_threshold:
            return AnomalyFlag.anomalous.value
        elif score >= anomaly_threshold * 0.7:
            return AnomalyFlag.suspicious.value
        else:
            return AnomalyFlag.normal.value

    result["anomaly_flag"] = result["anomaly_score"].apply(map_flag)

    return result


def compute_anomaly_shap(
    model: object,
    X: pd.DataFrame,
    customer_hashes: pd.Series,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Compute SHAP values for anomaly explanations.

    SHAP computation is expensive. max_samples limits computation to a
    representative subset. For full dataset explanation, increase max_samples
    but expect significantly longer runtime.

    SHAP values are returned as a JSON string column for portability
    across SQL, Parquet, and API serialization.

    Args:
        model: Trained IsolationForest model.
        X: Feature matrix for SHAP computation.
        customer_hashes: Customer identifiers corresponding to X rows.
        max_samples: Maximum number of samples to compute SHAP values for.

    Returns:
        DataFrame with columns: [customer_hash, shap_values] where shap_values
        is a JSON string encoding {feature_name: shap_value} for the top 3
        features by absolute SHAP value.
        Returns empty DataFrame with the same columns if SHAP fails.
    """
    import shap  # type: ignore[import-untyped]

    try:
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
            sampled_indices = X_sample.index
            customer_hashes_sample = customer_hashes.loc[sampled_indices]
        else:
            X_sample = X
            customer_hashes_sample = customer_hashes

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        results = []
        for i, (_idx, _row) in enumerate(X_sample.iterrows()):
            shap_row = shap_values[i] if len(shap_values.shape) > 1 else shap_values
            feature_importance = sorted(enumerate(shap_row), key=lambda x: abs(x[1]), reverse=True)[
                :3
            ]

            shap_dict: dict[str, float] = {
                ANOMALY_FEATURES[feat_idx]: float(shap_val)
                for feat_idx, shap_val in feature_importance
            }
            results.append(
                {
                    "customer_hash": customer_hashes_sample.iloc[i],
                    "shap_values": json.dumps(shap_dict),
                }
            )

        return pd.DataFrame(results)

    except Exception as e:
        warnings.warn(f"SHAP computation failed: {e}. Returning empty DataFrame.", stacklevel=2)
        return pd.DataFrame({"customer_hash": [], "shap_values": []})
