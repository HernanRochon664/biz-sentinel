"""Node functions for churn/risk scoring module.

This module contains functions for training churn prediction models using
LightGBM with MLflow experiment tracking. It integrates outputs from
Module A (anomaly detection) and Module B (customer segmentation) as
additional features, with SHAP-based interpretability.
"""

import warnings
from typing import Any

import pandas as pd


# Return a sentinel object instead of None — Kedro cannot save None to MemoryDataset
class _DPFailedSentinel:
    """Placeholder when DP training fails due to library incompatibility."""

    pass


CHURN_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "avg_review_score",
    "review_count",
    "payment_installments_avg",
    "avg_delivery_days",
    "late_delivery_rate",
    "anomaly_score",
    "segment_encoded",
]

CHURN_DAYS_THRESHOLD: int = 180

SEGMENT_ENCODING: dict[str, int] = {
    "champions": 0,
    "loyal": 1,
    "at_risk": 2,
    "new_customers": 3,
    "hibernating": 4,
    "lost": 5,
}


def build_churn_labels(
    feature_matrix: pd.DataFrame,
    churn_days_threshold: int = CHURN_DAYS_THRESHOLD,
) -> pd.DataFrame:
    """Build churn labels from recency using rule-based threshold.

    Churn labels are derived from recency_days using a rule-based threshold.
    This is a proxy for true churn — customers inactive for 180+ days are
    labeled as churned. Review the distribution before training.

    Args:
        feature_matrix: DataFrame with customer features including recency_days.
        churn_days_threshold: Number of days of inactivity before labeling as churned.

    Returns:
        Copy of feature_matrix with added columns: churned (int: 1 if churned,
        0 if active) and churn_label_source (str: "rule_based_recency").
    """
    churned = (feature_matrix["recency_days"] > churn_days_threshold).astype(int)

    print(f"Churn label distribution: {churned.value_counts().to_dict()}")
    print(f"Churn rate: {churned.mean():.2%}")

    result = feature_matrix.copy()
    result["churned"] = churned
    result["churn_label_source"] = "rule_based_recency"

    return result


def prepare_churn_features(
    feature_matrix_with_labels: pd.DataFrame,
    anomaly_scores: pd.DataFrame,
    segment_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare features for churn prediction by merging module outputs.

    Merges feature matrix with anomaly scores from Module A and segment
    assignments from Module B, then encodes segments and validates features.

    Args:
        feature_matrix_with_labels: DataFrame with customer features and churn labels.
        anomaly_scores: DataFrame with customer_hash and anomaly_score columns.
        segment_assignments: DataFrame with customer_hash and segment_label columns.

    Returns:
        Tuple of (X, y, customer_hashes) where X is the feature matrix with
        CHURN_FEATURES, y is the churn labels, and customer_hashes identifies each row.

    Raises:
        ValueError: If required features are missing or contain null values.
    """

    cols_to_drop = ["segment_label", "anomaly_score"]
    feature_matrix_with_labels = feature_matrix_with_labels.drop(
        columns=[c for c in cols_to_drop if c in feature_matrix_with_labels.columns],
        errors="ignore",
    )

    merged = feature_matrix_with_labels.merge(
        anomaly_scores[["customer_hash", "anomaly_score"]],
        on="customer_hash",
        how="left",
    )

    merged = merged.merge(
        segment_assignments[["customer_hash", "segment_label"]],
        on="customer_hash",
        how="left",
    )

    merged["segment_label"] = merged["segment_label"].fillna("hibernating")
    merged["segment_encoded"] = merged["segment_label"].map(SEGMENT_ENCODING)

    merged["anomaly_score"] = merged["anomaly_score"].fillna(0.5)

    missing_features = [f for f in CHURN_FEATURES if f not in merged.columns]
    if missing_features:
        raise ValueError(f"Missing required features for churn prediction: {missing_features}")

    X = merged[CHURN_FEATURES].copy()
    y = merged["churned"]
    customer_hashes = merged["customer_hash"]

    null_columns = X.columns[X.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Null values detected in feature columns: {null_columns}")

    return X, y, customer_hashes  # type: ignore[return-value]


def train_churn_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
    experiment_name: str = "biz_sentinel/churn_scoring",
) -> tuple[Any, dict[str, float]]:
    """Train LightGBM classifier for churn prediction with MLflow tracking.

    MLflow logging is a side effect of this function. The MLflow tracking URI
    must be configured before calling this function (via MLFLOW_TRACKING_URI
    env var or mlflow.set_tracking_uri()).

    Args:
        X: Feature matrix with CHURN_FEATURES.
        y: Binary churn labels (0 = active, 1 = churned).
        n_estimators: Number of boosting iterations.
        learning_rate: Learning rate for boosting.
        random_state: Random seed for reproducibility.
        experiment_name: MLflow experiment name for tracking.

    Returns:
        Tuple of (model, metrics) where model is the trained LGBMClassifier
        and metrics is a dictionary with classification metrics.
    """
    import lightgbm as lgb  # type: ignore[import-untyped]
    import mlflow  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Temporal split preserves order. Assumes feature_matrix is
    # sorted by recency (most recent customers last). This avoids data leakage
    # from future to past.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="lgbm_churn_training"):
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(CHURN_FEATURES))
        mlflow.log_param("features", str(CHURN_FEATURES))
        mlflow.log_param("split_strategy", "temporal_80_20")

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]  # type: ignore[index]
        y_pred = model.predict(X_test)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "train_size": float(len(X_train)),
            "test_size": float(len(X_test)),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test": float(y_test.mean()),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(  # type: ignore[attr-defined]
            model,
            artifact_path="lgbm_churn",
            registered_model_name="biz_sentinel_churn_scorer",
        )

    return model, metrics


def train_churn_model_with_dp(
    X: pd.DataFrame,
    y: pd.Series,
    epsilon: float,
    delta: float,
    random_state: int,
    experiment_name: str = "biz_sentinel/churn_scoring_dp",
) -> tuple[Any | None, dict[str, float]]:
    """Train differentially private Logistic Regression for churn prediction.

    Differential Privacy adds calibrated noise during training to protect
    individual records. Epsilon controls the privacy-utility tradeoff: lower epsilon =
    more privacy, more noise, lower accuracy. Compare metrics with train_churn_model
    to quantify the privacy cost.

    This function serves as a comparison baseline, not the primary model.

    Args:
        X: Feature matrix with CHURN_FEATURES.
        y: Binary churn labels (0 = active, 1 = churned).
        epsilon: Privacy parameter controlling noise magnitude.
        delta: Privacy parameter (failure probability).
        random_state: Random seed for reproducibility.
        experiment_name: MLflow experiment name for tracking.

    Returns:
        Tuple of (model, metrics) where model is the trained DP LogisticRegression
        (or None if training fails) and metrics is a dictionary with classification
        metrics including epsilon and delta values.
    """
    import mlflow  # type: ignore[import-untyped]
    from diffprivlib.models import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_experiment(experiment_name)
    run_name = f"dp_logistic_regression_eps_{epsilon}"

    try:
        dp_model = LogisticRegression(
            epsilon=epsilon,
            data_norm=1.0,
            random_state=random_state,
            max_iter=200,
        )
        dp_model.fit(X_train_scaled, y_train)

        y_pred_proba = dp_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = dp_model.predict(X_test_scaled)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "epsilon": epsilon,
            "delta": delta,
            "model_type": 1.0,
            "train_size": float(len(X_train)),
            "test_size": float(len(X_test)),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test": float(y_test.mean()),
        }

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", "DP_LogisticRegression")
            mlflow.log_param("epsilon", epsilon)
            mlflow.log_param("delta", delta)
            mlflow.log_param("split_strategy", "temporal_80_20")
            mlflow.log_param("n_features", len(CHURN_FEATURES))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(  # type: ignore[attr-defined]
                dp_model,
                artifact_path="dp_logistic_churn",
                registered_model_name="biz_sentinel_churn_scorer_dp",
            )

        return dp_model, metrics

    except Exception as e:
        warnings.warn(f"DP model training failed: {e}. Returning empty metrics.", stacklevel=2)
        # Return a sentinel object instead of None — Kedro cannot save None to MemoryDataset
        return (_DPFailedSentinel(), {"dp_training_failed": 1.0, "epsilon": epsilon})


def score_churn(
    model: object,
    X: pd.DataFrame,
    customer_hashes: pd.Series,
    churn_threshold: float,
) -> pd.DataFrame:
    """Score customers for churn risk using trained model.

    Args:
        model: Trained model with predict_proba method.
        X: Feature matrix for scoring.
        customer_hashes: Customer identifiers corresponding to X rows.
        churn_threshold: Threshold for predicting churn (probability >= threshold = churned).

    Returns:
        DataFrame with columns: [customer_hash, churn_probability, predicted_churn]

    Raises:
        TypeError: If model does not have required predict_proba method.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must have 'predict_proba' method for churn scoring")

    churn_probabilities = model.predict_proba(X)[:, 1]  # type: ignore[attr-defined, index]

    result = pd.DataFrame(
        {
            "customer_hash": customer_hashes.values,
            "churn_probability": churn_probabilities,
            "predicted_churn": churn_probabilities >= churn_threshold,
        }
    )

    return result


def compute_churn_shap(
    model: object,
    X: pd.DataFrame,
    customer_hashes: pd.Series,
    feature_names: list[str] | None = None,
    max_samples: int = 1000,
) -> pd.DataFrame:
    """Compute SHAP values for churn predictions.

    LightGBM SHAP values are signed: positive = increases churn probability,
    negative = decreases it. The magnitude indicates feature importance for
    this specific customer.

    SHAP computation is expensive. max_samples limits computation to a
    representative subset.

    Args:
        model: Trained LightGBM model.
        X: Feature matrix for SHAP computation.
        customer_hashes: Customer identifiers corresponding to X rows.
        feature_names: List of feature names (defaults to CHURN_FEATURES).
        max_samples: Maximum number of samples to compute SHAP values for.

    Returns:
        DataFrame with columns: [customer_hash, shap_feature_1..5, shap_value_1..5]
        Returns empty DataFrame with correct column structure if SHAP computation fails.
    """
    import shap  # type: ignore[import-untyped]

    if feature_names is None:
        feature_names = CHURN_FEATURES

    columns = ["customer_hash"]
    for i in range(1, 6):
        columns.append(f"shap_feature_{i}")
        columns.append(f"shap_value_{i}")

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
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        results = []
        for i, (_idx, _row) in enumerate(X_sample.iterrows()):
            shap_row = shap_values[i] if len(shap_values.shape) > 1 else shap_values
            feature_importance = sorted(enumerate(shap_row), key=lambda x: abs(x[1]), reverse=True)[
                :5
            ]

            result_row: dict[str, Any] = {"customer_hash": customer_hashes_sample.iloc[i]}
            for rank, (feat_idx, shap_val) in enumerate(feature_importance, start=1):
                result_row[f"shap_feature_{rank}"] = feature_names[feat_idx]
                result_row[f"shap_value_{rank}"] = float(shap_val)

            for rank in range(len(feature_importance) + 1, 6):
                result_row[f"shap_feature_{rank}"] = None
                result_row[f"shap_value_{rank}"] = None

            results.append(result_row)

        return pd.DataFrame(results)

    except Exception as e:
        warnings.warn(f"SHAP computation failed: {e}. Returning empty DataFrame.", stacklevel=2)
        return pd.DataFrame({col: [] for col in columns})
