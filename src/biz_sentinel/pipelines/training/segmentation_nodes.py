"""Node functions for customer segmentation using K-Means clustering.

This module contains functions for segmenting customers into business-meaningful
groups using K-Means clustering with MLflow experiment tracking.
"""

import pandas as pd
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import davies_bouldin_score, silhouette_score  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler

from biz_sentinel.domain.models import SegmentLabel

SEGMENTATION_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "avg_review_score",
    "payment_installments_avg",
    "late_delivery_rate",
]


def prepare_segmentation_features(
    feature_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Extract and scale features for customer segmentation.

    Validates that all required features are present and contain no null values,
    then applies StandardScaler for K-Means which is distance-based.

    Args:
        feature_matrix: DataFrame with customer features including SEGMENTATION_FEATURES
            and customer_hash column.

    Returns:
        Tuple of (X_scaled, customer_hashes, scaler) where X_scaled is the standardized
        feature matrix, customer_hashes is a Series of identifiers, and scaler is the
        fitted StandardScaler.

    Raises:
        ValueError: If required features are missing or contain null values.

    Note:
        Scaler is returned so it can be persisted and reused at inference time
        without refitting on new data.
    """
    missing_features = [f for f in SEGMENTATION_FEATURES if f not in feature_matrix.columns]
    if missing_features:
        raise ValueError(f"Missing required features for segmentation: {missing_features}")

    X_raw = feature_matrix[SEGMENTATION_FEATURES].copy()
    customer_hashes: pd.Series = feature_matrix["customer_hash"]  # type: ignore[assignment]

    null_columns = X_raw.columns[X_raw.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Null values detected in feature columns: {null_columns}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=SEGMENTATION_FEATURES,
        index=X_raw.index,
    )

    return X_scaled, customer_hashes, scaler


def find_optimal_k(
    X_scaled: pd.DataFrame,
    k_min: int,
    k_max: int,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate multiple K values to find optimal cluster count.

    Computes inertia, silhouette score, and Davies-Bouldin score for each k.
    Silhouette measures how similar points are to their own cluster vs other clusters
    (higher is better, range -1 to 1). Davies-Bouldin measures average similarity between
    clusters (lower is better, range 0 to inf). Inertia measures within-cluster sum of
    squares (lower is better, but should plateau).

    Args:
        X_scaled: Standardized feature matrix.
        k_min: Minimum number of clusters to evaluate.
        k_max: Maximum number of clusters to evaluate.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with keys: k_values, inertias, silhouette_scores, davies_bouldin_scores.

    Note:
        This function is intended for exploratory use. Review the returned metrics
        to select k before running train_kmeans with the chosen value.
    """
    k_values = []
    inertias = []
    silhouette_scores_list = []
    davies_bouldin_scores_list = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)

        k_values.append(k)
        inertias.append(km.inertia_)
        silhouette_scores_list.append(silhouette_score(X_scaled, km.labels_))
        davies_bouldin_scores_list.append(davies_bouldin_score(X_scaled, km.labels_))

    return pd.DataFrame({
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "davies_bouldin_scores": davies_bouldin_scores_list,
    })


def train_kmeans(
    X_scaled: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    experiment_name: str = "biz_sentinel/segmentation",
) -> tuple[object, dict[str, float]]:
    """Train K-Means clustering model for customer segmentation.

    MLflow logging is a side effect of this function. The MLflow tracking URI
    must be configured before calling this function (via MLFLOW_TRACKING_URI
    env var or mlflow.set_tracking_uri()).

    Args:
        X_scaled: Standardized feature matrix.
        n_clusters: Number of clusters to create.
        random_state: Random seed for reproducibility.
        experiment_name: MLflow experiment name for tracking.

    Returns:
        Tuple of (model, metrics) where model is the trained KMeans and metrics
        is a dictionary with clustering quality metrics.
    """
    import mlflow  # type: ignore[import-untyped]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="kmeans_training"):
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(SEGMENTATION_FEATURES))
        mlflow.log_param("features", str(SEGMENTATION_FEATURES))
        mlflow.log_param("n_samples", len(X_scaled))

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        model.fit(X_scaled)

        silhouette = silhouette_score(X_scaled, model.labels_)
        db_score = davies_bouldin_score(X_scaled, model.labels_)
        inertia = float(model.inertia_)

        metrics = {
            "silhouette_score": float(silhouette),
            "davies_bouldin_score": float(db_score),
            "inertia": inertia,
            "n_clusters": float(n_clusters),
        }

        mlflow.log_metrics(metrics)
        import mlflow.sklearn  # type: ignore[import-untyped]

        mlflow.sklearn.log_model(  # type: ignore[reportPrivateImportUsage]
            model,
            artifact_path="kmeans",
            registered_model_name="biz_sentinel_segmentation",
        )

    return model, metrics


def assign_segments(
    model: object,
    X_scaled: pd.DataFrame,
    customer_hashes: pd.Series,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Assign cluster labels and map to business segment labels.

    Uses centroid-based heuristic to map numeric K-Means clusters to business-meaningful
    SegmentLabel values based on RFM profiles.

    Args:
        model: Trained KMeans model.
        X_scaled: Standardized feature matrix used for prediction.
        customer_hashes: Customer identifiers corresponding to X_scaled rows.
        feature_matrix: Original (unscaled) feature matrix for centroid computation.

    Returns:
        DataFrame with columns: [customer_hash, cluster_id, segment_label]

    Raises:
        TypeError: If model does not have a predict method.
    """
    if not hasattr(model, "predict"):
        raise TypeError("Model must have 'predict' method for cluster assignment")

    labels = model.predict(X_scaled)  # type: ignore[attr-defined]

    cluster_profiles = feature_matrix.copy()
    cluster_profiles["cluster"] = labels
    centroids = cluster_profiles.groupby("cluster")[SEGMENTATION_FEATURES].mean()

    percentiles = {}
    for feat in SEGMENTATION_FEATURES:
        percentiles[feat] = {
            "p25": feature_matrix[feat].quantile(0.25),
            "p50": feature_matrix[feat].quantile(0.50),
            "p75": feature_matrix[feat].quantile(0.75),
            "p90": feature_matrix[feat].quantile(0.90),
        }

    cluster_to_label: dict[int, str] = {}

    for cluster_id, centroid in centroids.iterrows():
        scores: dict[SegmentLabel, float] = {
            SegmentLabel.champions: 0.0,
            SegmentLabel.loyal: 0.0,
            SegmentLabel.at_risk: 0.0,
            SegmentLabel.new_customers: 0.0,
            SegmentLabel.hibernating: 0.0,
            SegmentLabel.lost: 0.0,
        }

        recency = centroid["recency_days"]
        freq = centroid["frequency"]
        monetary = centroid["monetary_total"]

        p = percentiles

        is_high_freq = freq >= p["frequency"]["p75"]
        is_high_monetary = monetary >= p["monetary_total"]["p75"]
        is_low_recency = recency <= p["recency_days"]["p25"]
        if is_high_freq and is_high_monetary and is_low_recency:
            scores[SegmentLabel.champions] += 3
        if freq >= p["frequency"]["p50"]:
            scores[SegmentLabel.loyal] += 2
        if freq >= p["frequency"]["p25"] and recency >= p["recency_days"]["p50"]:
            scores[SegmentLabel.at_risk] += 2
        if freq <= 2 and recency <= p["recency_days"]["p50"]:
            scores[SegmentLabel.new_customers] += 2
        if freq <= p["frequency"]["p50"] and recency >= p["recency_days"]["p50"]:
            scores[SegmentLabel.hibernating] += 2
        if recency >= p["recency_days"]["p90"] and freq <= p["frequency"]["p25"]:
            scores[SegmentLabel.lost] += 3

        best_label = max(scores, key=scores.get)  # type: ignore[arg-type]
        cluster_to_label[cluster_id] = best_label.value

    segment_labels = [cluster_to_label[label] for label in labels]

    result = pd.DataFrame(
        {
            "customer_hash": customer_hashes.values,
            "cluster_id": labels,
            "segment_label": segment_labels,
        }
    )

    return result


def compute_segment_profiles(
    feature_matrix: pd.DataFrame,
    segment_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Compute aggregated profiles for each customer segment.

    Groups customers by segment_label and computes mean values for all RFM
    and behavioral features. Results are sorted by avg_monetary_total descending.

    Args:
        feature_matrix: Original feature matrix with customer features.
        segment_assignments: DataFrame with segment assignments from assign_segments.

    Returns:
        DataFrame with one row per segment_label containing count and mean
        feature values, sorted by avg_monetary_total descending.
    """
    feature_cols = [c for c in feature_matrix.columns if c != "segment_label"]
    merged = feature_matrix[feature_cols].merge(segment_assignments, on="customer_hash")

    profiles = merged.groupby("segment_label").agg(
        count=("customer_hash", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary_total=("monetary_total", "mean"),
        avg_monetary_avg=("monetary_avg", "mean"),
        avg_review_score=("avg_review_score", "mean"),
        avg_late_delivery_rate=("late_delivery_rate", "mean"),
    )

    profiles = profiles.sort_values("avg_monetary_total", ascending=False).reset_index()  # type: ignore[call-overload]

    return profiles
