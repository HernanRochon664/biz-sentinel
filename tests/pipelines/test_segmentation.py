"""Tests for segmentation pipeline nodes."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from biz_sentinel.domain.models import SegmentLabel
from biz_sentinel.pipelines.training.segmentation_nodes import (
    SEGMENTATION_FEATURES,
    assign_segments,
    compute_segment_profiles,
    find_optimal_k,
    prepare_segmentation_features,
    train_kmeans,
)


@pytest.fixture
def mock_mlflow():
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=MagicMock())
    mock_run.__exit__ = MagicMock(return_value=False)

    mock_start_run = MagicMock(return_value=mock_run)
    mock_log_model = MagicMock()
    mock_log_metrics = MagicMock()
    mock_log_param = MagicMock()
    mock_set_experiment = MagicMock()

    with patch.multiple(
        "mlflow",
        set_experiment=mock_set_experiment,
        start_run=mock_start_run,
        log_param=mock_log_param,
        log_metrics=mock_log_metrics,
        sklearn=MagicMock(log_model=mock_log_model),
    ):
        yield


@pytest.fixture
def sample_feature_matrix():
    np.random.seed(42)
    n = 50
    monetary_total = np.random.uniform(100, 5000, n)
    frequency = np.random.randint(1, 21, n)

    return pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:03d}" for i in range(1, n + 1)],
            "snapshot_date": pd.Timestamp("2018-10-01"),
            "recency_days": np.random.randint(1, 366, n),
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_avg": monetary_total / frequency,
            "avg_review_score": np.random.uniform(1.0, 5.0, n),
            "payment_installments_avg": np.random.uniform(1.0, 12.0, n),
            "late_delivery_rate": np.random.uniform(0.0, 1.0, n),
            "anomaly_score": np.random.uniform(0.0, 1.0, n),
            "segment_label": None,
        }
    )


@pytest.fixture
def sample_feature_matrix_missing_column(sample_feature_matrix):
    return sample_feature_matrix.drop(columns=["frequency"])


class TestPrepareSegmentationFeatures:
    def test_prepare_returns_correct_tuple_types(self, sample_feature_matrix):
        result = prepare_segmentation_features(sample_feature_matrix)

        assert len(result) == 3, "Result should be a tuple of 3 elements"
        X_scaled, customer_hashes, scaler = result
        assert isinstance(X_scaled, pd.DataFrame), "First element should be DataFrame"
        assert isinstance(customer_hashes, pd.Series), "Second element should be Series"
        assert hasattr(scaler, "fit_transform"), "Scaler should have fit_transform method"

    def test_prepare_scaled_features_have_zero_mean(self, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)

        for col in X_scaled.columns:
            mean_val = X_scaled[col].mean()
            assert abs(mean_val) < 0.01, f"Column {col} should have mean ~0"

    def test_prepare_raises_on_missing_feature(self, sample_feature_matrix_missing_column):
        with pytest.raises(ValueError, match="Missing required features for segmentation"):
            prepare_segmentation_features(sample_feature_matrix_missing_column)

    def test_prepare_no_nulls_in_scaled(self, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)

        assert X_scaled.isnull().sum().sum() == 0, "X_scaled should have no null values"


class TestFindOptimalK:
    def test_find_optimal_k_returns_expected_keys(self, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        result = find_optimal_k(X_scaled, k_min=2, k_max=5, random_state=42)

        expected_keys = ["k_values", "inertias", "silhouette_scores", "davies_bouldin_scores"]
        for key in expected_keys:
            assert key in result, f"Result should contain '{key}'"

    def test_find_optimal_k_correct_length(self, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        result = find_optimal_k(X_scaled, k_min=2, k_max=4, random_state=42)

        expected_length = 3
        assert len(result["k_values"]) == expected_length
        assert len(result["inertias"]) == expected_length
        assert len(result["silhouette_scores"]) == expected_length
        assert len(result["davies_bouldin_scores"]) == expected_length

    def test_find_optimal_k_inertia_decreases(self, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        result = find_optimal_k(X_scaled, k_min=2, k_max=5, random_state=42)

        inertias = result["inertias"]
        for i in range(1, len(inertias)):
            assert inertias[i] <= inertias[i - 1], "Inertia should decrease with more clusters"

    def test_find_optimal_k_silhouette_range(self, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        result = find_optimal_k(X_scaled, k_min=2, k_max=4, random_state=42)

        for score in result["silhouette_scores"]:
            assert -1 <= score <= 1, "Silhouette score should be between -1 and 1"


class TestTrainKmeans:
    def test_train_kmeans_returns_model_and_metrics(self, mock_mlflow, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        result = train_kmeans(X_scaled, n_clusters=3, random_state=42)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        model, metrics = result

    def test_train_kmeans_metrics_keys(self, mock_mlflow, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        _, metrics = train_kmeans(X_scaled, n_clusters=3, random_state=42)

        expected_keys = ["silhouette_score", "davies_bouldin_score", "inertia", "n_clusters"]
        for key in expected_keys:
            assert key in metrics, f"Metrics should contain '{key}'"

    def test_train_kmeans_model_has_predict(self, mock_mlflow, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        model, _ = train_kmeans(X_scaled, n_clusters=3, random_state=42)

        assert hasattr(model, "predict"), "Model should have predict method"

    def test_train_kmeans_silhouette_valid(self, mock_mlflow, sample_feature_matrix):
        X_scaled, _, _ = prepare_segmentation_features(sample_feature_matrix)
        _, metrics = train_kmeans(X_scaled, n_clusters=3, random_state=42)

        assert -1 <= metrics["silhouette_score"] <= 1, "Silhouette score should be between -1 and 1"


class TestAssignSegments:
    @pytest.fixture
    def trained_kmeans_and_data(self, sample_feature_matrix):
        X = sample_feature_matrix[SEGMENTATION_FEATURES]
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=SEGMENTATION_FEATURES)

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X_scaled)

        customer_hashes = sample_feature_matrix["customer_hash"]

        return km, X_scaled, customer_hashes, sample_feature_matrix

    def test_assign_segments_returns_expected_columns(self, trained_kmeans_and_data):
        model, X_scaled, customer_hashes, feature_matrix = trained_kmeans_and_data
        result = assign_segments(model, X_scaled, customer_hashes, feature_matrix)

        expected_cols = ["customer_hash", "cluster_id", "segment_label"]
        for col in expected_cols:
            assert col in result.columns, f"Result should contain '{col}'"

    def test_assign_segments_all_customers_present(self, trained_kmeans_and_data):
        model, X_scaled, customer_hashes, feature_matrix = trained_kmeans_and_data
        result = assign_segments(model, X_scaled, customer_hashes, feature_matrix)

        assert len(result) == 50, "Result should have 50 rows"

    def test_assign_segments_valid_labels(self, trained_kmeans_and_data):
        model, X_scaled, customer_hashes, feature_matrix = trained_kmeans_and_data
        result = assign_segments(model, X_scaled, customer_hashes, feature_matrix)

        valid_labels = {s.value for s in SegmentLabel}
        all_valid = result["segment_label"].isin(valid_labels).all()
        assert bool(all_valid), "All segment_label values should be valid"

    def test_assign_segments_no_nulls(self, trained_kmeans_and_data):
        model, X_scaled, customer_hashes, feature_matrix = trained_kmeans_and_data
        result = assign_segments(model, X_scaled, customer_hashes, feature_matrix)

        assert result.isnull().sum().sum() == 0, "Result should have no null values"


class TestComputeSegmentProfiles:
    @pytest.fixture
    def segment_data(self, sample_feature_matrix):
        X = sample_feature_matrix[SEGMENTATION_FEATURES]
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=SEGMENTATION_FEATURES)

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X_scaled)

        customer_hashes = sample_feature_matrix["customer_hash"]
        segment_assignments = assign_segments(km, X_scaled, customer_hashes, sample_feature_matrix)

        return sample_feature_matrix, segment_assignments

    def test_profiles_returns_one_row_per_segment(self, segment_data):
        feature_matrix, segment_assignments = segment_data
        result = compute_segment_profiles(feature_matrix, segment_assignments)

        assert result["segment_label"].duplicated().sum() == 0, "No duplicate segment_label values"

    def test_profiles_has_count_column(self, segment_data):
        feature_matrix, segment_assignments = segment_data
        result = compute_segment_profiles(feature_matrix, segment_assignments)

        assert "count" in result.columns, "Result should have 'count' column"

    def test_profiles_count_sums_to_total(self, segment_data):
        feature_matrix, segment_assignments = segment_data
        result = compute_segment_profiles(feature_matrix, segment_assignments)

        assert result["count"].sum() == 50, "Total count should sum to 50"

    def test_profiles_sorted_by_monetary(self, segment_data):
        feature_matrix, segment_assignments = segment_data
        result = compute_segment_profiles(feature_matrix, segment_assignments)

        assert result["avg_monetary_total"].is_monotonic_decreasing
