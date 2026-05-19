"""Tests for training pipeline nodes."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from biz_sentinel.pipelines.training.nodes import (
    ANOMALY_FEATURES,
    compute_anomaly_shap,
    prepare_anomaly_features,
    score_anomalies,
    train_isolation_forest,
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
            "review_count": np.random.randint(0, 31, n),
            "payment_installments_avg": np.random.uniform(1.0, 12.0, n),
            "avg_delivery_days": np.random.uniform(1.0, 30.0, n),
            "late_delivery_rate": np.random.uniform(0.0, 1.0, n),
            "anomaly_score": None,
            "segment_label": None,
        }
    )


@pytest.fixture
def sample_feature_matrix_missing_column(sample_feature_matrix):
    return sample_feature_matrix.drop(columns=["recency_days"])


class TestPrepareAnomalyFeatures:
    def test_prepare_returns_correct_shapes(self, sample_feature_matrix):
        X, customer_hashes = prepare_anomaly_features(sample_feature_matrix)

        assert X.shape[0] == 50, "X should have 50 rows"
        assert X.shape[1] == len(ANOMALY_FEATURES), f"X should have {len(ANOMALY_FEATURES)} columns"
        assert len(customer_hashes) == 50, "customer_hashes should have 50 elements"

    def test_prepare_raises_on_missing_feature(self, sample_feature_matrix_missing_column):
        with pytest.raises(ValueError, match="Missing required features for anomaly detection"):
            prepare_anomaly_features(sample_feature_matrix_missing_column)

    def test_prepare_X_has_no_nulls(self, sample_feature_matrix):
        X, _ = prepare_anomaly_features(sample_feature_matrix)

        assert X.isnull().sum().sum() == 0, "X should have no null values"

    def test_prepare_customer_hashes_match_input(self, sample_feature_matrix):
        X, customer_hashes = prepare_anomaly_features(sample_feature_matrix)

        input_hashes = set(sample_feature_matrix["customer_hash"])
        output_hashes = set(customer_hashes)
        assert output_hashes == input_hashes, "Customer hashes should match input"


class TestTrainIsolationForest:
    def test_train_returns_model_and_metrics(self, mock_mlflow, sample_feature_matrix):
        X, _ = prepare_anomaly_features(sample_feature_matrix)
        result = train_isolation_forest(X, contamination=0.1, random_state=42)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        model, metrics = result
        assert hasattr(model, "predict"), "Model should have predict method"
        assert isinstance(metrics, dict), "Metrics should be a dictionary"

    def test_train_metrics_contain_expected_keys(self, mock_mlflow, sample_feature_matrix):
        X, _ = prepare_anomaly_features(sample_feature_matrix)
        _, metrics = train_isolation_forest(X, contamination=0.1, random_state=42)

        expected_keys = ["anomaly_rate", "avg_anomaly_score", "n_anomalies", "contamination_param"]
        for key in expected_keys:
            assert key in metrics, f"Metrics should contain '{key}'"

    def test_train_anomaly_rate_is_valid_probability(self, mock_mlflow, sample_feature_matrix):
        X, _ = prepare_anomaly_features(sample_feature_matrix)
        _, metrics = train_isolation_forest(X, contamination=0.1, random_state=42)

        assert 0.0 <= metrics["anomaly_rate"] <= 1.0, "anomaly_rate should be between 0 and 1"

    def test_train_model_has_predict_method(self, mock_mlflow, sample_feature_matrix):
        X, _ = prepare_anomaly_features(sample_feature_matrix)
        model, _ = train_isolation_forest(X, contamination=0.1, random_state=42)

        assert hasattr(model, "predict"), "Model should have predict method"


class TestScoreAnomalies:
    @pytest.fixture
    def trained_model_and_data(self, sample_feature_matrix):
        X, customer_hashes = prepare_anomaly_features(sample_feature_matrix)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        return model, X, customer_hashes

    def test_score_returns_expected_columns(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_anomalies(model, X, customer_hashes, anomaly_threshold=0.7)

        expected_cols = ["customer_hash", "anomaly_score", "is_anomaly", "anomaly_flag"]
        for col in expected_cols:
            assert col in result.columns, f"Result should contain '{col}'"

    def test_score_anomaly_score_range(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_anomalies(model, X, customer_hashes, anomaly_threshold=0.7)

        scores = result["anomaly_score"]
        assert (scores >= 0.0).all(), "All anomaly scores should be >= 0.0"
        assert (scores <= 1.0).all(), "All anomaly scores should be <= 1.0"

    def test_score_anomaly_flag_valid_values(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_anomalies(model, X, customer_hashes, anomaly_threshold=0.7)

        valid_flags = {"normal", "suspicious", "anomalous"}
        all_valid = result["anomaly_flag"].isin(valid_flags).all()  # type: ignore[union]
        assert bool(all_valid), "All anomaly_flag values should be valid"

    def test_score_customer_hash_preserved(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_anomalies(model, X, customer_hashes, anomaly_threshold=0.7)

        assert set(result["customer_hash"]) == set(customer_hashes), (
            "Customer hashes should be preserved"
        )

    def test_score_invalid_model_raises(self, sample_feature_matrix):
        X, customer_hashes = prepare_anomaly_features(sample_feature_matrix)
        invalid_model = object()

        with pytest.raises(TypeError, match="Model must have"):
            score_anomalies(invalid_model, X, customer_hashes, anomaly_threshold=0.7)


class TestComputeAnomalyShap:
    @pytest.fixture
    def trained_model_and_data(self, sample_feature_matrix):
        X, customer_hashes = prepare_anomaly_features(sample_feature_matrix)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        return model, X, customer_hashes

    def test_shap_returns_expected_columns(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = compute_anomaly_shap(model, X, customer_hashes, max_samples=10)

        expected_cols = [
            "customer_hash",
            "shap_feature_1",
            "shap_value_1",
            "shap_feature_2",
            "shap_value_2",
            "shap_feature_3",
            "shap_value_3",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Result should contain '{col}'"

    def test_shap_does_not_raise_on_valid_input(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        try:
            compute_anomaly_shap(model, X, customer_hashes, max_samples=10)
        except Exception as e:
            pytest.fail(f"Function should not raise exception: {e}")

    def test_shap_max_samples_limits_rows(self, sample_feature_matrix):
        X, customer_hashes = prepare_anomaly_features(sample_feature_matrix)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)

        result = compute_anomaly_shap(model, X, customer_hashes, max_samples=10)
        assert len(result) <= 10, "Result should have at most max_samples rows"

    @patch("shap.TreeExplainer")
    def test_shap_returns_empty_df_gracefully(self, mock_tree_explainer, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data

        mock_explainer = MagicMock()
        mock_explainer.shap_values.side_effect = Exception("SHAP computation failed")
        mock_tree_explainer.return_value = mock_explainer

        result = compute_anomaly_shap(model, X, customer_hashes, max_samples=10)

        assert len(result) == 0, "Result should be empty when SHAP fails"
        expected_cols = ["customer_hash", "shap_feature_1", "shap_value_1",
                         "shap_feature_2", "shap_value_2", "shap_feature_3", "shap_value_3"]
        for col in expected_cols:
            assert col in result.columns, f"Result should contain '{col}' even when empty"
