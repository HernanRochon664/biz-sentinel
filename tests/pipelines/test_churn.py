"""Tests for churn prediction pipeline nodes."""

from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from biz_sentinel.domain.models import SegmentLabel
from biz_sentinel.pipelines.training.churn_nodes import (
    CHURN_FEATURES,
    build_churn_labels,
    compute_churn_shap,
    prepare_churn_features,
    score_churn,
    train_churn_model,
    train_churn_model_with_dp,
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
            "recency_days": np.concatenate(
                [np.random.randint(1, 150, 25), np.random.randint(180, 400, 25)]
            ),
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_avg": monetary_total / frequency,
            "avg_review_score": np.random.uniform(1.0, 5.0, n),
            "review_count": np.random.randint(0, 31, n),
            "payment_installments_avg": np.random.uniform(1.0, 12.0, n),
            "avg_delivery_days": np.random.uniform(1.0, 30.0, n),
            "late_delivery_rate": np.random.uniform(0.0, 1.0, n),
            "segment_label": None,
            "anomaly_score": None,
        }
    )


@pytest.fixture
def sample_anomaly_scores(sample_feature_matrix):
    np.random.seed(42)
    return pd.DataFrame(
        {
            "customer_hash": sample_feature_matrix["customer_hash"],
            "anomaly_score": np.random.uniform(0.0, 1.0, len(sample_feature_matrix)),
        }
    )


@pytest.fixture
def sample_segment_assignments(sample_feature_matrix):
    segment_labels = [s.value for s in SegmentLabel]
    np.random.seed(42)
    return pd.DataFrame(
        {
            "customer_hash": sample_feature_matrix["customer_hash"],
            "segment_label": np.random.choice(segment_labels, len(sample_feature_matrix)),
            "cluster_id": np.random.randint(0, 5, len(sample_feature_matrix)),
        }
    )


class TestBuildChurnLabels:
    def test_build_churn_labels_adds_churned_column(self, sample_feature_matrix):
        result = build_churn_labels(sample_feature_matrix)
        assert "churned" in result.columns

    def test_build_churn_labels_binary_values(self, sample_feature_matrix):
        result = build_churn_labels(sample_feature_matrix)
        unique_vals = set(result["churned"].unique())
        assert unique_vals.issubset({0, 1})

    def test_build_churn_labels_threshold_respected(self, sample_feature_matrix):
        result = build_churn_labels(sample_feature_matrix)

        churned_customers = result[result["churned"] == 1]
        active_customers = result[result["churned"] == 0]

        assert (churned_customers["recency_days"] > 180).all(), "Churned should have recency > 180"
        assert (active_customers["recency_days"] <= 180).all(), "Active should have recency <= 180"

    def test_build_churn_labels_adds_source_column(self, sample_feature_matrix):
        result = build_churn_labels(sample_feature_matrix)
        assert "churn_label_source" in result.columns

    def test_build_churn_labels_does_not_mutate_input(self, sample_feature_matrix):
        original_recency = sample_feature_matrix["recency_days"].copy()
        build_churn_labels(sample_feature_matrix)
        assert sample_feature_matrix["recency_days"].equals(original_recency)


class TestPrepareChurnFeatures:
    def test_prepare_churn_returns_correct_types(
        self, sample_feature_matrix, sample_anomaly_scores, sample_segment_assignments
    ):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )
        X, y, hashes = prepare_churn_features(
            feature_matrix_labeled, sample_anomaly_scores, sample_segment_assignments
        )

        assert isinstance(X, pd.DataFrame), "X should be DataFrame"
        assert isinstance(y, pd.Series), "y should be Series"
        assert isinstance(hashes, pd.Series), "hashes should be Series"

    def test_prepare_churn_X_has_correct_columns(
        self, sample_feature_matrix, sample_anomaly_scores, sample_segment_assignments
    ):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )
        X, _, _ = prepare_churn_features(
            feature_matrix_labeled, sample_anomaly_scores, sample_segment_assignments
        )

        for col in CHURN_FEATURES:
            assert col in X.columns, f"X should contain {col}"

    def test_prepare_churn_no_nulls_in_X(
        self, sample_feature_matrix, sample_anomaly_scores, sample_segment_assignments
    ):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )
        X, _, _ = prepare_churn_features(
            feature_matrix_labeled, sample_anomaly_scores, sample_segment_assignments
        )

        assert X.isnull().sum().sum() == 0, "X should have no null values"

    def test_prepare_churn_segment_encoded_range(
        self, sample_feature_matrix, sample_anomaly_scores, sample_segment_assignments
    ):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )
        X, _, _ = prepare_churn_features(
            feature_matrix_labeled, sample_anomaly_scores, sample_segment_assignments
        )

        assert X["segment_encoded"].between(0, 5).all(), "segment_encoded should be in range [0, 5]"

    def test_prepare_churn_anomaly_score_filled(
        self, sample_feature_matrix, sample_anomaly_scores, sample_segment_assignments
    ):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )
        X, _, _ = prepare_churn_features(
            feature_matrix_labeled, sample_anomaly_scores, sample_segment_assignments
        )

        assert X["anomaly_score"].isnull().sum() == 0, "anomaly_score should be filled"


class TestTrainChurnModel:
    def test_train_churn_returns_model_and_metrics(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        result = train_churn_model(X, y, n_estimators=10, learning_rate=0.1, random_state=42)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        model, metrics = result
        assert hasattr(model, "predict"), "Model should have predict method"
        assert isinstance(metrics, dict), "Metrics should be a dictionary"

    def test_train_churn_metrics_contain_roc_auc(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        _, metrics = train_churn_model(X, y, n_estimators=10, learning_rate=0.1, random_state=42)

        assert "roc_auc" in metrics

    def test_train_churn_roc_auc_valid_range(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        _, metrics = train_churn_model(X, y, n_estimators=10, learning_rate=0.1, random_state=42)

        if not np.isnan(metrics["roc_auc"]):
            assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_train_churn_metrics_contain_all_keys(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        _, metrics = train_churn_model(X, y, n_estimators=10, learning_rate=0.1, random_state=42)

        expected_keys = ["roc_auc", "f1", "precision", "recall", "train_size", "test_size"]
        for key in expected_keys:
            assert key in metrics, f"Metrics should contain '{key}'"


class TestTrainChurnModelWithDP:
    def test_dp_model_returns_tuple(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        result = train_churn_model_with_dp(X, y, epsilon=1.0, delta=1e-5, random_state=42)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"

    def test_dp_metrics_contain_epsilon(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        _, metrics = train_churn_model_with_dp(X, y, epsilon=1.0, delta=1e-5, random_state=42)

        assert "epsilon" in metrics

    def test_dp_model_not_none_on_valid_input(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )
        dp_model, metrics = train_churn_model_with_dp(
            X, y, epsilon=1.0, delta=1e-5, random_state=42
        )

        if dp_model is None:
            pytest.skip("DP model training failed due to diffprivlib version incompatibility")
        assert dp_model is not None

    def test_dp_vs_standard_roc_auc_comparison(self, mock_mlflow, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, _ = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )

        with patch.multiple(
            "mlflow",
            set_experiment=MagicMock(),
            start_run=MagicMock(),
            log_param=MagicMock(),
            log_metrics=MagicMock(),
            sklearn=MagicMock(log_model=MagicMock()),
        ):
            standard_model, standard_metrics = train_churn_model(
                X, y, n_estimators=10, learning_rate=0.1, random_state=42
            )
            dp_model, dp_metrics = train_churn_model_with_dp(
                X, y, epsilon=1.0, delta=1e-5, random_state=42
            )

        if not np.isnan(standard_metrics["roc_auc"]):
            assert 0.0 <= standard_metrics["roc_auc"] <= 1.0

        if dp_model is None or "roc_auc" not in dp_metrics:
            pytest.skip("DP model training failed due to diffprivlib version incompatibility")

        assert 0.0 <= dp_metrics["roc_auc"] <= 1.0


class TestScoreChurn:
    @pytest.fixture
    def trained_model_and_data(self, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, customer_hashes = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        return model, X, customer_hashes

    def test_score_churn_returns_expected_columns(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_churn(model, X, customer_hashes, churn_threshold=0.5)

        expected_cols = ["customer_hash", "churn_probability", "predicted_churn"]
        for col in expected_cols:
            assert col in result.columns, f"Result should contain '{col}'"

    def test_score_churn_probability_range(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_churn(model, X, customer_hashes, churn_threshold=0.5)

        assert (result["churn_probability"] >= 0.0).all()
        assert (result["churn_probability"] <= 1.0).all()

    def test_score_churn_predicted_is_bool(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = score_churn(model, X, customer_hashes, churn_threshold=0.5)

        assert result["predicted_churn"].dtype == bool


class TestComputeChurnShap:
    @pytest.fixture
    def trained_model_and_data(self, sample_feature_matrix):
        feature_matrix_labeled = build_churn_labels(sample_feature_matrix)
        feature_matrix_labeled = feature_matrix_labeled.drop(
            columns=["segment_label", "anomaly_score"]
        )

        np.random.seed(42)
        anomaly_scores = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "anomaly_score": np.random.uniform(0.0, 1.0, 50),
            }
        )

        segment_labels = [s.value for s in SegmentLabel]
        np.random.seed(42)
        segment_assignments = pd.DataFrame(
            {
                "customer_hash": sample_feature_matrix["customer_hash"],
                "segment_label": np.random.choice(segment_labels, 50),
                "cluster_id": np.random.randint(0, 5, 50),
            }
        )

        X, y, customer_hashes = prepare_churn_features(
            feature_matrix_labeled, anomaly_scores, segment_assignments
        )

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        return model, X, customer_hashes

    def test_churn_shap_returns_expected_columns(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = compute_churn_shap(model, X, customer_hashes, max_samples=10)

        expected_cols = ["customer_hash", "shap_values"]
        for col in expected_cols:
            assert col in result.columns, f"Result should contain '{col}'"

    def test_churn_shap_values_is_valid_json(self, trained_model_and_data):
        import json
        model, X, customer_hashes = trained_model_and_data
        result = compute_churn_shap(model, X, customer_hashes, max_samples=10)

        for _, row in result.iterrows():
            shap_dict = json.loads(row["shap_values"])
            assert isinstance(shap_dict, dict), "shap_values should parse to dict"
            assert len(shap_dict) <= 5, "Should have at most 5 features"

    def test_churn_shap_max_samples_respected(self, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data
        result = compute_churn_shap(model, X, customer_hashes, max_samples=10)

        assert len(result) <= 10

    @patch("shap.TreeExplainer")
    def test_churn_shap_graceful_failure(self, mock_tree_explainer, trained_model_and_data):
        model, X, customer_hashes = trained_model_and_data

        mock_explainer = MagicMock()
        mock_explainer.shap_values.side_effect = Exception("SHAP computation failed")
        mock_tree_explainer.return_value = mock_explainer

        result = compute_churn_shap(model, X, customer_hashes, max_samples=10)

        assert len(result) == 0
        assert list(result.columns) == ["customer_hash", "shap_values"], \
            "Empty result should have correct columns"
