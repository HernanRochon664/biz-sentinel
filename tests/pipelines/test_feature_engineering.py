"""Tests for feature engineering pipeline nodes."""

import pandas as pd
import pytest

from biz_sentinel.domain.models import CustomerFeatures
from biz_sentinel.pipelines.feature_engineering.nodes import (
    assemble_feature_matrix,
    compute_delivery_features,
    compute_review_features,
    compute_rfm,
)


@pytest.fixture
def snapshot_date_str():
    return "2018-10-01"


@pytest.fixture
def sample_transactions():
    return pd.DataFrame(
        {
            "order_id": [
                "ord_A1",
                "ord_A2",
                "ord_A3",
                "ord_A4",
                "ord_A5",
                "ord_B1",
                "ord_B2",
                "ord_B3",
                "ord_C1",
                "ord_C2",
                "ord_C3",
                "ord_C4",
                "ord_C5",
                "ord_C6",
                "ord_C7",
            ],
            "customer_id": [
                "cust_A",
                "cust_A",
                "cust_A",
                "cust_A",
                "cust_A",
                "cust_B",
                "cust_B",
                "cust_B",
                "cust_C",
                "cust_C",
                "cust_C",
                "cust_C",
                "cust_C",
                "cust_C",
                "cust_C",
            ],
            "payment_value": [
                150.00,
                250.50,
                80.00,
                320.75,
                190.00,
                75.00,
                200.00,
                95.50,
                120.00,
                400.00,
                55.00,
                600.00,
                85.00,
                180.00,
                250.00,
            ],
            "payment_installments": [1, 3, 1, 2, 1, 1, 4, 1, 2, 6, 1, 3, 1, 2, 1],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2017-01-15 10:00:00",
                    "2017-06-20 14:00:00",
                    "2017-09-10 09:00:00",
                    "2018-01-05 11:00:00",
                    "2018-06-01 16:00:00",
                    "2018-01-01 10:00:00",
                    "2018-02-15 12:00:00",
                    "2018-03-10 15:00:00",
                    "2017-03-01 10:00:00",
                    "2017-08-15 11:00:00",
                    "2017-11-20 14:00:00",
                    "2018-02-01 09:00:00",
                    "2018-04-10 10:00:00",
                    "2018-07-15 12:00:00",
                    "2018-09-20 15:00:00",
                ]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                [
                    "2017-01-25 10:00:00",
                    "2017-06-30 10:00:00",
                    "2017-09-20 10:00:00",
                    "2018-01-15 10:00:00",
                    "2018-06-10 10:00:00",
                    "2018-01-10 10:00:00",
                    "2018-02-25 10:00:00",
                    "2018-03-20 10:00:00",
                    "2017-03-10 10:00:00",
                    "2017-08-25 10:00:00",
                    None,
                    "2018-02-10 10:00:00",
                    "2018-04-20 10:00:00",
                    "2018-07-25 10:00:00",
                    "2018-09-30 10:00:00",
                ]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                [
                    "2017-01-20 10:00:00",
                    "2017-06-25 10:00:00",
                    "2017-09-15 10:00:00",
                    "2018-01-10 10:00:00",
                    "2018-06-05 10:00:00",
                    "2018-01-08 10:00:00",
                    "2018-02-20 10:00:00",
                    "2018-03-15 10:00:00",
                    "2017-03-05 10:00:00",
                    "2017-08-20 10:00:00",
                    "2017-11-25 10:00:00",
                    "2018-02-05 10:00:00",
                    "2018-04-15 10:00:00",
                    "2018-07-20 10:00:00",
                    "2018-09-25 10:00:00",
                ]
            ),
            "order_status": [
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
                "canceled",
                "delivered",
                "delivered",
                "delivered",
                "delivered",
            ],
        }
    )


@pytest.fixture
def sample_reviews():
    return pd.DataFrame(
        {
            "review_id": [
                "rev_1",
                "rev_2",
                "rev_3",
                "rev_4",
                "rev_5",
                "rev_6",
                "rev_7",
                "rev_8",
                "rev_9",
                "rev_10",
            ],
            "order_id": [
                "ord_A1",
                "ord_A2",
                "ord_A3",
                "ord_B1",
                "ord_B2",
                "ord_C1",
                "ord_C2",
                "ord_C3",
                "ord_C4",
                "ord_C5",
            ],
            "review_score": [5, 4, 2, 3, 5, 1, 4, 3, 5, 2],
            "review_creation_date": pd.to_datetime(
                [
                    "2017-01-16",
                    "2017-06-21",
                    "2017-09-11",
                    "2018-01-02",
                    "2018-02-16",
                    "2017-03-02",
                    "2017-08-16",
                    "2017-11-21",
                    "2018-02-02",
                    "2018-04-11",
                ]
            ),
        }
    )


@pytest.fixture
def sample_orders_for_reviews():
    return pd.DataFrame(
        {
            "order_id": [
                "ord_A1",
                "ord_A2",
                "ord_A3",
                "ord_B1",
                "ord_B2",
                "ord_C1",
                "ord_C2",
                "ord_C3",
                "ord_C4",
                "ord_C5",
            ],
            "customer_id": [
                "cust_A",
                "cust_A",
                "cust_A",
                "cust_B",
                "cust_B",
                "cust_C",
                "cust_C",
                "cust_C",
                "cust_C",
                "cust_C",
            ],
        }
    )


class TestComputeRfm:
    def test_rfm_returns_expected_columns(self, sample_transactions, snapshot_date_str):
        result = compute_rfm(sample_transactions, snapshot_date_str)
        expected_cols = [
            "customer_id",
            "recency_days",
            "frequency",
            "monetary_total",
            "monetary_avg",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Expected column {col} in RFM output"

    def test_rfm_frequency_counts_orders(self, sample_transactions, snapshot_date_str):
        result = compute_rfm(sample_transactions, snapshot_date_str)
        freq_map = dict(zip(result["customer_id"], result["frequency"], strict=True))
        assert freq_map.get("cust_A") == 5, "Expected cust_A to have frequency 5"
        assert freq_map.get("cust_C") == 7, "Expected cust_C to have frequency 7"

    def test_rfm_recency_is_positive(self, sample_transactions, snapshot_date_str):
        result = compute_rfm(sample_transactions, snapshot_date_str)
        assert (result["recency_days"] >= 0).all(), "Expected all recency_days to be non-negative"

    def test_rfm_monetary_avg_equals_total_over_frequency(
        self, sample_transactions, snapshot_date_str
    ):
        result = compute_rfm(sample_transactions, snapshot_date_str)
        for _, row in result.iterrows():
            expected_avg = row["monetary_total"] / row["frequency"]
            assert abs(row["monetary_avg"] - expected_avg) < 0.01, (
                f"monetary_avg should equal monetary_total / frequency for {row['customer_id']}"
            )

    def test_rfm_empty_snapshot_date_raises(self, sample_transactions):
        with pytest.raises(ValueError, match="snapshot_date must be a non-empty date string"):
            compute_rfm(sample_transactions, snapshot_date="")

    def test_rfm_all_customers_present(self, sample_transactions, snapshot_date_str):
        result = compute_rfm(sample_transactions, snapshot_date_str)
        expected_customers = {"cust_A", "cust_B", "cust_C"}
        result_customers = set(result["customer_id"])
        assert expected_customers == result_customers, "Expected all 3 customer IDs in result"


class TestComputeReviewFeatures:
    def test_review_features_returns_expected_columns(
        self, sample_reviews, sample_orders_for_reviews
    ):
        result = compute_review_features(sample_reviews, sample_orders_for_reviews)
        expected_cols = ["customer_id", "avg_review_score", "review_count"]
        for col in expected_cols:
            assert col in result.columns, f"Expected column {col} in review features"

    def test_review_features_score_range(self, sample_reviews, sample_orders_for_reviews):
        result = compute_review_features(sample_reviews, sample_orders_for_reviews)
        assert (result["avg_review_score"] >= 1.0).all() and (
            result["avg_review_score"] <= 5.0
        ).all(), "Expected avg_review_score between 1.0 and 5.0 for all customers"


class TestComputeDeliveryFeatures:
    def test_delivery_features_returns_expected_columns(self, sample_transactions):
        result = compute_delivery_features(sample_transactions)
        expected_cols = ["customer_id", "avg_delivery_days", "late_delivery_rate"]
        for col in expected_cols:
            assert col in result.columns, f"Expected column {col} in delivery features"

    def test_delivery_late_delivery_rate_range(self, sample_transactions):
        result = compute_delivery_features(sample_transactions)
        assert (result["late_delivery_rate"] >= 0.0).all() and (
            result["late_delivery_rate"] <= 1.0
        ).all(), "Expected late_delivery_rate between 0.0 and 1.0"

    def test_delivery_avg_days_non_negative(self, sample_transactions):
        result = compute_delivery_features(sample_transactions)
        assert (result["avg_delivery_days"] >= 0).all(), (
            "Expected all avg_delivery_days to be non-negative"
        )


class TestAssembleFeatureMatrix:
    def test_assembly_no_null_values_in_required_columns(
        self, sample_transactions, sample_reviews, sample_orders_for_reviews, snapshot_date_str
    ):
        rfm = compute_rfm(sample_transactions, snapshot_date_str)
        review_features = compute_review_features(sample_reviews, sample_orders_for_reviews)
        delivery_features = compute_delivery_features(sample_transactions)
        payment_features = pd.DataFrame(
            {
                "customer_id": ["cust_A", "cust_B", "cust_C"],
                "payment_installments_avg": [1.6, 2.0, 1.67],
                "unique_product_categories": [0, 0, 0],
            }
        )

        result = assemble_feature_matrix(
            rfm, review_features, delivery_features, payment_features, snapshot_date_str
        )

        required_cols = [
            "recency_days",
            "frequency",
            "monetary_total",
            "avg_review_score",
            "review_count",
            "avg_delivery_days",
            "late_delivery_rate",
            "payment_installments_avg",
        ]
        null_counts = result[required_cols].isnull().sum()
        nulls_found = null_counts[null_counts > 0].to_dict()
        assert null_counts.sum() == 0, (
            f"Expected no nulls in required columns, but found: {nulls_found}"
        )

    def test_assembly_column_names_match_customer_features_contract(
        self, sample_transactions, sample_reviews, sample_orders_for_reviews, snapshot_date_str
    ):
        rfm = compute_rfm(sample_transactions, snapshot_date_str)
        review_features = compute_review_features(sample_reviews, sample_orders_for_reviews)
        delivery_features = compute_delivery_features(sample_transactions)
        payment_features = pd.DataFrame(
            {
                "customer_id": ["cust_A", "cust_B", "cust_C"],
                "payment_installments_avg": [1.6, 2.0, 1.67],
                "unique_product_categories": [0, 0, 0],
            }
        )

        result = assemble_feature_matrix(
            rfm, review_features, delivery_features, payment_features, snapshot_date_str
        )

        expected_fields = set(CustomerFeatures.model_fields.keys()) - {
            "anomaly_score",
            "segment_label",
        }
        result_fields = set(result.columns) - {"anomaly_score", "segment_label"}
        assert expected_fields.issubset(result_fields), (
            f"Expected CustomerFeatures fields {expected_fields} to be present in result columns"
        )

    def test_assembly_has_snapshot_date_column(
        self, sample_transactions, sample_reviews, sample_orders_for_reviews, snapshot_date_str
    ):
        rfm = compute_rfm(sample_transactions, snapshot_date_str)
        review_features = compute_review_features(sample_reviews, sample_orders_for_reviews)
        delivery_features = compute_delivery_features(sample_transactions)
        payment_features = pd.DataFrame(
            {
                "customer_id": ["cust_A", "cust_B", "cust_C"],
                "payment_installments_avg": [1.6, 2.0, 1.67],
                "unique_product_categories": [0, 0, 0],
            }
        )

        result = assemble_feature_matrix(
            rfm, review_features, delivery_features, payment_features, snapshot_date_str
        )

        assert "snapshot_date" in result.columns, "Expected snapshot_date column in result"
        assert pd.api.types.is_datetime64_any_dtype(result["snapshot_date"]), (
            "Expected snapshot_date to be parsed as datetime"
        )

    def test_assembly_customer_id_renamed_to_customer_hash(
        self, sample_transactions, sample_reviews, sample_orders_for_reviews, snapshot_date_str
    ):
        rfm = compute_rfm(sample_transactions, snapshot_date_str)
        review_features = compute_review_features(sample_reviews, sample_orders_for_reviews)
        delivery_features = compute_delivery_features(sample_transactions)
        payment_features = pd.DataFrame(
            {
                "customer_id": ["cust_A", "cust_B", "cust_C"],
                "payment_installments_avg": [1.6, 2.0, 1.67],
                "unique_product_categories": [0, 0, 0],
            }
        )

        result = assemble_feature_matrix(
            rfm, review_features, delivery_features, payment_features, snapshot_date_str
        )

        assert "customer_hash" in result.columns, "Expected customer_hash column in result"
        assert "customer_id" not in result.columns, (
            "Expected customer_id to be renamed, not present in result"
        )
