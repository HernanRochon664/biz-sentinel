"""Tests for preprocessing pipeline nodes."""

import pandas as pd
import pytest

from biz_sentinel.pipelines.preprocessing.nodes import (
    build_transactions,
    clean_customers,
    clean_orders,
    clean_reviews,
    pseudonymize_customers,
)


@pytest.fixture
def sample_orders():
    return pd.DataFrame(
        {
            "order_id": [
                "ord_1",
                "ord_2",
                "ord_3",
                "ord_4",
                "ord_5",
                "ord_6",
                "ord_7",
                "ord_8",
                "ord_9",
                "ord_10",
            ],
            "customer_id": [
                "cust_a",
                "cust_b",
                "cust_c",
                "cust_d",
                "cust_e",
                "cust_f",
                "cust_g",
                "cust_h",
                "cust_i",
                "cust_j",
            ],
            "order_status": [
                "delivered",
                "canceled",
                "delivered",
                "invalid_status",
                "delivered",
                "delivered",
                "shipped",
                "canceled",
                "delivered",
                "delivered",
            ],
            "order_purchase_timestamp": [
                "2023-01-01 10:00:00",
                "2023-01-02 11:00:00",
                None,
                "2023-01-04 13:00:00",
                "2023-01-05 14:00:00",
                "2023-01-06 15:00:00",
                "2023-01-07 16:00:00",
                "2023-01-08 17:00:00",
                "2023-01-09 18:00:00",
                "2023-01-10 19:00:00",
            ],
            "order_approved_at": [
                "2023-01-01 10:05:00",
                "2023-01-02 11:10:00",
                "2023-01-04 13:15:00",
                "2023-01-05 14:20:00",
                "2023-01-06 15:25:00",
                "2023-01-07 16:30:00",
                "2023-01-08 17:35:00",
                "2023-01-09 18:40:00",
                "2023-01-10 19:45:00",
                "2023-01-11 20:00:00",
            ],
            "order_delivered_customer_date": [
                "2023-01-05 10:00:00",
                None,
                "2023-01-10 10:00:00",
                "2023-01-11 10:00:00",
                "2023-01-12 10:00:00",
                "2023-01-13 10:00:00",
                "2023-01-14 10:00:00",
                "2023-01-15 10:00:00",
                "2023-01-16 10:00:00",
                "2023-01-17 10:00:00",
            ],
            "order_estimated_delivery_date": [
                "2023-01-10 10:00:00",
                "2023-01-11 10:00:00",
                "2023-01-12 10:00:00",
                "2023-01-13 10:00:00",
                "2023-01-14 10:00:00",
                "2023-01-15 10:00:00",
                "2023-01-16 10:00:00",
                "2023-01-17 10:00:00",
                "2023-01-18 10:00:00",
                "2023-01-19 10:00:00",
            ],
        }
    )


@pytest.fixture
def sample_customers():
    return pd.DataFrame(
        {
            "customer_id": [
                "cust_1",
                "cust_2",
                "cust_3",
                "cust_4",
                "cust_5",
                "cust_6",
                "cust_7",
                "cust_8",
            ],
            "customer_unique_id": [
                "uniq_a",
                "uniq_b",
                "uniq_c",
                "uniq_a",
                "uniq_d",
                "uniq_e",
                "uniq_f",
                "uniq_f",
            ],
            "customer_zip_code_prefix": [
                "10000",
                "20000",
                "30000",
                "10000",
                "50000",
                "60000",
                "70000",
                "70000",
            ],
            "customer_city": [
                "São Paulo",
                " Rio de Janeiro ",
                "curitiba",
                "SÃO PAUL0",
                "belo horizonte",
                "Porto Alegre",
                "  Salvador  ",
                "recife",
            ],
            "customer_state": ["sp ", " rj ", "SP", "SP", "MG", "rs", " BA ", "PE"],
        }
    )


@pytest.fixture
def sample_customers_with_nulls():
    return pd.DataFrame(
        {
            "customer_id": [
                "cust_1",
                "cust_2",
                None,
                "cust_4",
                "cust_5",
                "cust_6",
                "cust_7",
                "cust_8",
            ],
            "customer_unique_id": [
                "uniq_a",
                "uniq_b",
                "uniq_c",
                "uniq_a",
                "uniq_d",
                "uniq_e",
                "uniq_f",
                "uniq_g",
            ],
            "customer_zip_code_prefix": [
                "10000",
                "20000",
                "30000",
                "10000",
                "50000",
                "60000",
                "70000",
                "80000",
            ],
            "customer_city": [
                "São Paulo",
                "Rio de Janeiro",
                "Curitiba",
                "São Paulo",
                "Belo Horizonte",
                "Porto Alegre",
                "Salvador",
                "Recife",
            ],
            "customer_state": ["SP", "RJ", "SP", "SP", "MG", "RS", "BA", "PE"],
        }
    )


@pytest.fixture
def sample_reviews():
    return pd.DataFrame(
        {
            "review_id": ["rev_1", "rev_2", "rev_3", "rev_4", "rev_5", "rev_6"],
            "order_id": ["ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "ord_6"],
            "review_score": [5, None, 1, 6, 3, 4],
            "review_creation_date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
            ],
            "review_answer_timestamp": [
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
            ],
        }
    )


@pytest.fixture
def sample_payments():
    return pd.DataFrame(
        {
            "order_id": ["ord_1", "ord_2", "ord_3", "ord_4", "ord_5"],
            "payment_sequential": [1, 1, 1, 1, 1],
            "payment_type": ["credit_card", "boleto", "credit_card", "debit_card", "credit_card"],
            "payment_installments": [1, 2, 3, 1, 4],
            "payment_value": [100.00, 50.00, 0.00, 75.50, 200.00],
        }
    )


@pytest.fixture
def sample_order_items():
    return pd.DataFrame(
        {
            "order_id": [
                "ord_1",
                "ord_2",
                "ord_3",
                "ord_4",
                "ord_5",
                "ord_6",
                "ord_7",
                "ord_8",
                "ord_9",
                "ord_10",
            ],
            "order_item_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "product_id": [
                "prod_a",
                "prod_b",
                "prod_c",
                "prod_d",
                "prod_e",
                "prod_f",
                "prod_g",
                "prod_h",
                "prod_i",
                "prod_j",
            ],
            "seller_id": [
                "sel_1",
                "sel_1",
                "sel_2",
                "sel_2",
                "sel_3",
                "sel_3",
                "sel_1",
                "sel_2",
                "sel_3",
                "sel_1",
            ],
            "price": [100.00, 150.00, 80.00, 200.00, 50.00, 120.00, 90.00, 180.00, 70.00, 110.00],
            "freight_value": [10.00, 15.00, 8.00, 20.00, 5.00, 12.00, 9.00, 18.00, 7.00, 11.00],
        }
    )


@pytest.fixture
def fake_salt():
    return "test_salt_do_not_use_in_production"


class TestCleanOrders:
    def test_clean_orders_filters_invalid_statuses(self, sample_orders):
        valid_statuses = ["delivered", "shipped"]
        result = clean_orders(sample_orders, valid_statuses)
        assert bool(result["order_status"].isin(valid_statuses).all()), (
            "Expected only valid statuses after cleaning"
        )

    def test_clean_orders_drops_null_timestamps(self, sample_orders):
        valid_statuses = ["delivered", "shipped"]
        result = clean_orders(sample_orders, valid_statuses)
        assert bool(result["order_purchase_timestamp"].notna().all()), (
            "Expected no nulls in purchase timestamp after clean"
        )

    def test_clean_orders_no_duplicate_order_ids(self, sample_orders):
        valid_statuses = ["delivered", "shipped"]
        result = clean_orders(sample_orders, valid_statuses)
        assert result["order_id"].is_unique, "Expected unique order_id after cleaning duplicates"

    def test_clean_orders_returns_dataframe(self, sample_orders):
        valid_statuses = ["delivered", "shipped"]
        result = clean_orders(sample_orders, valid_statuses)
        assert isinstance(result, pd.DataFrame), "Expected return type to be pd.DataFrame"


class TestCleanCustomers:
    def test_clean_customers_strips_whitespace(self, sample_customers):
        result = clean_customers(sample_customers)
        city_no_ws = result["customer_city"].str.strip() == result["customer_city"]
        state_no_ws = result["customer_state"].str.strip() == result["customer_state"]
        assert city_no_ws.all() and state_no_ws.all(), (
            "Expected no leading/trailing whitespace in city/state"
        )

    def test_clean_customers_uppercases_state(self, sample_customers):
        result = clean_customers(sample_customers)
        assert result["customer_state"].str.isupper().all(), (
            "Expected all state values to be uppercase"
        )

    def test_clean_customers_drops_null_ids(self, sample_customers_with_nulls):
        result = clean_customers(sample_customers_with_nulls)
        assert bool(result["customer_id"].notna().all()), (
            "Expected no nulls in customer_id after cleaning"
        )


class TestCleanReviews:
    def test_clean_reviews_score_range(self, sample_reviews):
        result = clean_reviews(sample_reviews)
        in_range = bool((result["review_score"] >= 1).all() and (result["review_score"] <= 5).all())
        assert in_range, "Expected all review_scores between 1 and 5"

    def test_clean_reviews_no_null_scores(self, sample_reviews):
        result = clean_reviews(sample_reviews)
        assert bool(result["review_score"].notna().all()), "Expected no null scores after cleaning"


class TestPseudonymizeCustomers:
    def test_pseudonymize_customers_removes_original_column(self, sample_customers, monkeypatch):
        monkeypatch.setenv("HMAC_SALT", "test_salt_123")
        cleaned = clean_customers(sample_customers)
        result = pseudonymize_customers(cleaned, {})
        assert "customer_id" not in result.columns, "Expected customer_id not in result columns"

    def test_pseudonymize_customers_adds_hash_column(self, sample_customers, monkeypatch):
        monkeypatch.setenv("HMAC_SALT", "test_salt_123")
        cleaned = clean_customers(sample_customers)
        result = pseudonymize_customers(cleaned, {})
        assert "customer_id_hash" in result.columns, "Expected customer_id_hash in result columns"

    def test_pseudonymize_customers_is_deterministic(self, sample_customers, monkeypatch):
        monkeypatch.setenv("HMAC_SALT", "test_salt_123")
        cleaned = clean_customers(sample_customers)
        result1 = pseudonymize_customers(cleaned, {})
        monkeypatch.setenv("HMAC_SALT", "test_salt_123")
        result2 = pseudonymize_customers(cleaned, {})
        assert (result1["customer_id_hash"] == result2["customer_id_hash"]).all(), (
            "Expected same salt to produce same hashes"
        )

    def test_pseudonymize_customers_different_salt_different_hash(
        self, sample_customers, monkeypatch
    ):
        monkeypatch.setenv("HMAC_SALT", "test_salt_123")
        cleaned = clean_customers(sample_customers)
        result1 = pseudonymize_customers(cleaned, {})
        monkeypatch.setenv("HMAC_SALT", "test_salt_456")
        result2 = pseudonymize_customers(cleaned, {})
        assert not (result1["customer_id_hash"] == result2["customer_id_hash"]).all(), (
            "Expected different salts to produce different hashes"
        )


class TestBuildTransactions:
    def test_build_transactions_has_expected_columns(
        self, sample_orders, sample_order_items, sample_payments
    ):
        valid_statuses = ["delivered", "shipped"]
        clean_ord = clean_orders(sample_orders, valid_statuses)
        clean_pay = sample_payments[sample_payments["payment_value"] > 0].copy()
        clean_pay = clean_pay.dropna(subset=["payment_type"])

        result = build_transactions(clean_ord, sample_order_items, clean_pay)
        expected_cols = [
            "order_id",
            "customer_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
            "payment_value",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Expected column {col} in transactions"

    def test_build_transactions_row_count_reasonable(
        self, sample_orders, sample_order_items, sample_payments
    ):
        valid_statuses = ["delivered", "shipped"]
        clean_ord = clean_orders(sample_orders, valid_statuses)
        clean_pay = sample_payments[sample_payments["payment_value"] > 0].copy()
        clean_pay = clean_pay.dropna(subset=["payment_type"])

        result = build_transactions(clean_ord, sample_order_items, clean_pay)
        assert result.shape[0] <= sample_order_items.shape[0], (
            "Expected at most len(order_items) rows in transactions"
        )


class TestEdgeCases:
    def test_clean_orders_empty_dataframe(self, monkeypatch):
        valid_statuses = ["delivered", "shipped"]
        empty_df = pd.DataFrame(
            columns=["order_id", "customer_id", "order_status", "order_purchase_timestamp"]
        )
        result = clean_orders(empty_df, valid_statuses)
        assert result.empty, "Expected empty DataFrame returned (no crash)"

    def test_pseudonymize_customers_empty_salt_raises(self, sample_customers, monkeypatch):
        monkeypatch.setenv("HMAC_SALT", "")
        with pytest.raises(ValueError):
            pseudonymize_customers(sample_customers, {})
