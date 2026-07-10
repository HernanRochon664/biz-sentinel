"""Integration tests for the API against data loaded through the real pipeline.

Unlike unit tests that insert data manually, this test exercises the full
parquet → database → API flow. It reuses the same load functions that
the production `load_scores_to_db` script uses.
"""

import os
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from biz_sentinel.scripts.load_scores_to_db import (
    generate_alerts,
    load_anomaly_scores,
    load_churn_scores,
    load_segment_assignments,
)
from biz_sentinel.serving.api.database import (
    init_db,
)
from biz_sentinel.serving.api.main import ALGORITHM, SECRET_KEY, app, get_db


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def loaded_db(db_session):
    anomaly_df = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:04d}" for i in range(1, 6)],
            "anomaly_score": [0.95, 0.82, 0.45, 0.12, 0.03],
            "is_anomaly": [True, True, False, False, False],
            "anomaly_flag": ["anomalous", "anomalous", "suspicious", "normal", "normal"],
        }
    )
    churn_df = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:04d}" for i in range(1, 6)],
            "churn_probability": [0.91, 0.75, 0.50, 0.30, 0.05],
            "predicted_churn": [True, True, False, False, False],
        }
    )
    segment_df = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:04d}" for i in range(1, 6)],
            "cluster_id": [0, 1, 2, 0, 1],
            "segment_label": ["champions", "at_risk", "loyal", "champions", "hibernating"],
        }
    )

    with (
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as anomaly_f,
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as churn_f,
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as segment_f,
    ):
        anomaly_path = anomaly_f.name
        churn_path = churn_f.name
        segment_path = segment_f.name
        anomaly_df.to_parquet(anomaly_path, index=False)
        churn_df.to_parquet(churn_path, index=False)
        segment_df.to_parquet(segment_path, index=False)

    try:
        load_anomaly_scores(db_session, anomaly_path)
        load_churn_scores(db_session, churn_path)
        load_segment_assignments(db_session, segment_path)
        generate_alerts(db_session)
        db_session.commit()
    finally:
        os.unlink(anomaly_path)
        os.unlink(churn_path)
        os.unlink(segment_path)

    return db_session


@pytest.fixture
def test_app(loaded_db):
    def override_get_db():
        yield loaded_db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    token = jwt.encode(
        {"sub": "integration_test_user"},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
class TestIntegrationAnomalySummary:
    def test_anomaly_counts_via_loaded_data(self, test_app, auth_headers):
        data = test_app.get("/anomalies/recent", headers=auth_headers).json()
        assert data["total_customers_scored"] == 5
        assert data["anomaly_count"] == 2
        assert data["suspicious_count"] == 1
        assert data["normal_count"] == 2

    def test_anomaly_rate_calculation(self, test_app, auth_headers):
        data = test_app.get("/anomalies/recent", headers=auth_headers).json()
        expected_rate = 2 / 5
        assert data["anomaly_rate"] == pytest.approx(expected_rate)


@pytest.mark.integration
class TestIntegrationCustomerRisk:
    def test_high_risk_customer_has_all_scores(self, test_app, auth_headers):
        data = test_app.get("/customers/hash_0001/risk", headers=auth_headers).json()
        assert data["customer_hash"] == "hash_0001"
        assert data["anomaly_score"] == pytest.approx(0.95)
        assert data["anomaly_flag"] == "anomalous"
        assert data["churn_probability"] == pytest.approx(0.91)
        assert data["predicted_churn"] is True

    def test_low_risk_customer_has_no_anomaly(self, test_app, auth_headers):
        data = test_app.get("/customers/hash_0005/risk", headers=auth_headers).json()
        assert data["anomaly_flag"] == "normal"
        assert data["predicted_churn"] is False

    def test_customer_not_found(self, test_app, auth_headers):
        response = test_app.get("/customers/nonexistent/risk", headers=auth_headers)
        assert response.status_code == 404


@pytest.mark.integration
class TestIntegrationAlerts:
    def test_alerts_generated_from_loaded_data(self, test_app, auth_headers):
        data = test_app.get("/alerts", headers=auth_headers).json()
        assert len(data) >= 2
        alert_types = {a["alert_type"] for a in data}
        assert "anomaly" in alert_types
        assert "churn_risk" in alert_types

    def test_alerts_default_unresolved(self, test_app, auth_headers):
        data = test_app.get("/alerts", headers=auth_headers).json()
        for alert in data:
            assert alert["is_resolved"] is False

    def test_alerts_have_required_schema(self, test_app, auth_headers):
        data = test_app.get("/alerts", headers=auth_headers).json()
        alert = data[0]
        assert "id" in alert
        assert "customer_hash" in alert
        assert "alert_type" in alert
        assert "score" in alert
        assert "threshold" in alert
        assert "created_at" in alert
