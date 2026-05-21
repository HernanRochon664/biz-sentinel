"""Tests for FastAPI serving endpoints."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from biz_sentinel.serving.api.main import app, get_db, SECRET_KEY, ALGORITHM
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from biz_sentinel.serving.api.database import (
    AnomalyScoreRecord,
    AlertRecord,
    ChurnScoreRecord,
    SegmentRecord,
    init_db,
)


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
def test_app(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    token = jwt.encode(
        {"sub": "test_user"}, SECRET_KEY, algorithm=ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def db_with_data(db_session):
    now = datetime.utcnow()

    records = [
        AnomalyScoreRecord(
            customer_hash="hash_001", anomaly_score=0.95,
            is_anomaly=True, anomaly_flag="anomalous", scored_at=now,
        ),
        AnomalyScoreRecord(
            customer_hash="hash_002", anomaly_score=0.75,
            is_anomaly=False, anomaly_flag="suspicious", scored_at=now,
        ),
        AnomalyScoreRecord(
            customer_hash="hash_003", anomaly_score=0.20,
            is_anomaly=False, anomaly_flag="normal", scored_at=now,
        ),
        ChurnScoreRecord(
            customer_hash="hash_001", churn_probability=0.85,
            predicted_churn=True, scored_at=now,
        ),
        ChurnScoreRecord(
            customer_hash="hash_002", churn_probability=0.40,
            predicted_churn=False, scored_at=now,
        ),
        ChurnScoreRecord(
            customer_hash="hash_003", churn_probability=0.10,
            predicted_churn=False, scored_at=now,
        ),
        SegmentRecord(
            customer_hash="hash_001", cluster_id=0,
            segment_label="high_value", assigned_at=now,
        ),
        SegmentRecord(
            customer_hash="hash_002", cluster_id=1,
            segment_label="at_risk", assigned_at=now,
        ),
        SegmentRecord(
            customer_hash="hash_003", cluster_id=2,
            segment_label="new", assigned_at=now,
        ),
        AlertRecord(
            customer_hash="hash_001", alert_type="anomaly",
            score=0.95, threshold=0.8, created_at=now, is_resolved=True,
        ),
        AlertRecord(
            customer_hash="hash_002", alert_type="churn_risk",
            score=0.75, threshold=0.7, created_at=now, is_resolved=False,
        ),
    ]
    for record in records:
        db_session.add(record)
    db_session.commit()


class TestHealthEndpoint:
    def test_health_returns_200(self, test_app):
        response = test_app.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, test_app):
        data = test_app.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data

    def test_health_status_is_ok(self, test_app):
        assert test_app.get("/health").json()["status"] == "ok"


class TestAuthentication:
    def test_protected_endpoint_requires_auth(self, test_app):
        response = test_app.get("/anomalies/recent")
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, test_app):
        response = test_app.get(
            "/anomalies/recent",
            headers={"Authorization": "Bearer invalidtoken"},
        )
        assert response.status_code == 401

    def test_valid_token_passes(self, test_app, auth_headers):
        response = test_app.get("/anomalies/recent", headers=auth_headers)
        assert response.status_code == 200


class TestAnomalySummaryEndpoint:
    def test_anomaly_summary_returns_correct_schema(
        self, test_app, auth_headers, db_with_data,
    ):
        data = test_app.get("/anomalies/recent", headers=auth_headers).json()
        assert "total_customers_scored" in data
        assert "anomaly_count" in data
        assert "suspicious_count" in data
        assert "normal_count" in data
        assert "anomaly_rate" in data
        assert "period_days" in data

    def test_anomaly_summary_counts_correct(self, test_app, auth_headers, db_with_data):
        data = test_app.get("/anomalies/recent", headers=auth_headers).json()
        assert data["total_customers_scored"] == 3
        assert data["anomaly_count"] == 1
        assert data["suspicious_count"] == 1
        assert data["normal_count"] == 1

    def test_anomaly_rate_is_valid_float(self, test_app, auth_headers, db_with_data):
        rate = test_app.get("/anomalies/recent", headers=auth_headers).json()["anomaly_rate"]
        assert 0.0 <= rate <= 1.0


class TestCustomerRiskEndpoint:
    def test_customer_risk_returns_data(self, test_app, auth_headers, db_with_data):
        response = test_app.get("/customers/hash_001/risk", headers=auth_headers)
        assert response.status_code == 200

    def test_customer_risk_404_for_unknown(self, test_app, auth_headers):
        response = test_app.get("/customers/unknown_hash/risk", headers=auth_headers)
        assert response.status_code == 404

    def test_customer_risk_schema(self, test_app, auth_headers, db_with_data):
        data = test_app.get(
            "/customers/hash_001/risk", headers=auth_headers,
        ).json()
        assert "customer_hash" in data
        assert data["customer_hash"] == "hash_001"


class TestAlertsEndpoint:
    def test_alerts_returns_list(self, test_app, auth_headers, db_with_data):
        data = test_app.get("/alerts", headers=auth_headers).json()
        assert isinstance(data, list)

    def test_alerts_resolved_filter(self, test_app, auth_headers, db_with_data):
        data = test_app.get(
            "/alerts", params={"resolved": "true"}, headers=auth_headers,
        ).json()
        assert all(alert["is_resolved"] for alert in data)

    def test_alerts_limit_respected(self, test_app, auth_headers, db_with_data):
        data = test_app.get(
            "/alerts", params={"limit": 1}, headers=auth_headers,
        ).json()
        assert len(data) <= 1
