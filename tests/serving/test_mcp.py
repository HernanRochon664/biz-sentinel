"""Tests for FastMCP tools.

FastMCP tools are plain Python functions decorated with @mcp.tool().
These tests call them directly with no MCP server needed.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from biz_sentinel.domain.models import SegmentLabel
from biz_sentinel.serving.api.database import (
    AlertRecord,
    AnomalyScoreRecord,
    ChurnScoreRecord,
    SegmentRecord,
    init_db,
)
from biz_sentinel.serving.mcp.server import (
    _get_marketing_suggestion,
    explain_alert,
    get_anomaly_summary,
    get_customer_risk,
    get_segment_profile,
)


@pytest.fixture
def mcp_db(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestSession()
    now = datetime.utcnow()

    records = [
        AnomalyScoreRecord(
            customer_hash="hash_anom", anomaly_score=0.95,
            is_anomaly=True, anomaly_flag="anomalous", scored_at=now,
        ),
        AnomalyScoreRecord(
            customer_hash="hash_susp", anomaly_score=0.65,
            is_anomaly=False, anomaly_flag="suspicious", scored_at=now,
        ),
        AnomalyScoreRecord(
            customer_hash="hash_norm", anomaly_score=0.15,
            is_anomaly=False, anomaly_flag="normal", scored_at=now,
        ),
        ChurnScoreRecord(
            customer_hash="hash_anom", churn_probability=0.85,
            predicted_churn=True, scored_at=now,
        ),
        ChurnScoreRecord(
            customer_hash="hash_norm", churn_probability=0.15,
            predicted_churn=False, scored_at=now,
        ),
        SegmentRecord(
            customer_hash="hash_anom", cluster_id=0,
            segment_label="at_risk", assigned_at=now,
        ),
        SegmentRecord(
            customer_hash="hash_champ", cluster_id=1,
            segment_label="champions", assigned_at=now,
        ),
        AlertRecord(
            id=1, customer_hash="hash_anom", alert_type="anomaly",
            score=0.95, threshold=0.8, created_at=now, is_resolved=False,
        ),
        AlertRecord(
            id=2, customer_hash="hash_susp", alert_type="churn_risk",
            score=0.75, threshold=0.7, created_at=now, is_resolved=True,
        ),
    ]
    for r in records:
        db.add(r)
    db.commit()
    db.close()

    monkeypatch.setattr(
        "biz_sentinel.serving.mcp.server._get_session",
        lambda: TestSession(),
    )


class TestGetAnomalySummary:
    def test_summary_returns_expected_keys(self, mcp_db):
        result = get_anomaly_summary()
        expected_keys = [
            "total_customers_scored",
            "anomalous_count",
            "suspicious_count",
            "normal_count",
            "anomaly_rate_percent",
            "interpretation",
        ]
        assert all(k in result for k in expected_keys)

    def test_summary_counts_match_inserted_data(self, mcp_db):
        result = get_anomaly_summary()
        assert result["anomalous_count"] == 1
        assert result["suspicious_count"] == 1
        assert result["normal_count"] == 1

    def test_summary_no_data_returns_status_no_data(self, monkeypatch):
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        init_db(engine)
        EmptySession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        monkeypatch.setattr(
            "biz_sentinel.serving.mcp.server._get_session",
            lambda: EmptySession(),
        )
        result = get_anomaly_summary()
        assert result["status"] == "no_data"

    def test_summary_interpretation_is_string(self, mcp_db):
        result = get_anomaly_summary()
        assert isinstance(result["interpretation"], str)

    def test_summary_top_anomalous_is_list(self, mcp_db):
        result = get_anomaly_summary()
        assert isinstance(result["top_anomalous_customers"], list)


class TestGetCustomerRisk:
    def test_customer_risk_found(self, mcp_db):
        result = get_customer_risk("hash_anom")
        assert "risk_level" in result

    def test_customer_risk_not_found(self, mcp_db):
        result = get_customer_risk("hash_unknown")
        assert result["status"] == "not_found"

    def test_customer_risk_recommendations_is_list(self, mcp_db):
        result = get_customer_risk("hash_anom")
        assert isinstance(result["recommendations"], list)

    def test_customer_risk_high_risk_detection(self, mcp_db):
        from biz_sentinel.serving.mcp.server import _get_session

        db = _get_session()
        now = datetime.utcnow()
        db.add(AnomalyScoreRecord(
            customer_hash="hash_high", anomaly_score=0.99,
            is_anomaly=True, anomaly_flag="anomalous", scored_at=now,
        ))
        db.add(ChurnScoreRecord(
            customer_hash="hash_high", churn_probability=0.9,
            predicted_churn=True, scored_at=now,
        ))
        db.commit()
        db.close()

        result = get_customer_risk("hash_high")
        assert result["risk_level"] == "high"

    def test_customer_risk_segment_description_populated(self, mcp_db):
        result = get_customer_risk("hash_anom")
        assert isinstance(result["segment_description"], str)
        assert len(result["segment_description"]) > 0


class TestExplainAlert:
    def test_explain_alert_found(self, mcp_db):
        result = explain_alert(1)
        assert "explanation" in result

    def test_explain_alert_not_found(self, mcp_db):
        result = explain_alert(9999)
        assert result["status"] == "not_found"

    def test_explain_alert_anomaly_type(self, mcp_db):
        result = explain_alert(1)
        text = result["explanation"].lower()
        assert "anomalous" in text or "unusual" in text

    def test_explain_alert_has_recommended_action(self, mcp_db):
        result = explain_alert(1)
        assert "recommended_action" in result


class TestGetSegmentProfile:
    def test_segment_profile_valid_label(self, mcp_db):
        result = get_segment_profile("champions")
        assert "description" in result

    def test_segment_profile_invalid_label(self):
        result = get_segment_profile("invalid_label")
        assert result["status"] == "invalid_segment"

    def test_segment_profile_all_valid_labels(self, mcp_db):
        for label in SegmentLabel:
            result = get_segment_profile(label.value)
            assert result.get("status") != "invalid_segment"

    def test_segment_profile_description_is_string(self, mcp_db):
        result = get_segment_profile("champions")
        assert isinstance(result["description"], str)

    def test_segment_profile_customer_count(self, mcp_db):
        result = get_segment_profile("champions")
        assert result["customer_count"] == 1


class TestGetMarketingSuggestion:
    def test_marketing_suggestion_all_segments(self):
        for label in SegmentLabel:
            suggestion = _get_marketing_suggestion(label.value)
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0

    def test_marketing_suggestion_unknown_returns_default(self):
        result = _get_marketing_suggestion("unknown")
        assert isinstance(result, str)
        assert len(result) > 0
