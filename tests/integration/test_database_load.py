"""Integration tests for the database loading pipeline.

Tests the parquet → SQLite data flow including alert generation.
Uses temporary parquet files and an in-memory SQLite database.
"""

import os
import tempfile

import pandas as pd
import pytest

from biz_sentinel.scripts.load_scores_to_db import (
    generate_alerts,
    load_anomaly_scores,
    load_churn_scores,
    load_segment_assignments,
)
from biz_sentinel.serving.api.database import (
    AlertRecord,
    AnomalyScoreRecord,
    ChurnScoreRecord,
    SegmentRecord,
    get_engine,
    get_session_factory,
    init_db,
)


@pytest.fixture
def db():
    engine = get_engine("sqlite://")
    init_db(engine)
    SessionFactory = get_session_factory(engine)
    session = SessionFactory()
    yield session
    session.close()


@pytest.fixture
def anomaly_parquet():
    df = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:04d}" for i in range(1, 6)],
            "anomaly_score": [0.95, 0.82, 0.45, 0.12, 0.03],
            "is_anomaly": [True, True, False, False, False],
            "anomaly_flag": ["anomalous", "anomalous", "suspicious", "normal", "normal"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
        df.to_parquet(path, index=False)
    yield path
    os.unlink(path)


@pytest.fixture
def churn_parquet():
    df = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:04d}" for i in range(1, 6)],
            "churn_probability": [0.91, 0.75, 0.50, 0.30, 0.05],
            "predicted_churn": [True, True, False, False, False],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
        df.to_parquet(path, index=False)
    yield path
    os.unlink(path)


@pytest.fixture
def segment_parquet():
    df = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:04d}" for i in range(1, 6)],
            "cluster_id": [0, 1, 2, 0, 1],
            "segment_label": ["champions", "at_risk", "loyal", "champions", "hibernating"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
        df.to_parquet(path, index=False)
    yield path
    os.unlink(path)


@pytest.mark.integration
class TestLoadAnomalyScores:
    def test_loads_correct_count(self, db, anomaly_parquet):
        count = load_anomaly_scores(db, anomaly_parquet)
        assert count == 5

    def test_data_integrity(self, db, anomaly_parquet):
        load_anomaly_scores(db, anomaly_parquet)
        db.commit()

        score = db.query(AnomalyScoreRecord).filter_by(customer_hash="hash_0001").first()
        assert score is not None
        assert score.anomaly_score == 0.95
        assert score.is_anomaly is True
        assert score.anomaly_flag == "anomalous"
        assert score.scored_at is not None

    def test_all_flags_present(self, db, anomaly_parquet):
        load_anomaly_scores(db, anomaly_parquet)
        flags = {r.anomaly_flag for r in db.query(AnomalyScoreRecord).all()}
        assert "anomalous" in flags
        assert "suspicious" in flags
        assert "normal" in flags

    def test_idempotent_reload(self, db, anomaly_parquet):
        load_anomaly_scores(db, anomaly_parquet)
        db.commit()
        first_count = db.query(AnomalyScoreRecord).count()

        load_anomaly_scores(db, anomaly_parquet)
        db.commit()
        second_count = db.query(AnomalyScoreRecord).count()

        assert first_count == second_count == 5


@pytest.mark.integration
class TestLoadChurnScores:
    def test_loads_correct_count(self, db, churn_parquet):
        count = load_churn_scores(db, churn_parquet)
        assert count == 5

    def test_churn_probability_precision(self, db, churn_parquet):
        load_churn_scores(db, churn_parquet)
        db.commit()

        high_risk = db.query(ChurnScoreRecord).filter_by(customer_hash="hash_0001").first()
        assert high_risk.churn_probability == pytest.approx(0.91)

        low_risk = db.query(ChurnScoreRecord).filter_by(customer_hash="hash_0005").first()
        assert low_risk.churn_probability == pytest.approx(0.05)

    def test_predicted_churn_logic(self, db, churn_parquet):
        load_churn_scores(db, churn_parquet)
        churned = (
            db.query(ChurnScoreRecord).filter(ChurnScoreRecord.predicted_churn.is_(True)).all()
        )
        assert len(churned) == 2


@pytest.mark.integration
class TestLoadSegmentAssignments:
    def test_loads_correct_count(self, db, segment_parquet):
        count = load_segment_assignments(db, segment_parquet)
        assert count == 5

    def test_segment_labels_present(self, db, segment_parquet):
        load_segment_assignments(db, segment_parquet)
        labels = {r.segment_label for r in db.query(SegmentRecord).all()}
        assert "champions" in labels
        assert "at_risk" in labels
        assert "loyal" in labels
        assert "hibernating" in labels

    def test_cluster_id_integrity(self, db, segment_parquet):
        load_segment_assignments(db, segment_parquet)
        distinct_clusters = db.query(SegmentRecord.cluster_id).distinct().all()
        assert len(distinct_clusters) == 3


@pytest.mark.integration
class TestGenerateAlerts:
    ANOMALY_ALERT_THRESHOLD = 0.8
    CHURN_ALERT_THRESHOLD = 0.7

    @pytest.fixture(autouse=True)
    def _set_thresholds(self, monkeypatch):
        monkeypatch.setattr(
            "biz_sentinel.scripts.load_scores_to_db.ANOMALY_ALERT_THRESHOLD",
            self.ANOMALY_ALERT_THRESHOLD,
        )
        monkeypatch.setattr(
            "biz_sentinel.scripts.load_scores_to_db.CHURN_ALERT_THRESHOLD",
            self.CHURN_ALERT_THRESHOLD,
        )

    def _load_all(self, db, anomaly_parquet, churn_parquet, segment_parquet):
        load_anomaly_scores(db, anomaly_parquet)
        load_churn_scores(db, churn_parquet)
        load_segment_assignments(db, segment_parquet)
        db.commit()

    def test_generates_alerts_for_high_scores(
        self, db, anomaly_parquet, churn_parquet, segment_parquet
    ):
        self._load_all(db, anomaly_parquet, churn_parquet, segment_parquet)
        alert_count = generate_alerts(db)
        db.commit()

        assert alert_count > 0
        anomaly_alerts = db.query(AlertRecord).filter(AlertRecord.alert_type == "anomaly").count()
        churn_alerts = db.query(AlertRecord).filter(AlertRecord.alert_type == "churn_risk").count()
        assert anomaly_alerts == 2  # hash_0001 (0.95), hash_0002 (0.82) >= 0.8
        assert churn_alerts == 2  # hash_0001 (0.91), hash_0002 (0.75) >= 0.7

    def test_alerts_have_required_fields(self, db, anomaly_parquet, churn_parquet, segment_parquet):
        self._load_all(db, anomaly_parquet, churn_parquet, segment_parquet)
        generate_alerts(db)

        alert = db.query(AlertRecord).first()
        assert alert.customer_hash is not None
        assert alert.alert_type in ("anomaly", "churn_risk")
        assert isinstance(alert.score, float)
        assert isinstance(alert.threshold, float)
        assert alert.created_at is not None
        assert alert.is_resolved is False

    def test_alerts_idempotent_reload(self, db, anomaly_parquet, churn_parquet, segment_parquet):
        self._load_all(db, anomaly_parquet, churn_parquet, segment_parquet)
        generate_alerts(db)
        first_count = db.query(AlertRecord).count()

        generate_alerts(db)
        second_count = db.query(AlertRecord).count()

        assert first_count == second_count

    def test_no_false_alerts_below_threshold(
        self, db, anomaly_parquet, churn_parquet, segment_parquet
    ):
        self._load_all(db, anomaly_parquet, churn_parquet, segment_parquet)

        MIN_ANOMALY = 0.8
        generate_alerts(db)

        for alert in db.query(AlertRecord).filter_by(alert_type="anomaly").all():
            assert alert.score >= MIN_ANOMALY

    def test_score_distribution_anchor(self, db, anomaly_parquet, churn_parquet, segment_parquet):
        self._load_all(db, anomaly_parquet, churn_parquet, segment_parquet)
        scores = [r.anomaly_score for r in db.query(AnomalyScoreRecord).all()]
        assert min(scores) == pytest.approx(0.03)
        assert max(scores) == pytest.approx(0.95)
