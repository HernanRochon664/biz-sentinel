"""Script to load pipeline outputs into the SQLite database.

Usage:
    uv run python -m biz_sentinel.scripts.load_scores_to_db

Reads parquet files from data/07_model_output/ and writes to
the SQLite database configured in DATABASE_URL env var.

Idempotent: clears existing records before inserting new ones.
Run this after every training pipeline execution.
"""

import os
import sys
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy.orm import Session

from biz_sentinel.serving.api.database import (
    AlertRecord,
    AnomalyScoreRecord,
    ChurnScoreRecord,
    SegmentRecord,
    get_engine,
    get_session_factory,
    init_db,
)

ANOMALY_SCORES_PATH = os.getenv(
    "ANOMALY_SCORES_PATH", "data/07_model_output/anomaly_scores.parquet"
)
CHURN_SCORES_PATH = os.getenv("CHURN_SCORES_PATH", "data/07_model_output/churn_scores.parquet")
SEGMENT_ASSIGNMENTS_PATH = os.getenv(
    "SEGMENT_ASSIGNMENTS_PATH", "data/07_model_output/segment_assignments.parquet"
)

ANOMALY_ALERT_THRESHOLD = float(os.getenv("ANOMALY_ALERT_THRESHOLD", "0.8"))
CHURN_ALERT_THRESHOLD = float(os.getenv("CHURN_ALERT_THRESHOLD", "0.7"))


def load_anomaly_scores(db: Session, path: str) -> int:
    """Load anomaly scores from parquet into DB. Returns row count."""
    df = pd.read_parquet(path)
    db.query(AnomalyScoreRecord).delete()
    now = datetime.now(UTC)
    records = [
        AnomalyScoreRecord(
            customer_hash=row["customer_hash"],
            anomaly_score=float(row["anomaly_score"]),
            is_anomaly=bool(row["is_anomaly"]),
            anomaly_flag=str(row["anomaly_flag"]),
            scored_at=now,
        )
        for _, row in df.iterrows()
    ]
    db.bulk_save_objects(records)
    return len(records)


def load_churn_scores(db: Session, path: str) -> int:
    """Load churn scores from parquet into DB. Returns row count."""
    df = pd.read_parquet(path)
    db.query(ChurnScoreRecord).delete()
    now = datetime.now(UTC)
    records = [
        ChurnScoreRecord(
            customer_hash=row["customer_hash"],
            churn_probability=float(row["churn_probability"]),
            predicted_churn=bool(row["predicted_churn"]),
            scored_at=now,
        )
        for _, row in df.iterrows()
    ]
    db.bulk_save_objects(records)
    return len(records)


def load_segment_assignments(db: Session, path: str) -> int:
    """Load segment assignments from parquet into DB. Returns row count."""
    df = pd.read_parquet(path)
    db.query(SegmentRecord).delete()
    now = datetime.now(UTC)
    records = [
        SegmentRecord(
            customer_hash=row["customer_hash"],
            cluster_id=int(row["cluster_id"]),
            segment_label=str(row["segment_label"]),
            assigned_at=now,
        )
        for _, row in df.iterrows()
    ]
    db.bulk_save_objects(records)
    return len(records)


def generate_alerts(db: Session) -> int:
    """Generate alerts from scores above thresholds. Returns alert count."""
    db.query(AlertRecord).delete()
    alerts = []
    now = datetime.now(UTC)

    anomaly_records = (
        db.query(AnomalyScoreRecord)
        .filter(AnomalyScoreRecord.anomaly_score >= ANOMALY_ALERT_THRESHOLD)
        .all()
    )
    for r in anomaly_records:
        alerts.append(
            AlertRecord(
                customer_hash=r.customer_hash,
                alert_type="anomaly",
                score=r.anomaly_score,
                threshold=ANOMALY_ALERT_THRESHOLD,
                created_at=now,
                is_resolved=False,
            )
        )

    churn_records = (
        db.query(ChurnScoreRecord)
        .filter(ChurnScoreRecord.churn_probability >= CHURN_ALERT_THRESHOLD)
        .all()
    )
    for r in churn_records:
        alerts.append(
            AlertRecord(
                customer_hash=r.customer_hash,
                alert_type="churn_risk",
                score=r.churn_probability,
                threshold=CHURN_ALERT_THRESHOLD,
                created_at=now,
                is_resolved=False,
            )
        )

    db.bulk_save_objects(alerts)
    return len(alerts)


def main() -> None:
    """Run the full load sequence."""
    print("BizSentinel — Loading pipeline outputs to database")
    print(f"Database: {os.getenv('DATABASE_URL', 'sqlite:///data/biz_sentinel.db')}")

    engine = get_engine()
    init_db(engine)
    SessionFactory = get_session_factory(engine)
    db = SessionFactory()

    try:
        results = {}

        for label, path, loader in [
            ("anomaly_scores", ANOMALY_SCORES_PATH, load_anomaly_scores),
            ("churn_scores", CHURN_SCORES_PATH, load_churn_scores),
            ("segment_assignments", SEGMENT_ASSIGNMENTS_PATH, load_segment_assignments),
        ]:
            if not os.path.exists(path):
                print(f"  SKIP {label}: file not found at {path}")
                results[label] = 0
                continue
            count = loader(db, path)
            results[label] = count
            print(f"  OK   {label}: {count} records loaded")

        alert_count = generate_alerts(db)
        results["alerts_generated"] = alert_count
        print(f"  OK   alerts: {alert_count} alerts generated")

        db.commit()
        print("\nDone. Summary:")
        for k, v in results.items():
            print(f"  {k}: {v}")

    except Exception as e:
        db.rollback()
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
