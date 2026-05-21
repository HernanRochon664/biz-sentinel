"""SQLAlchemy database connection for BizSentinel API."""

import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class AnomalyScoreRecord(Base):
    """ORM model for anomaly_scores table."""

    __tablename__ = "anomaly_scores"

    customer_hash = Column(String, primary_key=True)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    anomaly_flag = Column(String, nullable=False)
    scored_at = Column(DateTime, default=datetime.utcnow)


class ChurnScoreRecord(Base):
    """ORM model for churn_scores table."""

    __tablename__ = "churn_scores"

    customer_hash = Column(String, primary_key=True)
    churn_probability = Column(Float, nullable=False)
    predicted_churn = Column(Boolean, nullable=False)
    scored_at = Column(DateTime, default=datetime.utcnow)


class SegmentRecord(Base):
    """ORM model for segment_assignments table."""

    __tablename__ = "segment_assignments"

    customer_hash = Column(String, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    segment_label = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)


class AlertRecord(Base):
    """ORM model for alerts table."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_hash = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)  # "anomaly" | "churn_risk"
    score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)


def get_engine(database_url: str | None = None) -> Engine:
    """Create SQLAlchemy engine from DATABASE_URL env var or argument."""
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///data/biz_sentinel.db")
    return create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine: Engine) -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


# Module-level engine and session (initialized on import)
_engine = get_engine()
_SessionLocal = get_session_factory(_engine)


def get_db():
    """FastAPI dependency: yields a database session."""
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
