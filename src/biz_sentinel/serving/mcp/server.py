"""BizSentinel FastMCP Server.

Exposes four tools for LLM agents:
- get_anomaly_summary: recent anomaly statistics
- get_customer_risk: full risk profile for a specific customer
- explain_alert: natural language explanation of a specific alert
- get_segment_profile: description of a customer segment

Privacy: tools return scores and explanations only.
Raw customer data is never exposed.

Usage:
    uv run python -m biz_sentinel.serving.mcp.server

    Or with MCP inspector:
    npx @modelcontextprotocol/inspector python -m biz_sentinel.serving.mcp.server
"""

import os

import pandas as pd
from fastmcp import FastMCP  # type: ignore[import-untyped]

from biz_sentinel.serving.api.database import (
    AlertRecord,
    AnomalyScoreRecord,
    ChurnScoreRecord,
    SegmentRecord,
    get_engine,
    get_session_factory,
    init_db,
)

# --- Server setup ---
mcp = FastMCP(
    name="BizSentinel",
    instructions="""You are a business intelligence assistant for BizSentinel.
You help business owners understand their customer data through anomaly detection,
segmentation, and churn prediction.

Available tools:
- get_anomaly_summary: Get statistics about recent anomalies
- get_customer_risk: Get full risk profile for a specific customer hash
- explain_alert: Get a natural language explanation of a specific alert
- get_segment_profile: Get the profile and description of a customer segment

Privacy note: Customer data is pseudonymized. You will work with customer hashes,
not real names or emails. Never ask for or expose real customer identifiers.
""",
)

# --- Database session ---
_engine = get_engine()
init_db(_engine)
_Session = get_session_factory(_engine)


def _get_session():
    return _Session()


# --- Segment descriptions (human-readable, LLM-friendly) ---
SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "champions": (
        "Champions are your best customers. They buy frequently, spend the most, "
        "and purchased recently. Reward them with loyalty programs and early access."
    ),
    "loyal": (
        "Loyal customers buy regularly and have a good spending history. "
        "They respond well to upselling and membership offers."
    ),
    "at_risk": (
        "At-risk customers used to buy regularly but haven't purchased recently. "
        "They need re-engagement campaigns with personalized offers."
    ),
    "new_customers": (
        "New customers made their first purchase recently. "
        "Focus on onboarding, building trust, and encouraging a second purchase."
    ),
    "hibernating": (
        "Hibernating customers haven't purchased in a long time and buy infrequently. "
        "A win-back campaign with a strong discount may reactivate them."
    ),
    "lost": (
        "Lost customers show no recent activity and very low purchase history. "
        "Reactivation is costly — focus resources on higher-value segments."
    ),
}


# --- Tools ---


@mcp.tool()
def get_anomaly_summary(days: int = 7) -> dict:
    """Get summary statistics of recent customer anomalies.

    Args:
        days: Number of days to consider for the summary (default: 7).

    Returns:
        Dictionary with anomaly counts, rates, and interpretation.
    """
    db = _get_session()
    try:
        records = db.query(AnomalyScoreRecord).all()
        total = len(records)

        if total == 0:
            return {
                "status": "no_data",
                "message": "No anomaly scores found. Run the training pipeline first.",
                "total_customers_scored": 0,
            }

        anomalous = [r for r in records if r.anomaly_flag == "anomalous"]
        suspicious = [r for r in records if r.anomaly_flag == "suspicious"]
        anomaly_rate = len(anomalous) / total

        # Business interpretation
        if anomaly_rate > 0.15:
            interpretation = (
                "High anomaly rate detected. This may indicate a data quality issue, "
                "a seasonal event, or unusual market activity. Review flagged customers."
            )
        elif anomaly_rate > 0.05:
            interpretation = (
                "Moderate anomaly rate. Some customers show unusual behavior patterns. "
                "Review the top anomalous customers for potential action."
            )
        else:
            interpretation = (
                "Low anomaly rate. Customer behavior appears normal. "
                "Monitor flagged customers for any emerging patterns."
            )

        return {
            "period_days": days,
            "total_customers_scored": total,
            "anomalous_count": len(anomalous),
            "suspicious_count": len(suspicious),
            "normal_count": total - len(anomalous) - len(suspicious),
            "anomaly_rate_percent": round(anomaly_rate * 100, 2),
            "interpretation": interpretation,
            "top_anomalous_customers": [
                {"customer_hash": r.customer_hash, "score": round(r.anomaly_score, 4)}
                for r in sorted(anomalous, key=lambda x: x.anomaly_score, reverse=True)[:5]
            ],
        }
    finally:
        db.close()


@mcp.tool()
def get_customer_risk(customer_hash: str) -> dict:
    """Get the full risk profile for a specific customer.

    Args:
        customer_hash: The pseudonymized customer identifier (hex string).

    Returns:
        Dictionary with anomaly score, churn probability, segment,
        and actionable recommendations.
    """
    db = _get_session()
    try:
        anomaly = (
            db.query(AnomalyScoreRecord)
            .filter(AnomalyScoreRecord.customer_hash == customer_hash)
            .first()
        )
        churn = (
            db.query(ChurnScoreRecord)
            .filter(ChurnScoreRecord.customer_hash == customer_hash)
            .first()
        )
        segment = (
            db.query(SegmentRecord).filter(SegmentRecord.customer_hash == customer_hash).first()
        )

        if not anomaly and not churn:
            return {
                "status": "not_found",
                "customer_hash": customer_hash,
                "message": "No scores found for this customer hash.",
            }

        # Build risk level
        churn_prob = churn.churn_probability if churn else None
        anomaly_score = anomaly.anomaly_score if anomaly else None
        anomaly_flag = anomaly.anomaly_flag if anomaly else None
        segment_label = segment.segment_label if segment else None

        # Determine overall risk level
        risk_level = "low"
        if churn_prob is not None and churn_prob >= 0.6:
            risk_level = "high"
        elif churn_prob is not None and churn_prob >= 0.4:
            risk_level = "medium"
        if anomaly_flag == "anomalous":
            risk_level = "high"

        # Recommendations
        recommendations = []
        if risk_level == "high":
            recommendations.append(
                "Immediate attention recommended. Consider a personalized retention offer."
            )
        if anomaly_flag in ("anomalous", "suspicious"):
            recommendations.append(
                "Unusual behavior detected. Review recent transactions for fraud or errors."
            )
        if segment_label in ("at_risk", "hibernating"):
            recommendations.append(
                "Customer shows signs of disengagement. A re-engagement campaign may help."
            )
        if not recommendations:
            recommendations.append("No immediate action required. Continue monitoring.")

        return {
            "customer_hash": customer_hash,
            "risk_level": risk_level,
            "anomaly_score": round(anomaly_score, 4) if anomaly_score else None,
            "anomaly_flag": anomaly_flag,
            "churn_probability": round(churn_prob, 4) if churn_prob else None,
            "predicted_churn": churn.predicted_churn if churn else None,
            "segment": segment_label,
            "segment_description": SEGMENT_DESCRIPTIONS.get(segment_label or "", ""),
            "recommendations": recommendations,
        }
    finally:
        db.close()


@mcp.tool()
def explain_alert(alert_id: int) -> dict:
    """Get a natural language explanation of a specific alert.

    Args:
        alert_id: The integer ID of the alert to explain.

    Returns:
        Dictionary with alert details and a plain-language explanation.
    """
    db = _get_session()
    try:
        alert = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()

        if not alert:
            return {
                "status": "not_found",
                "alert_id": alert_id,
                "message": f"No alert found with ID {alert_id}.",
            }

        # Build explanation based on alert type
        if alert.alert_type == "anomaly":
            explanation = (
                f"Customer {alert.customer_hash[:8]}... was flagged as anomalous "
                f"with a score of {alert.score:.2f} (threshold: {alert.threshold:.2f}). "
                f"This means their behavior is significantly different from the typical "
                f"customer pattern. Possible causes: unusual purchase volume, abnormal "
                f"payment patterns, or sudden change in buying frequency."
            )
        elif alert.alert_type == "churn_risk":
            explanation = (
                f"Customer {alert.customer_hash[:8]}... has a churn probability of "
                f"{alert.score:.1%} (threshold: {alert.threshold:.1%}). "
                f"The model predicts this customer is likely to stop purchasing. "
                f"Key drivers may include: long time since last purchase, declining "
                f"order frequency, or low satisfaction scores."
            )
        else:
            explanation = (
                f"Alert type '{alert.alert_type}' triggered for customer "
                f"{alert.customer_hash[:8]}... with score {alert.score:.4f}."
            )

        return {
            "alert_id": alert_id,
            "customer_hash": alert.customer_hash,
            "alert_type": alert.alert_type,
            "score": round(alert.score, 4),
            "threshold": alert.threshold,
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "is_resolved": alert.is_resolved,
            "explanation": explanation,
            "recommended_action": (
                "Mark as resolved once reviewed."
                if not alert.is_resolved
                else "This alert has already been resolved."
            ),
        }
    finally:
        db.close()


@mcp.tool()
def get_segment_profile(segment_label: str) -> dict:
    """Get the profile and business description of a customer segment.

    Args:
        segment_label: One of: champions, loyal, at_risk,
                       new_customers, hibernating, lost.

    Returns:
        Dictionary with segment statistics and actionable description.
    """
    valid_labels = list(SEGMENT_DESCRIPTIONS.keys())
    if segment_label not in valid_labels:
        return {
            "status": "invalid_segment",
            "message": f"Invalid segment '{segment_label}'. "
            f"Valid options: {', '.join(valid_labels)}",
        }

    db = _get_session()
    try:
        records = db.query(SegmentRecord).filter(SegmentRecord.segment_label == segment_label).all()

        customer_count = len(records)

        # Load segment profiles from parquet if available
        profiles_path = os.getenv(
            "SEGMENT_PROFILES_PATH", "data/07_model_output/segment_profiles.parquet"
        )
        profile_stats: dict = {}
        try:
            profiles_df = pd.read_parquet(profiles_path)
            row = profiles_df[profiles_df["segment_label"] == segment_label]
            if not row.empty:
                profile_stats = row.iloc[0].to_dict()
                # Round float values for readability
                profile_stats = {
                    k: round(v, 2) if isinstance(v, float) else v for k, v in profile_stats.items()
                }
        except FileNotFoundError:
            profile_stats = {"note": "Detailed profile not available. Run training pipeline."}

        return {
            "segment_label": segment_label,
            "customer_count": customer_count,
            "description": SEGMENT_DESCRIPTIONS[segment_label],
            "profile_stats": profile_stats,
            "marketing_suggestion": _get_marketing_suggestion(segment_label),
        }
    finally:
        db.close()


def _get_marketing_suggestion(segment_label: str) -> str:
    """Return a concrete marketing action for each segment."""
    suggestions = {
        "champions": "Send a VIP early-access email for new products. Offer a referral bonus.",
        "loyal": "Introduce a points-based loyalty program. Offer bundle discounts.",
        "at_risk": "Send a 'We miss you' email with a 15% discount code. Limited time.",
        "new_customers": "Send a welcome series: onboarding tips + first repeat-purchase discount.",
        "hibernating": "Win-back campaign: bold subject line + 25% discount. One email only.",
        "lost": "Low priority. Include in quarterly bulk reactivation if budget allows.",
    }
    return suggestions.get(segment_label, "No specific suggestion available.")


if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
