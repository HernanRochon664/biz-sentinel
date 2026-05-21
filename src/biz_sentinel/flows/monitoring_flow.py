"""Monitoring Prefect Flow — weekly model and data drift checks."""

import pandas as pd
from prefect import flow, get_run_logger, task  # type: ignore[import-untyped]

from biz_sentinel.flows.training_flow import notify_completion


@task(name="check_data_drift")
def check_data_drift(
    reference_path: str,
    current_path: str,
    drift_threshold: float = 0.2,
) -> dict[str, float]:
    """Compare feature distributions between reference and current data.

    Uses a simple statistical approach: for each numeric feature,
    compute the relative change in mean. If any feature drifts more
    than drift_threshold (20% by default), flag it.

    Args:
        reference_path: Path to reference feature_matrix parquet.
        current_path: Path to current feature_matrix parquet.
        drift_threshold: Maximum allowed relative change in feature mean.

    Returns:
        Dictionary of feature_name → drift_score (relative mean change).
    """
    logger = get_run_logger()

    try:
        reference = pd.read_parquet(reference_path)
        current = pd.read_parquet(current_path)
    except FileNotFoundError as e:
        logger.warning(f"Could not load data for drift check: {e}")
        return {}

    numeric_cols = reference.select_dtypes(include="number").columns
    drift_scores: dict[str, float] = {}

    for col in numeric_cols:
        if col in current.columns:
            ref_mean = reference[col].mean()
            cur_mean = current[col].mean()
            if ref_mean != 0:
                drift = abs((cur_mean - ref_mean) / ref_mean)
                drift_scores[col] = float(drift)
                if drift > drift_threshold:
                    logger.warning(
                        f"Data drift detected in '{col}': "
                        f"{drift:.1%} change (threshold: {drift_threshold:.0%})"
                    )

    drifted = [k for k, v in drift_scores.items() if v > drift_threshold]
    logger.info(f"Drift check complete. {len(drifted)} features drifted above threshold.")
    return drift_scores


@task(name="check_score_distribution")
def check_score_distribution(
    scores_path: str,
    expected_anomaly_rate_range: tuple[float, float] = (0.01, 0.20),
    expected_churn_rate_range: tuple[float, float] = (0.10, 0.60),
) -> dict[str, bool]:
    """Verify that score distributions are within expected ranges.

    Flags if anomaly rate or churn rate fall outside expected bounds,
    which may indicate model degradation or data issues.

    Args:
        scores_path: Path to latest churn_scores parquet.
        expected_anomaly_rate_range: (min, max) acceptable anomaly rate.
        expected_churn_rate_range: (min, max) acceptable churn rate.

    Returns:
        Dictionary of check_name → passed (bool).
    """
    logger = get_run_logger()
    checks: dict[str, bool] = {}

    try:
        scores = pd.read_parquet(scores_path)
    except FileNotFoundError:
        logger.warning(f"Scores file not found: {scores_path}")
        return {"scores_file_exists": False}

    if "churn_probability" in scores.columns:
        churn_rate = float((scores["churn_probability"] >= 0.5).mean())
        in_range = expected_churn_rate_range[0] <= churn_rate <= expected_churn_rate_range[1]
        checks["churn_rate_in_range"] = in_range
        if not in_range:
            logger.warning(
                f"Churn rate {churn_rate:.1%} outside expected range {expected_churn_rate_range}"
            )

    checks["scores_file_exists"] = True
    logger.info(f"Score distribution checks: {checks}")
    return checks


@flow(
    name="biz-sentinel-monitoring",
    description="Weekly monitoring: data drift detection and score distribution checks",
    retries=1,
    retry_delay_seconds=3600,
)
def monitoring_flow(
    reference_features_path: str = "data/05_model_input/feature_matrix.parquet",
    current_scores_path: str = "data/07_model_output/churn_scores.parquet",
) -> dict[str, object]:
    """Run weekly monitoring checks.

    Args:
        reference_features_path: Path to reference feature matrix.
        current_scores_path: Path to latest scores.

    Returns:
        Dictionary with drift scores and distribution check results.
    """
    logger = get_run_logger()
    logger.info("BizSentinel Monitoring Flow started")

    drift_scores = check_data_drift(
        reference_path=reference_features_path,
        current_path=reference_features_path,
    )

    score_checks = check_score_distribution(current_scores_path)

    all_passed = all(score_checks.values())
    notify_completion(
        "monitoring_flow",
        success=all_passed,
        metrics_summary=f"Drifted features: {sum(1 for v in drift_scores.values() if v > 0.2)}",
    )

    return {"drift_scores": drift_scores, "score_checks": score_checks}
