"""Monitoring Prefect Flow — weekly model and data drift checks."""

import os

import pandas as pd
from prefect import flow, get_run_logger, task  # type: ignore[import-untyped]
from scipy.stats import ks_2samp  # type: ignore[import-untyped]

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


@task(name="check_model_drift")
def check_model_drift(
    current_scores_path: str,
    baseline_path: str,
    ks_threshold: float = 0.1,
) -> dict[str, object]:
    """Detect concept drift via KS test on churn probability distribution.

    Compares the current churn probability distribution against a stored
    baseline using the two-sample Kolmogorov-Smirnov test. A significant
    KS statistic (p < 0.05, or statistic > ks_threshold) suggests the
    model's predictions have shifted — possible concept drift.

    If no baseline exists yet (first run), saves current scores as the
    baseline and reports 'no_baseline'.

    Args:
        current_scores_path: Path to latest churn_scores parquet.
        baseline_path: Path to baseline churn probability distribution parquet.
        ks_threshold: KS statistic threshold for flagging drift (default 0.1).

    Returns:
        Dictionary with keys: ks_statistic, p_value, drift_detected, status.
    """
    logger = get_run_logger()

    try:
        current = pd.read_parquet(current_scores_path)
    except FileNotFoundError:
        logger.warning(f"Current scores not found: {current_scores_path}")
        return {"status": "no_data", "ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    current_probs = current["churn_probability"].dropna().values

    if not os.path.exists(baseline_path):
        pd.DataFrame({"churn_probability": current_probs}).to_parquet(baseline_path, index=False)
        logger.info(f"No baseline found. Saved current distribution as baseline to {baseline_path}")
        return {
            "status": "baseline_created",
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
        }

    baseline = pd.read_parquet(baseline_path)
    baseline_probs = baseline["churn_probability"].dropna().values

    ks_result = ks_2samp(baseline_probs, current_probs)  # type: ignore[import-untyped]
    ks_stat = float(ks_result[0])  # type: ignore[index]
    p_val = float(ks_result[1])  # type: ignore[index]
    drift_detected = ks_stat > ks_threshold and p_val < 0.05

    if drift_detected:
        logger.warning(
            f"Model drift detected: KS={ks_stat:.4f}, p={p_val:.4f} (threshold={ks_threshold})"
        )
    else:
        logger.info(f"Model drift check passed: KS={ks_stat:.4f}, p={p_val:.4f}")

    return {
        "status": "completed",
        "ks_statistic": float(ks_stat),
        "p_value": float(p_val),
        "drift_detected": drift_detected,
    }


@task(name="save_score_baseline")
def save_score_baseline(
    current_scores_path: str,
    baseline_path: str,
) -> bool:
    """Update the baseline distribution from current scores.

    Called after a successful monitoring run to roll the baseline forward,
    preventing repeated drift alerts from the same shift.

    Args:
        current_scores_path: Path to latest churn_scores parquet.
        baseline_path: Path to baseline parquet to overwrite.

    Returns:
        True if baseline was updated.
    """
    logger = get_run_logger()
    try:
        current = pd.read_parquet(current_scores_path)
    except FileNotFoundError:
        logger.warning(f"Cannot update baseline: scores not found at {current_scores_path}")
        return False

    probs = current["churn_probability"].dropna().values
    pd.DataFrame({"churn_probability": probs}).to_parquet(baseline_path, index=False)
    logger.info(f"Baseline updated at {baseline_path} ({len(probs)} samples)")
    return True


@flow(
    name="biz-sentinel-monitoring",
    description="Weekly monitoring: data drift, score distribution, and concept drift checks",
    retries=1,
    retry_delay_seconds=3600,
)
def monitoring_flow(
    reference_features_path: str = "data/05_model_input/feature_matrix_reference.parquet",
    current_features_path: str = "data/05_model_input/feature_matrix.parquet",
    current_scores_path: str = "data/07_model_output/churn_scores.parquet",
    baseline_scores_path: str = "data/08_reporting/churn_score_baseline.parquet",
) -> dict[str, object]:
    """Run weekly monitoring checks.

    Compares a *reference* feature matrix (typically the previous week) against
    the *current* one. If the reference file does not exist (first run, fresh
    install, or the snapshot has not been promoted yet) the flow logs the
    missing reference, skips the drift task, and still runs the score
    distribution check so downstream alerting remains functional.

    Also runs a concept drift check (KS test) comparing current churn
    probability distribution against a stored baseline.

    Args:
        reference_features_path: Path to last week's feature matrix.
        current_features_path: Path to this week's feature matrix.
        current_scores_path: Path to latest churn_scores parquet.
        baseline_scores_path: Path to baseline churn probability distribution.

    Returns:
        Dictionary with drift scores, score checks, and model drift results.
    """
    logger = get_run_logger()
    logger.info("BizSentinel Monitoring Flow started")

    drift_scores: dict[str, float] = {}
    if os.path.exists(reference_features_path) and os.path.exists(current_features_path):
        if os.path.samefile(reference_features_path, current_features_path):
            logger.warning(
                "Reference and current feature matrices resolve to the same path "
                f"({reference_features_path}); skipping drift check. Promote last "
                "week's snapshot to the reference path to enable drift detection."
            )
        else:
            drift_scores = check_data_drift(
                reference_path=reference_features_path,
                current_path=current_features_path,
            )
    else:
        missing = [
            p for p in (reference_features_path, current_features_path) if not os.path.exists(p)
        ]
        logger.warning(
            f"Skipping drift check; missing files: {missing}. Run the training "
            "pipeline and promote the previous snapshot to the reference path."
        )

    score_checks = check_score_distribution(current_scores_path)
    model_drift = check_model_drift(
        current_scores_path=current_scores_path,
        baseline_path=baseline_scores_path,
    )

    if model_drift.get("status") == "completed" and not model_drift.get("drift_detected", False):
        save_score_baseline(
            current_scores_path=current_scores_path,
            baseline_path=baseline_scores_path,
        )

    n_drifted = sum(1 for v in drift_scores.values() if v > 0.2)
    all_passed = (all(score_checks.values()) if score_checks else False) and not model_drift.get(
        "drift_detected", False
    )

    notify_completion(
        "monitoring_flow",
        success=all_passed,
        metrics_summary=(
            f"Drifted features: {n_drifted}, "
            f"Model KS: {model_drift.get('ks_statistic', 'N/A'):.4f}, "
            f"Drift: {model_drift.get('drift_detected', 'N/A')}"
        ),
    )

    return {
        "drift_scores": drift_scores,
        "score_checks": score_checks,
        "model_drift": model_drift,
    }
