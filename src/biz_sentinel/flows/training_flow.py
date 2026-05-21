"""Training Prefect Flow — orchestrates the full training pipeline.

Wraps Kedro pipeline execution with Prefect tasks for observability,
retry logic, and scheduling.
"""

import subprocess
import sys
from datetime import timedelta

from prefect import flow, get_run_logger, task  # type: ignore[import-untyped]
from prefect.tasks import task_input_hash


@task(
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    name="run_kedro_pipeline",
)
def run_kedro_pipeline(pipeline_name: str, env: str = "base") -> bool:
    """Run a Kedro pipeline as a subprocess task.

    Args:
        pipeline_name: Name of the Kedro pipeline to run.
        env: Kedro environment (base, local, production).

    Returns:
        True if pipeline completed successfully.

    Raises:
        RuntimeError: If pipeline exits with non-zero code.
    """
    logger = get_run_logger()
    logger.info(f"Starting Kedro pipeline: {pipeline_name}")

    result = subprocess.run(
        [sys.executable, "-m", "kedro", "run", "--pipeline", pipeline_name, "--env", env],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Pipeline {pipeline_name} failed:\n{result.stderr}")
        raise RuntimeError(f"Kedro pipeline '{pipeline_name}' failed with code {result.returncode}")

    logger.info(f"Pipeline {pipeline_name} completed successfully")
    logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return True


@task(name="validate_data_availability")
def validate_data_availability(required_files: list[str]) -> bool:
    """Check that required input files exist before running pipeline.

    Args:
        required_files: List of file paths that must exist.

    Returns:
        True if all files exist.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    import os

    logger = get_run_logger()
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Required data files missing: {missing}")
    logger.info(f"Data validation passed: {len(required_files)} files found")
    return True


@task(name="notify_completion")
def notify_completion(flow_name: str, success: bool, metrics_summary: str = "") -> None:
    """Log completion status. Extend this to send Slack/email notifications.

    Args:
        flow_name: Name of the completed flow.
        success: Whether the flow completed successfully.
        metrics_summary: Optional string summary of key metrics.
    """
    logger = get_run_logger()
    status = "SUCCESS" if success else "FAILURE"
    logger.info(f"[{status}] {flow_name} completed")
    if metrics_summary:
        logger.info(f"Metrics: {metrics_summary}")


@flow(
    name="biz-sentinel-training",
    description="Full training flow: preprocessing → feature engineering → all three ML modules",
    retries=1,
    retry_delay_seconds=300,
)
def training_flow(
    env: str = "base",
    required_raw_files: list[str] | None = None,
) -> dict[str, bool]:
    """Orchestrate the full BizSentinel training pipeline.

    Runs pipelines in order:
    1. preprocessing
    2. feature_engineering
    3. training (anomaly + segmentation + churn)

    Args:
        env: Kedro environment to use.
        required_raw_files: Optional list of raw data files to validate before running.

    Returns:
        Dictionary of pipeline_name → success status.
    """
    logger = get_run_logger()
    logger.info("BizSentinel Training Flow started")

    results: dict[str, bool] = {}

    if required_raw_files:
        validate_data_availability(required_raw_files)

    for pipeline in ["preprocessing", "feature_engineering", "training"]:
        results[pipeline] = run_kedro_pipeline(pipeline, env=env)

    notify_completion("training_flow", success=all(results.values()))
    return results
