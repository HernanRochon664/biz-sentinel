"""Inference Prefect Flow — daily scoring of all customers."""

from prefect import flow, get_run_logger, task  # type: ignore[import-untyped]

from biz_sentinel.flows.training_flow import notify_completion, run_kedro_pipeline


@task(name="check_champion_model_exists")
def check_champion_model_exists() -> bool:
    """Verify that a champion model exists in MLflow registry before scoring.

    Returns:
        True if champion model found.

    Raises:
        RuntimeError: If no champion model is registered.
    """
    logger = get_run_logger()
    try:
        from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]

        client = MlflowClient()
        models = client.search_registered_models(filter_string="name='biz_sentinel_churn_scorer'")
        if not models:
            raise RuntimeError(
                "No champion model found in MLflow registry. Run training_flow first."
            )
        logger.info(f"Champion model found: {models[0].name}")
        return True
    except Exception as e:
        raise RuntimeError(f"MLflow model check failed: {e}") from e


@flow(
    name="biz-sentinel-inference",
    description="Daily inference flow: score all customers and generate alerts",
    retries=2,
    retry_delay_seconds=1800,
)
def inference_flow(env: str = "base") -> dict[str, bool]:
    """Run daily inference on all customers.

    Checks for champion model, runs inference pipeline,
    and notifies on completion.

    Args:
        env: Kedro environment.

    Returns:
        Dictionary of task_name → success status.
    """
    logger = get_run_logger()
    logger.info("BizSentinel Inference Flow started")

    check_champion_model_exists()
    result = run_kedro_pipeline("inference", env=env)

    notify_completion("inference_flow", success=result)
    return {"inference": result}
