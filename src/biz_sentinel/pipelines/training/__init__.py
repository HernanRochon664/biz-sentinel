"""Training pipeline for BizSentinel."""

from kedro.pipeline import Pipeline, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline.

    Args:
        kwargs: Optional keyword arguments for the pipeline.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline([])