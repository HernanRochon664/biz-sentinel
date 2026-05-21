"""Inference pipeline for BizSentinel."""

from kedro.pipeline import Pipeline, pipeline  # type: ignore[import-untyped]


def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline.

    Args:
        kwargs: Optional keyword arguments for the pipeline.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline([])
