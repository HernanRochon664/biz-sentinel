"""Pipeline registry for BizSentinel."""

from kedro.pipeline import pipeline  # type: ignore[import-untyped]

# Import pipelines (these will be created later)
# Using a try/except block to allow the project to run even if pipelines don't exist yet
try:
    from biz_sentinel.pipelines import (
        feature_engineering,
        inference,
        preprocessing,
        training,
    )
except ImportError:
    # Create empty pipelines as placeholders
    preprocessing = feature_engineering = training = inference = pipeline([])


def register_pipelines():
    """Register the project's pipelines.

    Returns:
        A dictionary mapping a pipeline name to a ``Pipeline`` object.
    """
    # Create individual pipelines (these will be defined in their respective modules)
    preprocessing_pipeline = preprocessing.create_pipeline()  # type: ignore[union-attr]
    feature_engineering_pipeline = feature_engineering.create_pipeline()  # type: ignore[union-attr]
    training_pipeline = training.create_pipeline()  # type: ignore[union-attr]
    inference_pipeline = inference.create_pipeline()  # type: ignore[union-attr]

    # Define the default pipeline that runs all four in order
    __default__ = (
        preprocessing_pipeline
        + feature_engineering_pipeline
        + training_pipeline
        + inference_pipeline
    )

    return {
        "__default__": __default__,
        "preprocessing": preprocessing_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "training": training_pipeline,
        "inference": inference_pipeline,
        # You can also define combinations like:
        # "preprocessing_plus_feature_eng": preprocessing_pipeline + feature_engineering_pipeline,
    }
