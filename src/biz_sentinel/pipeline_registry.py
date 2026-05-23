"""Pipeline registry for BizSentinel."""

from kedro.pipeline import Pipeline  # type: ignore[import-untyped]

from biz_sentinel.pipelines.feature_engineering import create_pipeline as create_feature_engineering
from biz_sentinel.pipelines.inference import create_pipeline as create_inference
from biz_sentinel.pipelines.preprocessing import create_pipeline as create_preprocessing
from biz_sentinel.pipelines.training import create_pipeline as create_training


def register_pipelines() -> dict[str, Pipeline]:
    preprocessing = create_preprocessing()
    feature_engineering = create_feature_engineering()
    training = create_training()
    inference = create_inference()

    return {
        "preprocessing": preprocessing,
        "feature_engineering": feature_engineering,
        "training": training,
        "inference": inference,
        "__default__": preprocessing + feature_engineering + training,
    }
