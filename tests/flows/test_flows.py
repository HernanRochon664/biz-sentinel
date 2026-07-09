"""Tests for Prefect flows."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pandas as pd
import pytest
from prefect.testing.utilities import prefect_test_harness

from biz_sentinel.flows.inference_flow import (
    check_champion_model_exists,
    inference_flow,
)
from biz_sentinel.flows.monitoring_flow import (
    check_data_drift,
    check_score_distribution,
    monitoring_flow,
)
from biz_sentinel.flows.training_flow import (
    run_kedro_pipeline,
    training_flow,
    validate_data_availability,
)


@pytest.fixture
def mock_subprocess_success(monkeypatch):
    def mock_run(*args, **kwargs):
        return CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="Pipeline complete",
        )

    monkeypatch.setattr("subprocess.run", mock_run)


@pytest.fixture
def mock_subprocess_failure(monkeypatch):
    def mock_run(*args, **kwargs):
        return CompletedProcess(
            args=args[0] if args else [],
            returncode=1,
            stderr="Error in pipeline",
        )

    monkeypatch.setattr("subprocess.run", mock_run)


@pytest.fixture
def sample_parquet_files(tmp_path):
    feature_matrix = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:03d}" for i in range(1, 21)],
            "feature_a": list(range(20)),
            "feature_b": [x * 1.5 for x in range(20)],
            "feature_c": [float(x) for x in range(20)],
        }
    )
    feature_matrix.to_parquet(tmp_path / "feature_matrix.parquet", index=False)

    churn_scores = pd.DataFrame(
        {
            "customer_hash": [f"hash_{i:03d}" for i in range(1, 21)],
            "churn_probability": [0.1 + i * 0.02 for i in range(20)],
        }
    )
    churn_scores.to_parquet(tmp_path / "churn_scores.parquet", index=False)

    return tmp_path


@pytest.mark.integration
class TestValidateDataAvailability:
    def test_validate_data_availability_passes_when_files_exist(self, tmp_path):
        test_file = tmp_path / "test.parquet"
        test_file.touch()

        with prefect_test_harness():
            result = validate_data_availability([str(test_file)])

        assert result is True

    def test_validate_data_availability_raises_when_missing(self, tmp_path):
        non_existent = str(tmp_path / "nonexistent.parquet")

        with prefect_test_harness(), pytest.raises(FileNotFoundError):
            validate_data_availability([non_existent])


@pytest.mark.integration
class TestRunKedroPipeline:
    def test_run_kedro_pipeline_success(self, mock_subprocess_success):
        with prefect_test_harness():
            result = run_kedro_pipeline("preprocessing")

        assert result is True

    def test_run_kedro_pipeline_failure(self, mock_subprocess_failure):
        with prefect_test_harness(), pytest.raises(RuntimeError, match="failed with code 1"):
            run_kedro_pipeline("preprocessing", env="test")


@pytest.mark.integration
class TestTrainingFlow:
    @pytest.mark.integration
    def test_training_flow_runs_all_three_pipelines(self, mock_subprocess_success):
        with prefect_test_harness():
            result = training_flow(env="base")

        assert "preprocessing" in result
        assert "feature_engineering" in result
        assert "training" in result
        assert all(result.values())


@pytest.mark.integration
class TestCheckChampionModelExists:
    def test_check_champion_model_raises_when_no_mlflow(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client:
            mock_client.side_effect = Exception("MLflow unavailable")

            with prefect_test_harness(), pytest.raises(RuntimeError):
                check_champion_model_exists()


@pytest.mark.integration
class TestInferenceFlow:
    @pytest.mark.integration
    def test_inference_flow_runs_with_mocked_kedro(self, mock_subprocess_success):
        with (
            patch(
                "biz_sentinel.flows.inference_flow.check_champion_model_exists",
                return_value=True,
            ),
            prefect_test_harness(),
        ):
            result = inference_flow(env="test")

        assert result["inference"] is True


@pytest.mark.integration
class TestCheckDataDrift:
    def test_check_data_drift_returns_dict(self, sample_parquet_files):
        ref_path = str(sample_parquet_files / "feature_matrix.parquet")
        cur_path = str(sample_parquet_files / "feature_matrix.parquet")

        with prefect_test_harness():
            result = check_data_drift(ref_path, cur_path)

        assert isinstance(result, dict)

    def test_check_data_drift_missing_file_returns_empty(self, tmp_path):
        non_existent = str(tmp_path / "nonexistent.parquet")

        with prefect_test_harness():
            result = check_data_drift(non_existent, non_existent)

        assert result == {}


@pytest.mark.integration
class TestCheckScoreDistribution:
    def test_check_score_distribution_passes_for_valid_data(self, sample_parquet_files):
        scores_path = str(sample_parquet_files / "churn_scores.parquet")

        with prefect_test_harness():
            result = check_score_distribution(scores_path)

        assert "scores_file_exists" in result
        assert result["scores_file_exists"] is True


@pytest.mark.integration
class TestMonitoringFlow:
    @pytest.mark.integration
    def test_monitoring_flow_completes(self, sample_parquet_files, mock_subprocess_success):
        reference = str(sample_parquet_files / "feature_matrix.parquet")
        current = str(sample_parquet_files / "feature_matrix.parquet")
        scores = str(sample_parquet_files / "churn_scores.parquet")

        with prefect_test_harness():
            result = monitoring_flow(
                reference_features_path=reference,
                current_features_path=current,
                current_scores_path=scores,
            )

        assert "drift_scores" in result
        assert "score_checks" in result

    @pytest.mark.integration
    def test_monitoring_flow_skips_drift_when_paths_match(
        self, sample_parquet_files, mock_subprocess_success
    ):
        same = str(sample_parquet_files / "feature_matrix.parquet")
        scores = str(sample_parquet_files / "churn_scores.parquet")

        with prefect_test_harness():
            result = monitoring_flow(
                reference_features_path=same,
                current_features_path=same,
                current_scores_path=scores,
            )

        assert result["drift_scores"] == {}

    @pytest.mark.integration
    def test_monitoring_flow_skips_drift_when_files_missing(
        self, sample_parquet_files, mock_subprocess_success, tmp_path
    ):
        scores = str(sample_parquet_files / "churn_scores.parquet")
        missing_ref = str(tmp_path / "missing_reference.parquet")
        missing_cur = str(tmp_path / "missing_current.parquet")

        with prefect_test_harness():
            result = monitoring_flow(
                reference_features_path=missing_ref,
                current_features_path=missing_cur,
                current_scores_path=scores,
            )

        assert result["drift_scores"] == {}
