"""
Conftest for pipeline tests.

Patches mlflow at the sys.modules level BEFORE any test imports training
functions. This prevents mlflow.sklearn.log_model() from serializing models,
making training tests ~10x faster.
"""
from unittest.mock import MagicMock
import sys

# Replace mlflow in sys.modules before any test code imports it.
# Training functions do `import mlflow` inside their bodies —
# Python resolves this from sys.modules, so this intercepts all calls.
_mock_mlflow = MagicMock()
sys.modules["mlflow"] = _mock_mlflow
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["mlflow.tracking"] = MagicMock()
