"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_config() -> DictConfig:
    """Create a sample configuration for testing."""
    config = {
        "project": {
            "name": "titanic-enterprise-test",
            "version": "1.0.0",
            "random_seed": 42,
        },
        "paths": {
            "data_dir": "data",
            "model_dir": "models",
            "output_dir": "outputs",
            "log_dir": "logs",
        },
        "model": {
            "name": "gradient_boosted_trees",
            "task": "CLASSIFICATION",
            "hyperparameters": {
                "num_trees": 10,  # Small for testing
                "max_depth": 3,
                "min_examples": 5,
            },
        },
        "data": {
            "dataset": {
                "name": "titanic",
                "version": "1.0",
            },
            "preprocessing": {
                "imputation": {
                    "age_strategy": "median",
                    "fare_strategy": "median",
                    "embarked_strategy": "mode",
                },
            },
        },
        "mlflow": {
            "tracking_uri": "file:./test_mlruns",
            "experiment_name": "test-experiment",
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def sample_titanic_data() -> pd.DataFrame:
    """Create sample Titanic data for testing."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "Allen, Mr. William Henry",
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "Cabin": [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", "S"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_model_config() -> DictConfig:
    """Create a mock model configuration."""
    config = {
        "name": "test_model",
        "task": "CLASSIFICATION",
        "hyperparameters": {
            "num_trees": 5,
            "max_depth": 3,
        },
    }
    return OmegaConf.create(config)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test-specific environment variables
    os.environ["TESTING"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = "file:./test_mlruns"
    
    yield
    
    # Clean up
    os.environ.pop("TESTING", None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


@pytest.fixture
def mock_trained_model():
    """Create a mock trained model for testing."""
    class MockModel:
        def __init__(self):
            self.is_trained = True
            self.feature_names = ["Pclass", "Sex", "Age", "Fare"]
        
        def predict(self, X):
            # Return dummy predictions
            return [0] * len(X)
        
        def predict_proba(self, X):
            # Return dummy probabilities
            return [[0.6, 0.4]] * len(X)
    
    return MockModel()


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_data: mark test as requiring external data")
    config.addinivalue_line("markers", "requires_model: mark test as requiring trained model")