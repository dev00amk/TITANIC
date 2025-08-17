"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

# Sample data for testing
SAMPLE_TRAIN_DATA = {
    "PassengerId": [1, 2, 3, 4, 5],
    "Survived": [0, 1, 1, 0, 1],
    "Pclass": [3, 1, 3, 1, 3],
    "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", 
             "Heikkinen, Miss. Laina", "Futrelle, Mrs. Jacques Heath",
             "Allen, Mr. William Henry"],
    "Sex": ["male", "female", "female", "female", "male"],
    "Age": [22.0, 38.0, 26.0, 35.0, np.nan],
    "SibSp": [1, 1, 0, 1, 0],
    "Parch": [0, 0, 0, 0, 0],
    "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
    "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
    "Cabin": [np.nan, "C85", np.nan, "C123", np.nan],
    "Embarked": ["S", "C", "S", "S", "S"]
}

SAMPLE_TEST_DATA = {
    "PassengerId": [892, 893, 894, 895, 896],
    "Pclass": [3, 3, 2, 3, 3],
    "Name": ["Kelly, Mr. James", "Wilkes, Mrs. James", "Kraeff, Mr. Theodor",
             "Rothes, the Countess. of", "Svensson, Mr. Johan"],
    "Sex": ["male", "female", "male", "female", "male"],
    "Age": [34.5, 47.0, np.nan, 49.0, 14.0],
    "SibSp": [0, 1, 0, 0, 0],
    "Parch": [0, 0, 0, 0, 0],
    "Ticket": ["330911", "363272", "349909", "11755", "347082"],
    "Fare": [7.8292, 7.0, 8.6625, 227.525, 7.4958],
    "Cabin": [np.nan, np.nan, np.nan, "B77", np.nan],
    "Embarked": ["Q", "S", "S", "C", "S"]
}


@pytest.fixture
def sample_train_df():
    """Sample training dataframe for testing."""
    return pd.DataFrame(SAMPLE_TRAIN_DATA)


@pytest.fixture
def sample_test_df():
    """Sample test dataframe for testing."""
    return pd.DataFrame(SAMPLE_TEST_DATA)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "cv": {
            "n_splits": 3,
            "n_seeds": 2,
            "random_state": 42
        },
        "catboost": {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 4,
            "l2_leaf_reg": 3,
            "verbose": False,
            "allow_writing_files": False
        },
        "features": {
            "title_mapping": {
                "standard": ["Mr", "Mrs", "Miss", "Master"],
                "nobility": ["Sir", "Lady", "Don", "Dona"],
                "officer": ["Capt", "Col", "Major", "Rev"],
                "professional": ["Dr"]
            },
            "deck_mapping": {
                "upper": ["A", "B", "C"],
                "middle": ["D", "E"],
                "lower": ["F", "G"],
                "other": ["T"]
            },
            "age_bins": [0, 12, 18, 30, 50, 65, 100],
            "age_labels": ["Child", "Teen", "YoungAdult", "Adult", "Senior", "Elderly"]
        },
        "target_encoding": {
            "enabled": True,
            "alpha": 10.0,
            "features": ["Title", "CabinDeck", "TicketPrefix", "Embarked"]
        },
        "validation": {
            "age_range": [0, 100],
            "fare_min": 0
        },
        "data": {
            "train_path": "test_train.csv",
            "test_path": "test_test.csv"
        }
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_features_matrix():
    """Sample feature matrix for testing."""
    np.random.seed(42)
    return np.random.randn(100, 15)


@pytest.fixture
def sample_target():
    """Sample target vector for testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


@pytest.fixture
def sample_cv_splits():
    """Sample CV splits for testing."""
    np.random.seed(42)
    indices = np.arange(100)
    splits = []
    for i in range(3):
        val_start = i * 20
        val_end = (i + 1) * 20
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        splits.append((train_idx, val_idx))
    return splits


@pytest.fixture
def setup_test_data_files(temp_dir, sample_train_df, sample_test_df):
    """Setup test data files in temp directory."""
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    
    sample_train_df.to_csv(train_path, index=False)
    sample_test_df.to_csv(test_path, index=False)
    
    return {"train_path": train_path, "test_path": test_path}


class PropertyTestData:
    """Generate property-based test data."""
    
    @staticmethod
    def generate_passenger_names(n: int) -> list:
        """Generate valid passenger names."""
        titles = ["Mr.", "Mrs.", "Miss.", "Master.", "Dr.", "Rev.", "Capt."]
        surnames = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore"]
        first_names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael"]
        
        names = []
        for i in range(n):
            surname = np.random.choice(surnames)
            title = np.random.choice(titles)
            first_name = np.random.choice(first_names)
            names.append(f"{surname}, {title} {first_name}")
        
        return names
    
    @staticmethod
    def generate_valid_ages(n: int) -> list:
        """Generate valid age values."""
        return np.random.uniform(0, 100, n).tolist()
    
    @staticmethod
    def generate_valid_fares(n: int) -> list:
        """Generate valid fare values."""
        return np.random.uniform(0, 500, n).tolist()
    
    @staticmethod
    def generate_ticket_numbers(n: int) -> list:
        """Generate valid ticket numbers."""
        prefixes = ["PC", "PARIS", "SC", "SOTON", "CA", ""]
        tickets = []
        for i in range(n):
            prefix = np.random.choice(prefixes)
            number = np.random.randint(1000, 999999)
            if prefix:
                tickets.append(f"{prefix} {number}")
            else:
                tickets.append(str(number))
        return tickets


@pytest.fixture
def property_test_data():
    """Property-based test data generator."""
    return PropertyTestData


# Performance test helpers
class PerformanceTestHelper:
    """Helper for performance testing."""
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time a function execution."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def memory_usage(func, *args, **kwargs):
        """Measure memory usage of a function."""
        import tracemalloc
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak / 1024 / 1024  # Convert to MB


@pytest.fixture
def performance_helper():
    """Performance testing helper."""
    return PerformanceTestHelper