"""Constants used throughout the Titanic Enterprise ML infrastructure."""

from typing import List

# Data constants
TITANIC_FEATURES = [
    "PassengerId",
    "Survived", 
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked"
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Pclass"
]

NUMERICAL_FEATURES = [
    "Age",
    "SibSp", 
    "Parch",
    "Fare"
]

TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"

# Feature engineering constants
TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss", 
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Countess": "Rare",
    "Ms": "Miss",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Mme": "Mrs",
    "Capt": "Rare",
    "Sir": "Rare"
}

AGE_GROUPS = {
    "Child": (0, 12),
    "Teen": (13, 19), 
    "Adult": (20, 59),
    "Senior": (60, 100)
}

FARE_GROUPS = {
    "Low": (0, 7.91),
    "Medium": (7.91, 14.45),
    "High": (14.45, 31.0),
    "VeryHigh": (31.0, float('inf'))
}

# Model constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# MLflow constants
MLFLOW_EXPERIMENT_NAME = "titanic-tfdf"
MODEL_REGISTRY_NAME = "titanic_tfdf_model"

# File paths (relative to project root)
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
LOGS_DIR = "logs"
CONFIGS_DIR = "configs"

# File names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission.csv"

# Model performance thresholds
MIN_ACCURACY = 0.80
MIN_PRECISION = 0.75
MIN_RECALL = 0.75
MIN_F1_SCORE = 0.77
MIN_AUC_ROC = 0.85

# Data quality thresholds
MAX_MISSING_RATIO = 0.5
MIN_UNIQUE_VALUES = 2
MAX_CARDINALITY = 100

# Logging constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Environment variables
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_MLFLOW_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_RANDOM_SEED = "RANDOM_SEED"