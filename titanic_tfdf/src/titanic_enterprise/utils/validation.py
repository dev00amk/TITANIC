"""Validation utilities for data and model validation."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from titanic_enterprise.utils.constants import (
    TITANIC_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN,
    MAX_MISSING_RATIO,
    MIN_UNIQUE_VALUES,
    MAX_CARDINALITY
)
from titanic_enterprise.utils.exceptions import DataValidationError
from titanic_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    allow_extra_columns: bool = True
) -> None:
    """
    Validate DataFrame schema against required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        allow_extra_columns: Whether to allow extra columns
        
    Raises:
        DataValidationError: If schema validation fails
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}",
            details={"missing_columns": list(missing_columns)}
        )
    
    if not allow_extra_columns:
        extra_columns = set(df.columns) - set(required_columns)
        if extra_columns:
            raise DataValidationError(
                f"Unexpected columns found: {extra_columns}",
                details={"extra_columns": list(extra_columns)}
            )
    
    logger.debug(f"Schema validation passed for DataFrame with {len(df.columns)} columns")


def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> None:
    """
    Validate DataFrame column data types.
    
    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types
        
    Raises:
        DataValidationError: If data type validation fails
    """
    type_errors = []
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            continue
            
        actual_type = str(df[column].dtype)
        
        # Check if types match (with some flexibility)
        if expected_type == "int" and not pd.api.types.is_integer_dtype(df[column]):
            type_errors.append(f"{column}: expected int, got {actual_type}")
        elif expected_type == "float" and not pd.api.types.is_numeric_dtype(df[column]):
            type_errors.append(f"{column}: expected float, got {actual_type}")
        elif expected_type == "str" and not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
            type_errors.append(f"{column}: expected string, got {actual_type}")
    
    if type_errors:
        raise DataValidationError(
            f"Data type validation failed: {type_errors}",
            details={"type_errors": type_errors}
        )
    
    logger.debug("Data type validation passed")


def validate_data_quality(df: pd.DataFrame, config: Optional[DictConfig] = None) -> Dict[str, Any]:
    """
    Validate data quality and return quality report.
    
    Args:
        df: DataFrame to validate
        config: Optional configuration with quality thresholds
        
    Returns:
        Dictionary with quality metrics and validation results
        
    Raises:
        DataValidationError: If critical quality issues are found
    """
    if config:
        max_missing_ratio = config.get("quality", {}).get("max_missing_ratio", MAX_MISSING_RATIO)
        min_unique_values = config.get("quality", {}).get("min_unique_values", MIN_UNIQUE_VALUES)
        max_cardinality = config.get("quality", {}).get("max_cardinality", MAX_CARDINALITY)
    else:
        max_missing_ratio = MAX_MISSING_RATIO
        min_unique_values = MIN_UNIQUE_VALUES
        max_cardinality = MAX_CARDINALITY
    
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "unique_values": {},
        "data_types": {},
        "quality_issues": []
    }
    
    # Check missing values
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_ratio = missing_count / len(df)
        quality_report["missing_values"][column] = {
            "count": int(missing_count),
            "ratio": float(missing_ratio)
        }
        
        if missing_ratio > max_missing_ratio:
            issue = f"Column '{column}' has {missing_ratio:.2%} missing values (threshold: {max_missing_ratio:.2%})"
            quality_report["quality_issues"].append(issue)
    
    # Check unique values
    for column in df.columns:
        unique_count = df[column].nunique()
        quality_report["unique_values"][column] = int(unique_count)
        
        if unique_count < min_unique_values:
            issue = f"Column '{column}' has only {unique_count} unique values (minimum: {min_unique_values})"
            quality_report["quality_issues"].append(issue)
        
        if unique_count > max_cardinality and pd.api.types.is_object_dtype(df[column]):
            issue = f"Column '{column}' has {unique_count} unique values (maximum: {max_cardinality})"
            quality_report["quality_issues"].append(issue)
    
    # Check data types
    for column in df.columns:
        quality_report["data_types"][column] = str(df[column].dtype)
    
    # Log quality issues
    if quality_report["quality_issues"]:
        logger.warning(f"Found {len(quality_report['quality_issues'])} data quality issues")
        for issue in quality_report["quality_issues"]:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data quality validation passed")
    
    return quality_report


def validate_titanic_data(df: pd.DataFrame, is_train: bool = True) -> None:
    """
    Validate Titanic dataset specific requirements.
    
    Args:
        df: DataFrame to validate
        is_train: Whether this is training data (should have target column)
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check required columns
    required_columns = TITANIC_FEATURES.copy()
    if not is_train:
        required_columns.remove(TARGET_COLUMN)
    
    validate_dataframe_schema(df, required_columns, allow_extra_columns=True)
    
    # Check ID column
    if ID_COLUMN not in df.columns:
        raise DataValidationError(f"Missing ID column: {ID_COLUMN}")
    
    # Check for duplicate IDs
    if df[ID_COLUMN].duplicated().any():
        duplicates = df[df[ID_COLUMN].duplicated()][ID_COLUMN].tolist()
        raise DataValidationError(
            f"Duplicate IDs found: {duplicates}",
            details={"duplicate_ids": duplicates}
        )
    
    # Check target column for training data
    if is_train:
        if TARGET_COLUMN not in df.columns:
            raise DataValidationError(f"Missing target column: {TARGET_COLUMN}")
        
        # Check target values
        valid_targets = {0, 1}
        invalid_targets = set(df[TARGET_COLUMN].dropna().unique()) - valid_targets
        if invalid_targets:
            raise DataValidationError(
                f"Invalid target values: {invalid_targets}. Expected: {valid_targets}",
                details={"invalid_targets": list(invalid_targets)}
            )
    
    # Check specific column constraints
    if "Pclass" in df.columns:
        valid_classes = {1, 2, 3}
        invalid_classes = set(df["Pclass"].dropna().unique()) - valid_classes
        if invalid_classes:
            raise DataValidationError(
                f"Invalid Pclass values: {invalid_classes}. Expected: {valid_classes}",
                details={"invalid_pclass": list(invalid_classes)}
            )
    
    if "Sex" in df.columns:
        valid_sex = {"male", "female"}
        invalid_sex = set(df["Sex"].dropna().unique()) - valid_sex
        if invalid_sex:
            raise DataValidationError(
                f"Invalid Sex values: {invalid_sex}. Expected: {valid_sex}",
                details={"invalid_sex": list(invalid_sex)}
            )
    
    if "Embarked" in df.columns:
        valid_embarked = {"C", "Q", "S"}
        invalid_embarked = set(df["Embarked"].dropna().unique()) - valid_embarked
        if invalid_embarked:
            raise DataValidationError(
                f"Invalid Embarked values: {invalid_embarked}. Expected: {valid_embarked}",
                details={"invalid_embarked": list(invalid_embarked)}
            )
    
    logger.info(f"Titanic data validation passed for {'training' if is_train else 'test'} data")


def validate_predictions(predictions: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
    """
    Validate model predictions.
    
    Args:
        predictions: Model predictions to validate
        
    Returns:
        Validated predictions as numpy array
        
    Raises:
        DataValidationError: If predictions are invalid
    """
    # Convert to numpy array
    if isinstance(predictions, (pd.Series, list)):
        predictions = np.array(predictions)
    
    if not isinstance(predictions, np.ndarray):
        raise DataValidationError(f"Predictions must be array-like, got {type(predictions)}")
    
    # Check for valid values (0 or 1 for binary classification)
    valid_values = {0, 1}
    unique_values = set(np.unique(predictions))
    invalid_values = unique_values - valid_values
    
    if invalid_values:
        raise DataValidationError(
            f"Invalid prediction values: {invalid_values}. Expected: {valid_values}",
            details={"invalid_predictions": list(invalid_values)}
        )
    
    # Check for NaN values
    if np.isnan(predictions).any():
        nan_count = np.isnan(predictions).sum()
        raise DataValidationError(
            f"Predictions contain {nan_count} NaN values",
            details={"nan_count": int(nan_count)}
        )
    
    logger.debug(f"Prediction validation passed for {len(predictions)} predictions")
    return predictions


def validate_model_metrics(metrics: Dict[str, float], thresholds: Optional[Dict[str, float]] = None) -> None:
    """
    Validate model performance metrics against thresholds.
    
    Args:
        metrics: Dictionary of metric names and values
        thresholds: Optional dictionary of metric thresholds
        
    Raises:
        DataValidationError: If metrics don't meet thresholds
    """
    if thresholds is None:
        thresholds = {}
    
    failed_metrics = []
    
    for metric_name, metric_value in metrics.items():
        if metric_name in thresholds:
            threshold = thresholds[metric_name]
            if metric_value < threshold:
                failed_metrics.append(f"{metric_name}: {metric_value:.4f} < {threshold:.4f}")
    
    if failed_metrics:
        raise DataValidationError(
            f"Model metrics below thresholds: {failed_metrics}",
            details={"failed_metrics": failed_metrics}
        )
    
    logger.info("Model metrics validation passed")


def validate_config(config: DictConfig, required_keys: List[str]) -> None:
    """
    Validate configuration contains required keys.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys in dot notation
        
    Raises:
        DataValidationError: If required keys are missing
    """
    from omegaconf import OmegaConf
    
    missing_keys = []
    
    for key in required_keys:
        try:
            value = OmegaConf.select(config, key)
            if value is None:
                missing_keys.append(key)
        except Exception:
            missing_keys.append(key)
    
    if missing_keys:
        raise DataValidationError(
            f"Missing required configuration keys: {missing_keys}",
            details={"missing_keys": missing_keys}
        )
    
    logger.debug("Configuration validation passed")