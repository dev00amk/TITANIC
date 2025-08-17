"""Custom exceptions for the Titanic Enterprise ML infrastructure."""


class TitanicEnterpriseError(Exception):
    """Base exception for the project."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(TitanicEnterpriseError):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(TitanicEnterpriseError):
    """Raised when model training fails."""
    pass


class ConfigurationError(TitanicEnterpriseError):
    """Raised when configuration is invalid."""
    pass


class MLflowError(TitanicEnterpriseError):
    """Raised when MLflow operations fail."""
    pass


class DataLoadingError(TitanicEnterpriseError):
    """Raised when data loading fails."""
    pass


class PreprocessingError(TitanicEnterpriseError):
    """Raised when data preprocessing fails."""
    pass


class ModelInferenceError(TitanicEnterpriseError):
    """Raised when model inference fails."""
    pass


class FeatureEngineeringError(TitanicEnterpriseError):
    """Raised when feature engineering fails."""
    pass