"""Unit tests for base model interface."""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from titanic_enterprise.models.base import BaseModel


class MockModel(BaseModel):
    """Mock implementation of BaseModel for testing."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns)
        self.target_name = y_train.name if hasattr(y_train, 'name') else 'target'
        self.is_trained = True
        return {"accuracy": 0.85, "loss": 0.3}
    
    def predict(self, X):
        self.validate_input(X)
        X_processed = self.preprocess_input(X)
        predictions = [0] * len(X_processed)
        return self.postprocess_output(predictions)
    
    def predict_proba(self, X):
        self.validate_input(X)
        X_processed = self.preprocess_input(X)
        probabilities = [[0.6, 0.4]] * len(X_processed)
        return probabilities
    
    def evaluate(self, X, y):
        return {"accuracy": 0.85, "precision": 0.80, "recall": 0.90}
    
    def get_feature_importance(self):
        if self.feature_names:
            return {name: 0.1 for name in self.feature_names}
        return {}
    
    def save_model(self, path):
        # Mock implementation
        pass
    
    def load_model(self, path):
        # Mock implementation
        self.is_trained = True


class TestBaseModel:
    """Test the BaseModel abstract class."""
    
    @pytest.fixture
    def mock_model(self, mock_model_config):
        """Create a mock model instance."""
        return MockModel(mock_model_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': ['A', 'B', 'A', 'B', 'A']
        })
        y = pd.Series([0, 1, 0, 1, 0], name='target')
        return X, y
    
    def test_model_initialization(self, mock_model_config):
        """Test model initialization."""
        model = MockModel(mock_model_config)
        
        assert model.config == mock_model_config
        assert model.model is None
        assert not model.is_trained
        assert model.feature_names is None
        assert model.target_name is None
    
    def test_model_training(self, mock_model, sample_data):
        """Test model training."""
        X, y = sample_data
        
        result = mock_model.train(X, y)
        
        assert mock_model.is_trained
        assert mock_model.feature_names == list(X.columns)
        assert mock_model.target_name == 'target'
        assert isinstance(result, dict)
        assert 'accuracy' in result
    
    def test_model_prediction(self, mock_model, sample_data):
        """Test model prediction."""
        X, y = sample_data
        
        # Train first
        mock_model.train(X, y)
        
        # Make predictions
        predictions = mock_model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(isinstance(p, int) for p in predictions)
    
    def test_model_predict_proba(self, mock_model, sample_data):
        """Test model probability prediction."""
        X, y = sample_data
        
        # Train first
        mock_model.train(X, y)
        
        # Make probability predictions
        probabilities = mock_model.predict_proba(X)
        
        assert len(probabilities) == len(X)
        assert all(len(p) == 2 for p in probabilities)  # Binary classification
    
    def test_model_evaluation(self, mock_model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Train first
        mock_model.train(X, y)
        
        # Evaluate
        metrics = mock_model.evaluate(X, y)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_get_feature_importance(self, mock_model, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        # Train first
        mock_model.train(X, y)
        
        # Get feature importance
        importance = mock_model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        assert all(name in importance for name in X.columns)
    
    def test_get_model_info(self, mock_model, sample_data):
        """Test model information retrieval."""
        X, y = sample_data
        
        # Before training
        info = mock_model.get_model_info()
        assert info['model_type'] == 'MockModel'
        assert not info['is_trained']
        assert info['feature_names'] is None
        
        # After training
        mock_model.train(X, y)
        info = mock_model.get_model_info()
        assert info['is_trained']
        assert info['feature_names'] == list(X.columns)
        assert info['target_name'] == 'target'
    
    def test_validate_input_success(self, mock_model, sample_data):
        """Test successful input validation."""
        X, y = sample_data
        mock_model.train(X, y)
        
        # Should not raise any exception
        mock_model.validate_input(X)
    
    def test_validate_input_wrong_type(self, mock_model):
        """Test input validation with wrong type."""
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            mock_model.validate_input([1, 2, 3])
    
    def test_validate_input_missing_features(self, mock_model, sample_data):
        """Test input validation with missing features."""
        X, y = sample_data
        mock_model.train(X, y)
        
        # Create DataFrame with missing features
        X_missing = X.drop('feature1', axis=1)
        
        with pytest.raises(ValueError, match="Missing required features"):
            mock_model.validate_input(X_missing)
    
    def test_preprocess_input(self, mock_model, sample_data):
        """Test input preprocessing."""
        X, y = sample_data
        mock_model.train(X, y)
        
        # Add extra column
        X_extra = X.copy()
        X_extra['extra_feature'] = [1, 2, 3, 4, 5]
        
        # Preprocess should select only required features
        X_processed = mock_model.preprocess_input(X_extra)
        
        assert list(X_processed.columns) == mock_model.feature_names
        assert len(X_processed.columns) == len(X.columns)
    
    def test_postprocess_output(self, mock_model):
        """Test output postprocessing."""
        predictions = [0, 1, 0, 1, 0]
        
        # Default implementation should return as-is
        processed = mock_model.postprocess_output(predictions)
        assert processed == predictions
    
    def test_model_repr(self, mock_model):
        """Test model string representation."""
        repr_str = repr(mock_model)
        assert 'MockModel' in repr_str
        assert 'trained=False' in repr_str
        
        # After training
        mock_model.is_trained = True
        repr_str = repr(mock_model)
        assert 'trained=True' in repr_str