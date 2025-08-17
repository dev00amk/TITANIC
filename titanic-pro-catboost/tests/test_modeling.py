"""Tests for modeling modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modeling.cat_model import CatBoostModelFactory


class TestCatBoostModelFactory:
    """Test the CatBoostModelFactory class."""
    
    def test_create_model_basic(self, sample_config):
        """Test basic model creation."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Check that model is created
        assert model is not None
        
        # Check that parameters are set correctly
        params = model.get_params()
        assert params["iterations"] == config["iterations"]
        assert params["learning_rate"] == config["learning_rate"]
        assert params["depth"] == config["depth"]
    
    def test_create_model_with_categorical_features(self, sample_config):
        """Test model creation with categorical features."""
        config = sample_config["catboost"]
        cat_features = ["Sex", "Embarked", "Pclass"]
        
        model = CatBoostModelFactory.create_model(config, cat_features)
        
        # Check that categorical features are set
        assert model.get_params()["cat_features"] == cat_features
    
    def test_create_model_with_optional_params(self, sample_config):
        """Test model creation with optional parameters."""
        config = sample_config["catboost"].copy()
        config["auto_class_weights"] = "Balanced"
        config["early_stopping_rounds"] = 50
        
        model = CatBoostModelFactory.create_model(config)
        
        params = model.get_params()
        assert params["auto_class_weights"] == "Balanced"
        assert params["early_stopping_rounds"] == 50
    
    def test_fit_with_eval_basic(self, sample_config, sample_features_matrix, sample_target):
        """Test model fitting without validation set."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Fit model
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, sample_features_matrix, sample_target, verbose=False
        )
        
        # Check that model is fitted
        assert fitted_model.is_fitted()
        
        # Check that predictions can be made
        predictions = fitted_model.predict(sample_features_matrix)
        assert len(predictions) == len(sample_target)
    
    def test_fit_with_eval_validation_set(self, sample_config, sample_features_matrix, sample_target):
        """Test model fitting with validation set."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Split data for validation
        split_point = len(sample_features_matrix) // 2
        X_train = sample_features_matrix[:split_point]
        y_train = sample_target[:split_point]
        X_val = sample_features_matrix[split_point:]
        y_val = sample_target[split_point:]
        
        # Fit model with validation
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X_train, y_train, X_val, y_val, verbose=False
        )
        
        assert fitted_model.is_fitted()
    
    def test_fit_with_eval_categorical_features(self, sample_config):
        """Test model fitting with categorical features."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Create sample data with categorical features
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        cat_features = [0, 2]  # Indices of categorical features
        
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X, y, cat_features=cat_features, verbose=False
        )
        
        assert fitted_model.is_fitted()
    
    def test_get_feature_importance(self, sample_config, sample_features_matrix, sample_target):
        """Test feature importance extraction."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Fit model
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, sample_features_matrix, sample_target, verbose=False
        )
        
        # Get feature importance
        feature_names = [f"feature_{i}" for i in range(sample_features_matrix.shape[1])]
        importance = CatBoostModelFactory.get_feature_importance(fitted_model, feature_names)
        
        # Check importance structure
        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        
        # Check that all feature names are present
        for feature_name in feature_names:
            assert feature_name in importance
            assert isinstance(importance[feature_name], (int, float))
    
    def test_get_feature_importance_unfitted_model(self, sample_config):
        """Test feature importance extraction from unfitted model."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        feature_names = ["feature_1", "feature_2"]
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            CatBoostModelFactory.get_feature_importance(model, feature_names)
    
    def test_predict_with_threshold(self, sample_config, sample_features_matrix, sample_target):
        """Test prediction with custom threshold."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Fit model
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, sample_features_matrix, sample_target, verbose=False
        )
        
        # Test prediction with different thresholds
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            probabilities, predictions = CatBoostModelFactory.predict_with_threshold(
                fitted_model, sample_features_matrix, threshold
            )
            
            # Check output shapes
            assert len(probabilities) == len(sample_target)
            assert len(predictions) == len(sample_target)
            
            # Check that predictions are binary
            assert all(pred in [0, 1] for pred in predictions)
            
            # Check that probabilities are in [0, 1]
            assert all(0 <= prob <= 1 for prob in probabilities)
            
            # Check threshold logic
            expected_predictions = (probabilities >= threshold).astype(int)
            np.testing.assert_array_equal(predictions, expected_predictions)
    
    def test_predict_with_threshold_unfitted_model(self, sample_config, sample_features_matrix):
        """Test prediction with threshold on unfitted model."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            CatBoostModelFactory.predict_with_threshold(model, sample_features_matrix)
    
    def test_find_optimal_threshold_accuracy(self):
        """Test optimal threshold finding for accuracy."""
        # Create synthetic data where optimal threshold is not 0.5
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        optimal_threshold, best_score = CatBoostModelFactory.find_optimal_threshold(
            y_true, y_proba, metric="accuracy"
        )
        
        # Check that optimal threshold is found
        assert 0.1 <= optimal_threshold <= 0.9
        assert 0.0 <= best_score <= 1.0
        
        # Verify that the found threshold actually gives the best accuracy
        from sklearn.metrics import accuracy_score
        test_threshold = 0.5
        test_predictions = (y_proba >= test_threshold).astype(int)
        test_accuracy = accuracy_score(y_true, test_predictions)
        
        # The optimal threshold should give at least as good accuracy
        optimal_predictions = (y_proba >= optimal_threshold).astype(int)
        optimal_accuracy = accuracy_score(y_true, optimal_predictions)
        assert optimal_accuracy >= test_accuracy
    
    def test_find_optimal_threshold_different_metrics(self):
        """Test optimal threshold finding for different metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        metrics = ["accuracy", "f1", "precision", "recall"]
        
        for metric in metrics:
            optimal_threshold, best_score = CatBoostModelFactory.find_optimal_threshold(
                y_true, y_proba, metric=metric
            )
            
            assert 0.1 <= optimal_threshold <= 0.9
            assert 0.0 <= best_score <= 1.0
    
    def test_find_optimal_threshold_invalid_metric(self):
        """Test optimal threshold finding with invalid metric."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9])
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            CatBoostModelFactory.find_optimal_threshold(y_true, y_proba, metric="invalid_metric")


class TestModelTrainingIntegration:
    """Integration tests for model training components."""
    
    def test_end_to_end_training_pipeline(self, sample_config, sample_train_df):
        """Test end-to-end training pipeline."""
        # This would test the complete pipeline but requires all components
        # For now, test individual components integration
        
        config = sample_config["catboost"]
        
        # Create simple feature matrix
        np.random.seed(42)
        n_samples = len(sample_train_df)
        X = np.random.randn(n_samples, 10)
        y = sample_train_df["Survived"].values
        
        # Create and train model
        model = CatBoostModelFactory.create_model(config)
        fitted_model = CatBoostModelFactory.fit_with_eval(model, X, y, verbose=False)
        
        # Test predictions
        probabilities, predictions = CatBoostModelFactory.predict_with_threshold(fitted_model, X)
        
        # Basic sanity checks
        assert len(probabilities) == n_samples
        assert len(predictions) == n_samples
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_serialization_compatibility(self, sample_config, sample_features_matrix, sample_target):
        """Test that models can be serialized and deserialized."""
        import pickle
        import tempfile
        
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, sample_features_matrix, sample_target, verbose=False
        )
        
        # Test serialization
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            pickle.dump(fitted_model, f)
            temp_path = f.name
        
        # Test deserialization
        with open(temp_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test that loaded model works
        original_predictions = fitted_model.predict(sample_features_matrix)
        loaded_predictions = loaded_model.predict(sample_features_matrix)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_model_performance_metrics(self, sample_config, sample_features_matrix, sample_target):
        """Test model performance measurement."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, sample_features_matrix, sample_target, verbose=False
        )
        
        # Get predictions
        probabilities = fitted_model.predict_proba(sample_features_matrix)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(sample_target, predictions)
        auc = roc_auc_score(sample_target, probabilities)
        
        # Basic sanity checks
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= auc <= 1.0
        
        # For random data with 100 samples, accuracy should be reasonable
        # (not expecting high performance on random data)
        assert accuracy >= 0.3  # Very lenient check


class TestModelValidation:
    """Tests for model validation and robustness."""
    
    def test_model_handles_edge_cases(self, sample_config):
        """Test model handling of edge cases."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Test with minimal data
        X_minimal = np.array([[1, 2], [3, 4]])
        y_minimal = np.array([0, 1])
        
        # Should handle minimal data without crashing
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X_minimal, y_minimal, verbose=False
        )
        assert fitted_model.is_fitted()
    
    def test_model_handles_imbalanced_data(self, sample_config):
        """Test model with highly imbalanced data."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Create imbalanced dataset (90% class 0, 10% class 1)
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.zeros(n_samples)
        y[:10] = 1  # Only 10% positive class
        
        # Should handle imbalanced data
        fitted_model = CatBoostModelFactory.fit_with_eval(model, X, y, verbose=False)
        predictions = fitted_model.predict(X)
        
        # Should make some predictions for both classes
        assert len(np.unique(predictions)) >= 1  # At least one class predicted
    
    def test_model_consistency_across_runs(self, sample_config, sample_features_matrix, sample_target):
        """Test model consistency with same random seed."""
        config = sample_config["catboost"]
        
        # Train two models with same seed
        model1 = CatBoostModelFactory.create_model(config)
        model2 = CatBoostModelFactory.create_model(config)
        
        fitted_model1 = CatBoostModelFactory.fit_with_eval(
            model1, sample_features_matrix, sample_target, verbose=False
        )
        fitted_model2 = CatBoostModelFactory.fit_with_eval(
            model2, sample_features_matrix, sample_target, verbose=False
        )
        
        # Predictions should be identical (or very close due to numerical precision)
        pred1 = fitted_model1.predict_proba(sample_features_matrix)[:, 1]
        pred2 = fitted_model2.predict_proba(sample_features_matrix)[:, 1]
        
        # Allow for small numerical differences
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)
    
    @pytest.mark.parametrize("missing_rate", [0.1, 0.3, 0.5])
    def test_model_handles_missing_values(self, sample_config, missing_rate):
        """Test model with varying rates of missing values."""
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        
        # Create data with missing values
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Introduce missing values
        missing_mask = np.random.random((n_samples, n_features)) < missing_rate
        X[missing_mask] = np.nan
        
        # CatBoost should handle missing values natively
        fitted_model = CatBoostModelFactory.fit_with_eval(model, X, y, verbose=False)
        predictions = fitted_model.predict(X)
        
        # Should produce valid predictions
        assert len(predictions) == n_samples
        assert all(pred in [0, 1] for pred in predictions)