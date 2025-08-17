"""Integration tests for the complete pipeline."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.build import FeatureEngineer
from features.target_encoding import OutOfFoldTargetEncoder
from modeling.cat_model import CatBoostModelFactory
from core.utils import set_seed, create_family_groups, stratified_group_split


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @pytest.fixture
    def pipeline_data(self):
        """Create realistic pipeline test data."""
        np.random.seed(42)
        n_samples = 200
        
        # Create more realistic Titanic-like data
        data = {
            "PassengerId": range(1, n_samples + 1),
            "Survived": np.random.randint(0, 2, n_samples),
            "Pclass": np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
            "Name": [f"Passenger_{i}, Mr. Test" for i in range(n_samples)],
            "Sex": np.random.choice(["male", "female"], n_samples, p=[0.6, 0.4]),
            "Age": np.random.uniform(1, 80, n_samples),
            "SibSp": np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            "Parch": np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            "Ticket": [f"TICKET_{i}" for i in range(n_samples)],
            "Fare": np.random.uniform(5, 500, n_samples),
            "Cabin": [f"A{i}" if i % 4 == 0 else np.nan for i in range(n_samples)],
            "Embarked": np.random.choice(["S", "C", "Q"], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        # Introduce some missing values realistically
        missing_indices = np.random.choice(n_samples, n_samples // 10, replace=False)
        for idx in missing_indices:
            if np.random.random() < 0.5:
                data["Age"][idx] = np.nan
            if np.random.random() < 0.1:
                data["Embarked"][idx] = np.nan
        
        return pd.DataFrame(data)
    
    def test_feature_engineering_to_modeling_pipeline(self, sample_config, pipeline_data):
        """Test feature engineering feeding into modeling."""
        # Feature engineering
        engineer = FeatureEngineer(sample_config)
        engineered_df, _ = engineer.engineer_features(pipeline_data)
        
        # Check that engineered features are valid for modeling
        feature_cols = [col for col in engineered_df.columns 
                       if col not in ["Survived", "PassengerId"]]
        
        X = engineered_df[feature_cols].values
        y = engineered_df["Survived"].values
        
        # Should not have any infinite or extremely large values
        assert np.isfinite(X).all(), "Feature matrix contains non-finite values"
        
        # Create and train model
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        fitted_model = CatBoostModelFactory.fit_with_eval(model, X, y, verbose=False)
        
        # Model should be able to make predictions
        predictions = fitted_model.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_cross_validation_pipeline(self, sample_config, pipeline_data):
        """Test cross-validation pipeline integration."""
        # Setup feature engineering
        engineer = FeatureEngineer(sample_config)
        engineered_df, _ = engineer.engineer_features(pipeline_data)
        
        # Setup cross-validation
        family_groups = create_family_groups(engineered_df)
        cv_splits = stratified_group_split(
            engineered_df,
            family_groups,
            target_col="Survived",
            n_splits=3,
            random_state=42
        )
        
        assert len(cv_splits) == 3
        
        # Test that splits are valid
        total_samples = len(engineered_df)
        for train_idx, val_idx in cv_splits:
            # No overlap between train and validation
            assert len(set(train_idx) & set(val_idx)) == 0
            
            # All indices are covered
            assert len(train_idx) + len(val_idx) == total_samples
            
            # Train and validation sets have reasonable sizes
            assert len(train_idx) > len(val_idx)  # More training than validation
            assert len(val_idx) > 10  # At least some validation samples
    
    def test_target_encoding_pipeline(self, sample_config, pipeline_data):
        """Test target encoding in the pipeline."""
        # Feature engineering first
        engineer = FeatureEngineer(sample_config)
        engineered_df, _ = engineer.engineer_features(pipeline_data)
        
        # Setup CV splits
        family_groups = create_family_groups(engineered_df)
        cv_splits = stratified_group_split(
            engineered_df, family_groups, target_col="Survived", 
            n_splits=3, random_state=42
        )
        
        # Apply target encoding
        encoder = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        features_to_encode = ["Title", "CabinDeck", "Embarked"]
        
        # Filter features that exist
        features_to_encode = [f for f in features_to_encode if f in engineered_df.columns]
        
        if features_to_encode:
            encoded_df = encoder.fit_transform_oof(
                engineered_df, engineered_df["Survived"], 
                features_to_encode, cv_splits
            )
            
            # Check that encoded features are added
            for feature in features_to_encode:
                encoded_col = f"{feature}_SurvivalRate"
                assert encoded_col in encoded_df.columns
                
                # Should not have missing values
                assert not encoded_df[encoded_col].isna().any()
                
                # Should be in reasonable range [0, 1]
                assert encoded_df[encoded_col].min() >= 0
                assert encoded_df[encoded_col].max() <= 1
    
    def test_end_to_end_training_pipeline(self, sample_config, pipeline_data):
        """Test complete end-to-end training pipeline."""
        set_seed(42)
        
        # 1. Feature Engineering
        engineer = FeatureEngineer(sample_config)
        engineered_df, _ = engineer.engineer_features(pipeline_data)
        
        # 2. Cross-validation setup
        family_groups = create_family_groups(engineered_df)
        cv_splits = stratified_group_split(
            engineered_df, family_groups, target_col="Survived", 
            n_splits=3, random_state=42
        )
        
        # 3. Target encoding
        encoder = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        features_to_encode = ["Title", "CabinDeck", "Embarked"]
        features_to_encode = [f for f in features_to_encode if f in engineered_df.columns]
        
        if features_to_encode:
            encoded_df = encoder.fit_transform_oof(
                engineered_df, engineered_df["Survived"], 
                features_to_encode, cv_splits
            )
        else:
            encoded_df = engineered_df
        
        # 4. Model training across folds
        feature_cols = [col for col in encoded_df.columns 
                       if col not in ["Survived", "PassengerId"]]
        X = encoded_df[feature_cols].values
        y = encoded_df["Survived"].values
        
        models = []
        oof_predictions = np.zeros(len(y))
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            config = sample_config["catboost"]
            model = CatBoostModelFactory.create_model(config)
            fitted_model = CatBoostModelFactory.fit_with_eval(
                model, X_train, y_train, X_val, y_val, verbose=False
            )
            
            # Make predictions
            val_pred = fitted_model.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = val_pred
            
            models.append(fitted_model)
        
        # 5. Evaluate results
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        oof_pred_binary = (oof_predictions >= 0.5).astype(int)
        auc_score = roc_auc_score(y, oof_predictions)
        accuracy = accuracy_score(y, oof_pred_binary)
        
        # Basic sanity checks
        assert 0.0 <= auc_score <= 1.0
        assert 0.0 <= accuracy <= 1.0
        assert len(models) == len(cv_splits)
        
        # For random-ish data, expect reasonable but not great performance
        assert auc_score > 0.4  # Better than completely random
        assert accuracy > 0.3   # Better than completely random
    
    def test_inference_pipeline(self, sample_config, pipeline_data):
        """Test inference pipeline with train/test consistency."""
        # Split data into train/test
        train_df = pipeline_data.iloc[:150].copy()
        test_df = pipeline_data.iloc[150:].copy()
        test_df = test_df.drop("Survived", axis=1)  # Remove target for test set
        
        # 1. Feature engineering on train
        engineer = FeatureEngineer(sample_config)
        train_engineered, test_engineered = engineer.engineer_features(train_df, test_df)
        
        # 2. Train simple model
        feature_cols = [col for col in train_engineered.columns 
                       if col not in ["Survived", "PassengerId"]]
        
        X_train = train_engineered[feature_cols].values
        y_train = train_engineered["Survived"].values
        
        config = sample_config["catboost"]
        model = CatBoostModelFactory.create_model(config)
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X_train, y_train, verbose=False
        )
        
        # 3. Make predictions on test set
        X_test = test_engineered[feature_cols].values
        test_predictions = fitted_model.predict_proba(X_test)[:, 1]
        
        # Check predictions
        assert len(test_predictions) == len(test_df)
        assert all(0 <= pred <= 1 for pred in test_predictions)
        
        # Check feature consistency between train and test
        assert list(train_engineered[feature_cols].columns) == list(test_engineered[feature_cols].columns)
        assert X_train.shape[1] == X_test.shape[1]


class TestDataValidationIntegration:
    """Test data validation throughout the pipeline."""
    
    def test_invalid_data_handling(self, sample_config):
        """Test pipeline handling of invalid data."""
        # Create data with various invalid conditions
        invalid_data = pd.DataFrame({
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 2, -1, 1],  # Invalid values (2, -1)
            "Pclass": [1, 2, 3, 4, 0],     # Invalid values (4, 0)
            "Name": ["", None, "Valid Name", "Another", "Name"],
            "Sex": ["male", "female", "invalid", "male", "female"],
            "Age": [-5, 150, 25, np.inf, -np.inf],  # Invalid ages
            "SibSp": [-1, 1, 2, 3, 100],  # Some unrealistic values
            "Parch": [0, 1, -1, 2, 50],   # Some unrealistic values
            "Ticket": ["T1", "T2", "", None, "T5"],
            "Fare": [-10, 0, 50, np.inf, -np.inf],  # Invalid fares
            "Cabin": ["A1", None, "", "B2", "C3"],
            "Embarked": ["S", "C", "Q", "X", None]  # Invalid port "X"
        })
        
        engineer = FeatureEngineer(sample_config)
        
        # Feature engineering should handle invalid data gracefully
        try:
            result_df, _ = engineer.engineer_features(invalid_data)
            
            # After processing, critical values should be cleaned
            assert result_df["Age"].min() >= 0
            assert result_df["Age"].max() <= 100
            assert result_df["Fare"].min() >= 0
            assert not result_df["Age"].isna().any()
            assert not result_df["Embarked"].isna().any()
            
        except Exception as e:
            # If it fails, it should fail gracefully with informative error
            assert isinstance(e, (ValueError, TypeError))
    
    def test_missing_data_patterns(self, sample_config):
        """Test handling of different missing data patterns."""
        # Create data with systematic missing patterns
        base_data = {
            "PassengerId": range(1, 101),
            "Survived": np.random.randint(0, 2, 100),
            "Pclass": np.random.choice([1, 2, 3], 100),
            "Name": [f"Person_{i}, Mr. Test" for i in range(100)],
            "Sex": np.random.choice(["male", "female"], 100),
            "Age": np.random.uniform(1, 80, 100),
            "SibSp": np.random.randint(0, 4, 100),
            "Parch": np.random.randint(0, 3, 100),
            "Ticket": [f"TICKET_{i}" for i in range(100)],
            "Fare": np.random.uniform(5, 500, 100),
            "Cabin": [f"A{i}" if i % 3 == 0 else np.nan for i in range(100)],
            "Embarked": np.random.choice(["S", "C", "Q"], 100)
        }
        
        # Test different missing patterns
        missing_patterns = [
            # All ages missing for first class
            lambda df: df.loc[df["Pclass"] == 1, "Age"] = np.nan,
            # All fares missing for certain embarked ports
            lambda df: df.loc[df["Embarked"] == "Q", "Fare"] = np.nan,
            # Random 50% missing ages
            lambda df: df.loc[np.random.choice(100, 50, replace=False), "Age"] = np.nan,
        ]
        
        for pattern_func in missing_patterns:
            test_df = pd.DataFrame(base_data.copy())
            pattern_func(test_df)
            
            engineer = FeatureEngineer(sample_config)
            result_df, _ = engineer.engineer_features(test_df)
            
            # Should handle missing patterns without failing
            assert not result_df["Age"].isna().any()
            assert not result_df["Fare"].isna().any()
            assert not result_df["Embarked"].isna().any()
    
    def test_data_type_consistency(self, sample_config, pipeline_data):
        """Test data type consistency throughout pipeline."""
        engineer = FeatureEngineer(sample_config)
        result_df, _ = engineer.engineer_features(pipeline_data)
        
        # Check expected data types
        numeric_columns = ["Age", "Fare", "FamilySize"]
        categorical_columns = ["Sex", "Embarked", "Title", "CabinDeck"]
        
        for col in numeric_columns:
            if col in result_df.columns:
                assert pd.api.types.is_numeric_dtype(result_df[col]), f"{col} should be numeric"
        
        for col in categorical_columns:
            if col in result_df.columns:
                # Should be object or category type
                assert result_df[col].dtype in ['object', 'category'], f"{col} should be categorical"


class TestPerformanceIntegration:
    """Test performance characteristics of the integrated pipeline."""
    
    def test_pipeline_memory_usage(self, sample_config, performance_helper):
        """Test memory usage of complete pipeline."""
        # Create larger dataset
        np.random.seed(42)
        n_samples = 1000
        
        large_data = pd.DataFrame({
            "PassengerId": range(1, n_samples + 1),
            "Survived": np.random.randint(0, 2, n_samples),
            "Pclass": np.random.choice([1, 2, 3], n_samples),
            "Name": [f"Person_{i}, Mr. Test" for i in range(n_samples)],
            "Sex": np.random.choice(["male", "female"], n_samples),
            "Age": np.random.uniform(1, 80, n_samples),
            "SibSp": np.random.randint(0, 4, n_samples),
            "Parch": np.random.randint(0, 3, n_samples),
            "Ticket": [f"TICKET_{i}" for i in range(n_samples)],
            "Fare": np.random.uniform(5, 500, n_samples),
            "Cabin": [f"A{i}" if i % 3 == 0 else np.nan for i in range(n_samples)],
            "Embarked": np.random.choice(["S", "C", "Q"], n_samples)
        })
        
        def run_pipeline():
            engineer = FeatureEngineer(sample_config)
            result_df, _ = engineer.engineer_features(large_data)
            return result_df
        
        # Test memory usage
        result, memory_usage = performance_helper.memory_usage(run_pipeline)
        
        # Should use reasonable amount of memory (less than 50MB for 1000 samples)
        assert memory_usage < 50, f"Pipeline used {memory_usage:.2f} MB"
    
    def test_pipeline_execution_time(self, sample_config, performance_helper):
        """Test execution time of pipeline components."""
        # Test with medium-sized dataset
        np.random.seed(42)
        n_samples = 500
        
        medium_data = pd.DataFrame({
            "PassengerId": range(1, n_samples + 1),
            "Survived": np.random.randint(0, 2, n_samples),
            "Pclass": np.random.choice([1, 2, 3], n_samples),
            "Name": [f"Person_{i}, Mr. Test" for i in range(n_samples)],
            "Sex": np.random.choice(["male", "female"], n_samples),
            "Age": np.random.uniform(1, 80, n_samples),
            "SibSp": np.random.randint(0, 4, n_samples),
            "Parch": np.random.randint(0, 3, n_samples),
            "Ticket": [f"TICKET_{i}" for i in range(n_samples)],
            "Fare": np.random.uniform(5, 500, n_samples),
            "Cabin": [f"A{i}" if i % 3 == 0 else np.nan for i in range(n_samples)],
            "Embarked": np.random.choice(["S", "C", "Q"], n_samples)
        })
        
        # Test feature engineering time
        engineer = FeatureEngineer(sample_config)
        _, fe_time = performance_helper.time_function(
            engineer.engineer_features, medium_data
        )
        
        # Should complete feature engineering quickly (less than 2 seconds)
        assert fe_time < 2.0, f"Feature engineering took {fe_time:.2f} seconds"
        
        # Test model training time
        result_df, _ = engineer.engineer_features(medium_data)
        feature_cols = [col for col in result_df.columns 
                       if col not in ["Survived", "PassengerId"]]
        X = result_df[feature_cols].values
        y = result_df["Survived"].values
        
        def train_model():
            config = sample_config["catboost"]
            model = CatBoostModelFactory.create_model(config)
            return CatBoostModelFactory.fit_with_eval(model, X, y, verbose=False)
        
        _, training_time = performance_helper.time_function(train_model)
        
        # Model training should be reasonable (less than 5 seconds for 500 samples)
        assert training_time < 5.0, f"Model training took {training_time:.2f} seconds"


class TestRobustnessIntegration:
    """Test pipeline robustness and error handling."""
    
    def test_pipeline_with_extreme_data_distributions(self, sample_config):
        """Test pipeline with extreme data distributions."""
        # Create extreme but valid data
        extreme_data = pd.DataFrame({
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 0],
            "Pclass": [1, 1, 1],  # All first class
            "Name": ["Person 1, Mr. Test", "Person 2, Mrs. Test", "Person 3, Miss. Test"],
            "Sex": ["male", "male", "male"],  # All male
            "Age": [0.1, 0.2, 0.3],  # All infants
            "SibSp": [0, 0, 0],  # No siblings
            "Parch": [5, 6, 7],  # Many parents/children
            "Ticket": ["SAME", "SAME", "SAME"],  # Same ticket
            "Fare": [1000, 2000, 3000],  # Very expensive
            "Cabin": ["A1", "A2", "A3"],  # All have cabins
            "Embarked": ["S", "S", "S"]  # All same port
        })
        
        engineer = FeatureEngineer(sample_config)
        
        # Should handle extreme distributions
        result_df, _ = engineer.engineer_features(extreme_data)
        
        # Basic sanity checks
        assert len(result_df) == 3
        assert "FamilySize" in result_df.columns
        assert all(result_df["FamilySize"] >= 6)  # Large families due to Parch
    
    def test_pipeline_determinism(self, sample_config, pipeline_data):
        """Test that pipeline produces deterministic results."""
        # Run pipeline twice with same inputs
        engineer1 = FeatureEngineer(sample_config)
        engineer2 = FeatureEngineer(sample_config)
        
        result1, _ = engineer1.engineer_features(pipeline_data.copy())
        result2, _ = engineer2.engineer_features(pipeline_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Test with target encoding
        family_groups = create_family_groups(result1)
        cv_splits = stratified_group_split(
            result1, family_groups, target_col="Survived", 
            n_splits=3, random_state=42
        )
        
        encoder1 = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        encoder2 = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        
        features_to_encode = ["Title", "CabinDeck"]
        features_to_encode = [f for f in features_to_encode if f in result1.columns]
        
        if features_to_encode:
            encoded1 = encoder1.fit_transform_oof(
                result1, result1["Survived"], features_to_encode, cv_splits
            )
            encoded2 = encoder2.fit_transform_oof(
                result2, result2["Survived"], features_to_encode, cv_splits
            )
            
            # Should be identical
            pd.testing.assert_frame_equal(encoded1, encoded2)