"""Tests for adversarial validation."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent / "src"))

from validation.adv_validation import prepare_adversarial_data, run_adversarial_validation


class TestAdversarialValidation:
    """Test adversarial validation for distribution shift detection."""
    
    def test_prepare_adversarial_data_basic(self):
        """Test basic adversarial data preparation."""
        # Create sample train and test data
        train_df = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 0],
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C']
        })
        
        test_df = pd.DataFrame({
            'PassengerId': [4, 5],
            'feature1': [4, 5],
            'feature2': ['D', 'E']
        })
        
        X, y = prepare_adversarial_data(train_df, test_df)
        
        # Check output shapes and labels
        assert len(X) == 5  # 3 train + 2 test
        assert len(y) == 5
        assert list(y) == [0, 0, 0, 1, 1]  # First 3 are train (0), last 2 are test (1)
        
        # Check that PassengerId and Survived are removed
        assert 'PassengerId' not in X.columns
        assert 'Survived' not in X.columns
        assert 'is_test' not in X.columns
    
    def test_prepare_adversarial_data_missing_target(self):
        """Test adversarial data prep when train data missing target."""
        train_df = pd.DataFrame({
            'PassengerId': [1, 2],
            'feature1': [1, 2]
        })
        
        test_df = pd.DataFrame({
            'PassengerId': [3, 4],
            'feature1': [3, 4]
        })
        
        X, y = prepare_adversarial_data(train_df, test_df)
        
        assert len(X) == 4
        assert len(y) == 4
        assert list(y) == [0, 0, 1, 1]
    
    def test_identical_distributions(self):
        """Test adversarial validation with identical train/test distributions."""
        # Create identical data
        np.random.seed(42)
        base_data = pd.DataFrame({
            'PassengerId': range(1, 101),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'Survived': np.random.randint(0, 2, 100)
        })
        
        # Split into train/test randomly (should be similar distributions)
        train_df = base_data.iloc[:80].copy()
        test_df = base_data.iloc[80:].drop('Survived', axis=1).copy()
        
        X, y = prepare_adversarial_data(train_df, test_df)
        
        # Train simple classifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Convert categorical columns to numeric
        X_numeric = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        
        scores = cross_val_score(clf, X_numeric, y, cv=3, scoring='roc_auc')
        avg_auc = scores.mean()
        
        # AUC should be close to 0.5 (no distribution shift)
        assert 0.4 <= avg_auc <= 0.6
    
    def test_different_distributions(self):
        """Test adversarial validation with different distributions."""
        np.random.seed(42)
        
        # Create train data
        train_df = pd.DataFrame({
            'PassengerId': range(1, 101),
            'feature1': np.random.normal(0, 1, 100),    # Mean 0
            'feature2': np.random.choice(['A', 'B'], 100, p=[0.7, 0.3]),  # Mostly A
            'Survived': np.random.randint(0, 2, 100)
        })
        
        # Create test data with different distribution
        test_df = pd.DataFrame({
            'PassengerId': range(101, 151),
            'feature1': np.random.normal(2, 1, 50),     # Mean 2 (shifted)
            'feature2': np.random.choice(['A', 'B'], 50, p=[0.3, 0.7])   # Mostly B
        })
        
        X, y = prepare_adversarial_data(train_df, test_df)
        
        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Convert categorical to numeric
        X_numeric = X.copy()
        X_numeric['feature2'] = pd.Categorical(X_numeric['feature2']).codes
        
        scores = cross_val_score(clf, X_numeric, y, cv=3, scoring='roc_auc')
        avg_auc = scores.mean()
        
        # Should detect distribution shift (AUC > 0.6)
        assert avg_auc > 0.6
    
    def test_feature_importance_extraction(self):
        """Test that adversarial validation identifies shift-driving features."""
        np.random.seed(42)
        
        # Create data where only feature1 differs between train/test
        train_df = pd.DataFrame({
            'PassengerId': range(1, 101),
            'feature1': np.random.normal(0, 1, 100),    # Different distribution
            'feature2': np.random.normal(5, 1, 100),    # Same distribution
            'feature3': np.random.choice(['X', 'Y'], 100),  # Same distribution
            'Survived': np.random.randint(0, 2, 100)
        })
        
        test_df = pd.DataFrame({
            'PassengerId': range(101, 151),
            'feature1': np.random.normal(3, 1, 50),     # Shifted mean
            'feature2': np.random.normal(5, 1, 50),     # Same mean
            'feature3': np.random.choice(['X', 'Y'], 50)   # Same distribution
        })
        
        X, y = prepare_adversarial_data(train_df, test_df)
        
        # Train classifier and get feature importance
        from sklearn.ensemble import RandomForestClassifier
        
        X_numeric = X.copy()
        for col in X.select_dtypes(include=['object']).columns:
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_numeric, y)
        
        # Get feature importances
        importances = dict(zip(X_numeric.columns, clf.feature_importances_))
        
        # feature1 should have highest importance (it's the one that differs)
        assert importances['feature1'] > importances['feature2']
        assert importances['feature1'] > importances['feature3']
    
    def test_adversarial_validation_integration(self, sample_config, temp_dir):
        """Test full adversarial validation integration."""
        # Create test data files
        train_data = pd.DataFrame({
            'PassengerId': range(1, 51),
            'Survived': np.random.randint(0, 2, 50),
            'Pclass': np.random.choice([1, 2, 3], 50),
            'Name': [f'Person {i}, Mr. Test' for i in range(50)],
            'Sex': np.random.choice(['male', 'female'], 50),
            'Age': np.random.uniform(1, 80, 50),
            'SibSp': np.random.randint(0, 4, 50),
            'Parch': np.random.randint(0, 3, 50),
            'Ticket': [f'TICKET_{i}' for i in range(50)],
            'Fare': np.random.uniform(5, 500, 50),
            'Cabin': [f'A{i}' if i % 3 == 0 else np.nan for i in range(50)],
            'Embarked': np.random.choice(['S', 'C', 'Q'], 50)
        })
        
        test_data = train_data.iloc[:25].drop('Survived', axis=1).copy()
        test_data['PassengerId'] = range(100, 125)
        
        train_path = temp_dir / "train.csv"
        test_path = temp_dir / "test.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Update config with temp paths
        test_config = sample_config.copy()
        test_config["data"]["train_path"] = str(train_path)
        test_config["data"]["test_path"] = str(test_path)
        
        # Run adversarial validation
        results = run_adversarial_validation(test_config)
        
        # Check results structure
        assert 'adversarial_auc' in results
        assert 'shift_detected' in results
        assert 'top_shift_features' in results
        assert 'n_train' in results
        assert 'n_test' in results
        
        # AUC should be reasonable
        assert 0.3 <= results['adversarial_auc'] <= 0.8
    
    def test_missing_test_data_handling(self, sample_config, temp_dir):
        """Test handling when test data is missing."""
        # Create only train data
        train_data = pd.DataFrame({
            'PassengerId': range(1, 101),
            'Survived': np.random.randint(0, 2, 100),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.choice(['A', 'B'], 100)
        })
        
        train_path = temp_dir / "train.csv"
        train_data.to_csv(train_path, index=False)
        
        test_config = sample_config.copy()
        test_config["data"]["train_path"] = str(train_path)
        test_config["data"]["test_path"] = str(temp_dir / "nonexistent_test.csv")
        
        # Should handle gracefully and create synthetic test data
        results = run_adversarial_validation(test_config)
        
        assert 'adversarial_auc' in results
        assert results['n_train'] > 0
        assert results['n_test'] > 0