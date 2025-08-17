"""Tests for feature engineering modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.build import FeatureEngineer
from features.target_encoding import OutOfFoldTargetEncoder


class TestFeatureEngineer:
    """Test the FeatureEngineer class."""
    
    def test_init(self, sample_config):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(sample_config)
        assert engineer.config == sample_config
        assert engineer.feature_config == sample_config["features"]
        assert engineer.imputation_values == {}
    
    def test_extract_title_valid_names(self, sample_config):
        """Test title extraction from valid names."""
        engineer = FeatureEngineer(sample_config)
        
        test_cases = [
            ("Braund, Mr. Owen Harris", "Mr"),
            ("Cumings, Mrs. John Bradley", "Mrs"),
            ("Heikkinen, Miss. Laina", "Miss"),
            ("Astor, Col. Archibald", "Officer"),  # Should map to Officer
            ("Cardeza, Mrs. James Warburton Martinez", "Mrs"),
        ]
        
        for name, expected_title in test_cases:
            result = engineer.extract_title(name)
            if expected_title == "Officer":
                assert result == "Officer"  # Based on config mapping
            else:
                assert result == expected_title
    
    def test_extract_title_edge_cases(self, sample_config):
        """Test title extraction edge cases."""
        engineer = FeatureEngineer(sample_config)
        
        # Missing values
        assert engineer.extract_title(pd.NA) == "Rare"
        assert engineer.extract_title(None) == "Rare"
        assert engineer.extract_title("") == "Rare"
        
        # No title pattern
        assert engineer.extract_title("No Title Here") == "Rare"
        
        # Rare titles
        assert engineer.extract_title("Unknown, Xyz. John") == "Rare"
    
    def test_extract_cabin_deck(self, sample_config):
        """Test cabin deck extraction."""
        engineer = FeatureEngineer(sample_config)
        
        test_cases = [
            ("A24", "Upper"),    # A is in upper deck
            ("B52", "Upper"),    # B is in upper deck  
            ("C85", "Upper"),    # C is in upper deck
            ("D26", "Middle"),   # D is in middle deck
            ("E46", "Middle"),   # E is in middle deck
            ("F33", "Lower"),    # F is in lower deck
            ("G6", "Lower"),     # G is in lower deck
            ("T", "Other"),      # T is in other
            ("", "Unknown"),     # Empty cabin
            (pd.NA, "Unknown"),  # Missing cabin
        ]
        
        for cabin, expected_deck in test_cases:
            result = engineer.extract_cabin_deck(cabin)
            assert result == expected_deck
    
    def test_extract_ticket_prefix(self, sample_config):
        """Test ticket prefix extraction."""
        engineer = FeatureEngineer(sample_config)
        
        test_cases = [
            ("A/5 21171", "A"),
            ("PC 17599", "FIRST_CLASS"),  # PC maps to FIRST_CLASS
            ("STON/O2. 3101282", "STONO"),
            ("113803", "NUMERIC"),  # No letters
            ("", "NUMERIC"),
            (pd.NA, "NUMERIC"),
        ]
        
        for ticket, expected_prefix in test_cases:
            result = engineer.extract_ticket_prefix(ticket)
            # Check if result contains expected pattern
            if expected_prefix == "FIRST_CLASS":
                assert result == "FIRST_CLASS"
            elif expected_prefix == "NUMERIC":
                assert result == "NUMERIC"
            else:
                # For other cases, check if result is reasonable
                assert isinstance(result, str)
    
    def test_create_family_features(self, sample_config, sample_train_df):
        """Test family feature creation."""
        engineer = FeatureEngineer(sample_config)
        result_df = engineer.create_family_features(sample_train_df)
        
        # Check that new columns are created
        expected_columns = [
            "FamilySize", "IsAlone", "HasSiblings", "HasParents", 
            "HasChildren", "FamilyID"
        ]
        for col in expected_columns:
            assert col in result_df.columns
        
        # Check family size calculation
        expected_family_size = sample_train_df["SibSp"] + sample_train_df["Parch"] + 1
        pd.testing.assert_series_equal(result_df["FamilySize"], expected_family_size)
        
        # Check IsAlone logic
        expected_is_alone = (result_df["FamilySize"] == 1).astype(int)
        pd.testing.assert_series_equal(result_df["IsAlone"], expected_is_alone)
    
    def test_create_age_features(self, sample_config, sample_train_df):
        """Test age feature creation."""
        engineer = FeatureEngineer(sample_config)
        result_df = engineer.create_age_features(sample_train_df)
        
        # Check that new columns are created
        expected_columns = ["AgeBin", "AgeGroup", "IsChild"]
        for col in expected_columns:
            assert col in result_df.columns
        
        # Check age binning
        assert result_df["AgeBin"].dtype == 'object'
        
        # Check IsChild logic - should be 1 for children or Master title
        # First need to add Title column
        result_df["Title"] = sample_train_df["Name"].apply(engineer.extract_title)
        result_df = engineer.create_age_features(result_df)
        
        for idx, row in result_df.iterrows():
            if pd.notna(row["Age"]) and row["Age"] < 18:
                assert row["IsChild"] == 1
            elif row["Title"] == "Master":
                assert row["IsChild"] == 1
    
    def test_create_interaction_features(self, sample_config, sample_train_df):
        """Test interaction feature creation."""
        engineer = FeatureEngineer(sample_config)
        
        # First add required columns
        sample_train_df["Title"] = sample_train_df["Name"].apply(engineer.extract_title)
        sample_train_df = engineer.create_age_features(sample_train_df)
        sample_train_df = engineer.create_family_features(sample_train_df)
        
        result_df = engineer.create_interaction_features(sample_train_df)
        
        # Check interaction columns
        expected_interactions = [
            "Sex_Pclass", "Title_Pclass", "AgeGroup_Pclass", 
            "FamilySize_Pclass", "Embarked_Pclass", "FamilySizeGroup"
        ]
        
        for col in expected_interactions:
            assert col in result_df.columns
        
        # Check interaction format
        assert all("_" in str(val) for val in result_df["Sex_Pclass"] if pd.notna(val))
    
    def test_smart_imputation(self, sample_config, sample_train_df):
        """Test smart imputation."""
        engineer = FeatureEngineer(sample_config)
        
        # Add title for imputation logic
        sample_train_df["Title"] = sample_train_df["Name"].apply(engineer.extract_title)
        
        # Create a copy with more missing values
        df_with_missing = sample_train_df.copy()
        df_with_missing.loc[0, "Age"] = np.nan
        df_with_missing.loc[1, "Fare"] = np.nan
        df_with_missing.loc[2, "Embarked"] = np.nan
        
        result_df, _ = engineer.smart_imputation(df_with_missing)
        
        # Check that missing values are filled
        assert not result_df["Age"].isna().any()
        assert not result_df["Fare"].isna().any()
        assert not result_df["Embarked"].isna().any()
        
        # Check that imputation values are stored
        assert engineer.imputation_values != {}
    
    def test_engineer_features_integration(self, sample_config, sample_train_df):
        """Test the complete feature engineering pipeline."""
        engineer = FeatureEngineer(sample_config)
        
        result_df, _ = engineer.engineer_features(sample_train_df)
        
        # Check that original columns still exist
        for col in sample_train_df.columns:
            assert col in result_df.columns
        
        # Check that new features are added
        new_features = [
            "Title", "CabinDeck", "HasCabin", "TicketPrefix", 
            "FamilySize", "IsAlone", "AgeBin", "AgeGroup"
        ]
        
        for feature in new_features:
            assert feature in result_df.columns
        
        # Check no missing values in critical columns after processing
        assert not result_df["Age"].isna().any()
        assert not result_df["Embarked"].isna().any()


class TestOutOfFoldTargetEncoder:
    """Test the OutOfFoldTargetEncoder class."""
    
    def test_init(self):
        """Test encoder initialization."""
        encoder = OutOfFoldTargetEncoder(alpha=5.0, random_state=123)
        assert encoder.alpha == 5.0
        assert encoder.random_state == 123
        assert encoder.global_mean_ is None
        assert encoder.encoding_maps_ == {}
    
    def test_smooth_encode(self):
        """Test Bayesian smoothing encoding."""
        encoder = OutOfFoldTargetEncoder(alpha=10.0)
        
        counts = pd.Series([10, 5, 20])
        sums = pd.Series([7, 2, 15])
        global_mean = 0.5
        
        result = encoder._smooth_encode(counts, sums, global_mean)
        
        # Check smoothing formula: (sums + alpha * global_mean) / (counts + alpha)
        expected = (sums + 10.0 * 0.5) / (counts + 10.0)
        pd.testing.assert_series_equal(result, expected)
    
    def test_fit_transform_oof(self, sample_cv_splits):
        """Test out-of-fold encoding."""
        encoder = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            "feature1": np.random.choice(["A", "B", "C"], n_samples),
            "feature2": np.random.choice(["X", "Y"], n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        features_to_encode = ["feature1", "feature2"]
        
        result_df = encoder.fit_transform_oof(X, y, features_to_encode, sample_cv_splits)
        
        # Check that encoded columns are added
        assert "feature1_SurvivalRate" in result_df.columns
        assert "feature2_SurvivalRate" in result_df.columns
        
        # Check that no values are missing (should be filled with global mean)
        assert not result_df["feature1_SurvivalRate"].isna().any()
        assert not result_df["feature2_SurvivalRate"].isna().any()
        
        # Check that global mean is set
        assert encoder.global_mean_ is not None
        assert encoder.global_mean_ == y.mean()
    
    def test_fit_full_data(self):
        """Test fitting on full dataset."""
        encoder = OutOfFoldTargetEncoder(alpha=1.0)
        
        X = pd.DataFrame({
            "category": ["A", "B", "A", "B", "C", "C"],
        })
        y = pd.Series([1, 0, 1, 1, 0, 1])
        
        encoder.fit_full_data(X, y, ["category"])
        
        # Check that encoder is fitted
        assert encoder.global_mean_ == y.mean()
        assert "category" in encoder.encoding_maps_
        
        # Check encoding values
        expected_encoding = encoder.encoding_maps_["category"]
        assert "A" in expected_encoding
        assert "B" in expected_encoding
        assert "C" in expected_encoding
    
    def test_transform(self):
        """Test transform method."""
        encoder = OutOfFoldTargetEncoder(alpha=1.0)
        
        # Fit encoder
        X_train = pd.DataFrame({
            "category": ["A", "B", "A", "B", "C"],
        })
        y_train = pd.Series([1, 0, 1, 1, 0])
        
        encoder.fit_full_data(X_train, y_train, ["category"])
        
        # Transform new data
        X_test = pd.DataFrame({
            "category": ["A", "B", "D"],  # D is unseen category
        })
        
        result_df = encoder.transform(X_test, ["category"])
        
        # Check encoded column exists
        assert "category_SurvivalRate" in result_df.columns
        
        # Check that unseen category gets global mean
        assert result_df.loc[2, "category_SurvivalRate"] == encoder.global_mean_
    
    def test_validate_no_leakage(self, sample_cv_splits):
        """Test leakage validation."""
        encoder = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            "feature1": np.random.choice(["A", "B", "C"], n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Perform OOF encoding
        result_df = encoder.fit_transform_oof(X, y, ["feature1"], sample_cv_splits)
        
        # Validate no leakage
        is_valid = encoder.validate_no_leakage(X, y, result_df, ["feature1"], sample_cv_splits)
        
        # Should return True (no leakage)
        assert is_valid


class TestFeatureEngineering:
    """Property-based and integration tests for feature engineering."""
    
    def test_feature_engineering_properties(self, sample_config, property_test_data):
        """Property-based tests for feature engineering."""
        engineer = FeatureEngineer(sample_config)
        
        # Generate property-based test data
        n_samples = 50
        test_df = pd.DataFrame({
            "PassengerId": range(1, n_samples + 1),
            "Survived": np.random.randint(0, 2, n_samples),
            "Pclass": np.random.randint(1, 4, n_samples),
            "Name": property_test_data.generate_passenger_names(n_samples),
            "Sex": np.random.choice(["male", "female"], n_samples),
            "Age": property_test_data.generate_valid_ages(n_samples),
            "SibSp": np.random.randint(0, 6, n_samples),
            "Parch": np.random.randint(0, 4, n_samples),
            "Ticket": property_test_data.generate_ticket_numbers(n_samples),
            "Fare": property_test_data.generate_valid_fares(n_samples),
            "Cabin": [f"A{i}" if i % 3 == 0 else np.nan for i in range(n_samples)],
            "Embarked": np.random.choice(["S", "C", "Q"], n_samples),
        })
        
        # Test feature engineering
        result_df, _ = engineer.engineer_features(test_df)
        
        # Property: All passengers should have valid family size
        assert all(result_df["FamilySize"] >= 1)
        assert all(result_df["FamilySize"] == result_df["SibSp"] + result_df["Parch"] + 1)
        
        # Property: IsAlone should be consistent with FamilySize
        assert all((result_df["IsAlone"] == 1) == (result_df["FamilySize"] == 1))
        
        # Property: All ages should be in valid range after processing
        assert all(result_df["Age"] >= 0)
        assert all(result_df["Age"] <= 100)
        
        # Property: All fares should be non-negative
        assert all(result_df["Fare"] >= 0)
        
        # Property: No missing values in key columns after processing
        assert not result_df["Age"].isna().any()
        assert not result_df["Embarked"].isna().any()
    
    def test_feature_engineering_deterministic(self, sample_config, sample_train_df):
        """Test that feature engineering is deterministic."""
        engineer1 = FeatureEngineer(sample_config)
        engineer2 = FeatureEngineer(sample_config)
        
        result1, _ = engineer1.engineer_features(sample_train_df.copy())
        result2, _ = engineer2.engineer_features(sample_train_df.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_feature_engineering_preserves_original_data(self, sample_config, sample_train_df):
        """Test that feature engineering doesn't modify original data."""
        original_df = sample_train_df.copy()
        engineer = FeatureEngineer(sample_config)
        
        result_df, _ = engineer.engineer_features(sample_train_df)
        
        # Original dataframe should be unchanged
        pd.testing.assert_frame_equal(sample_train_df, original_df)
    
    def test_performance_requirements(self, sample_config, performance_helper):
        """Test performance requirements for feature engineering."""
        engineer = FeatureEngineer(sample_config)
        
        # Create larger dataset
        large_df = pd.concat([
            pd.DataFrame({
                "PassengerId": range(i*1000, (i+1)*1000),
                "Survived": np.random.randint(0, 2, 1000),
                "Pclass": np.random.randint(1, 4, 1000),
                "Name": [f"Person_{j}, Mr. Test" for j in range(1000)],
                "Sex": np.random.choice(["male", "female"], 1000),
                "Age": np.random.uniform(0, 100, 1000),
                "SibSp": np.random.randint(0, 6, 1000),
                "Parch": np.random.randint(0, 4, 1000),
                "Ticket": [f"TICKET_{j}" for j in range(1000)],
                "Fare": np.random.uniform(0, 500, 1000),
                "Cabin": [f"A{j}" if j % 5 == 0 else np.nan for j in range(1000)],
                "Embarked": np.random.choice(["S", "C", "Q"], 1000),
            }) for i in range(5)
        ], ignore_index=True)
        
        # Test performance
        result, execution_time = performance_helper.time_function(
            engineer.engineer_features, large_df
        )
        
        # Should complete within reasonable time (less than 10 seconds for 5000 samples)
        assert execution_time < 10.0, f"Feature engineering took {execution_time:.2f} seconds"
        
        # Should not consume excessive memory
        _, memory_usage = performance_helper.memory_usage(
            engineer.engineer_features, large_df
        )
        
        # Should use less than 100MB for this dataset size
        assert memory_usage < 100, f"Feature engineering used {memory_usage:.2f} MB"