"""Tests for OOF target encoding leakage validation."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.target_encoding import OutOfFoldTargetEncoder


class TestOOFLeakageValidation:
    """Test OOF target encoding for leakage prevention."""
    
    def test_oof_vs_full_data_encoding_difference(self):
        """Test that OOF encoding differs from full-data encoding."""
        # Create sample data
        X = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'] * 5  # 30 samples
        })
        y = pd.Series([1, 0, 1, 1, 0, 1] * 5)  # Mixed targets
        
        # Create CV splits
        cv_splits = [
            (np.array(range(15)), np.array(range(15, 30))),
            (np.array(list(range(0, 7)) + list(range(22, 30))), np.array(range(7, 22)))
        ]
        
        # OOF encoding
        encoder = OutOfFoldTargetEncoder(alpha=1.0, random_state=42)
        oof_encoded = encoder.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # Full data encoding
        encoder_full = OutOfFoldTargetEncoder(alpha=1.0)
        encoder_full.fit_full_data(X, y, ['category'])
        full_encoded = encoder_full.transform(X, ['category'])
        
        # OOF and full encodings should be different
        oof_values = oof_encoded['category_SurvivalRate'].values
        full_values = full_encoded['category_SurvivalRate'].values
        
        # At least some values should be different
        assert not np.allclose(oof_values, full_values, rtol=1e-10)
    
    def test_oof_encoding_no_future_information(self):
        """Test that OOF encoding doesn't use future information."""
        # Create data where target has clear pattern
        X = pd.DataFrame({
            'category': ['A'] * 10 + ['B'] * 10
        })
        # First 10 are positive, last 10 are negative
        y = pd.Series([1] * 10 + [0] * 10)
        
        # CV split that would reveal future info if leaked
        cv_splits = [
            (np.array(range(10)), np.array(range(10, 20)))  # Train on A, test on B
        ]
        
        encoder = OutOfFoldTargetEncoder(alpha=1.0)
        encoded_df = encoder.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # Validation fold (B category) should not get encoding from full data
        val_encodings = encoded_df.iloc[10:]['category_SurvivalRate']
        
        # All validation encodings should be global mean (no info about B category)
        global_mean = y.mean()
        assert np.allclose(val_encodings, global_mean)
    
    def test_leakage_validation_method(self):
        """Test the validate_no_leakage method."""
        X = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        y = pd.Series([1, 0, 1, 1, 0, 1])
        
        cv_splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5]))
        ]
        
        encoder = OutOfFoldTargetEncoder(alpha=1.0)
        encoded_df = encoder.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # Validate no leakage
        is_valid = encoder.validate_no_leakage(X, y, encoded_df, ['category'], cv_splits)
        
        # Should pass validation (no leakage)
        assert is_valid == True
    
    def test_artificial_leakage_detection(self):
        """Test detection of artificially introduced leakage."""
        X = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        y = pd.Series([1, 0, 1, 1, 0, 1])
        
        cv_splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5]))
        ]
        
        # Create legitimate OOF encoding
        encoder = OutOfFoldTargetEncoder(alpha=1.0)
        encoded_df = encoder.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # Artificially introduce leakage by using full data encoding
        encoder_full = OutOfFoldTargetEncoder(alpha=1.0)
        encoder_full.fit_full_data(X, y, ['category'])
        leaked_df = encoder_full.transform(X, ['category'])
        
        # Copy leaked values to validation indices
        leaked_encoded = encoded_df.copy()
        leaked_encoded.loc[[4, 5], 'category_SurvivalRate'] = leaked_df.loc[[4, 5], 'category_SurvivalRate']
        
        # Validation should detect this leakage
        is_valid = encoder.validate_no_leakage(X, y, leaked_encoded, ['category'], cv_splits)
        
        # Should detect leakage
        assert is_valid == False
    
    def test_oof_encoding_with_rare_categories(self):
        """Test OOF encoding behavior with rare categories."""
        # Create data with rare category that appears only in validation
        X = pd.DataFrame({
            'category': ['A', 'A', 'A', 'A', 'B', 'RARE']  # RARE only in validation
        })
        y = pd.Series([1, 0, 1, 0, 1, 1])
        
        cv_splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5]))  # RARE in validation only
        ]
        
        encoder = OutOfFoldTargetEncoder(alpha=1.0)
        encoded_df = encoder.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # RARE category should get global mean (no training data)
        rare_encoding = encoded_df.loc[5, 'category_SurvivalRate']
        global_mean = y[:4].mean()  # Mean from training fold
        
        assert abs(rare_encoding - global_mean) < 1e-10
    
    def test_oof_encoding_properties(self):
        """Test mathematical properties of OOF encoding."""
        # Create balanced data
        X = pd.DataFrame({
            'category': ['A'] * 20 + ['B'] * 20
        })
        y = pd.Series([1, 0] * 20)  # Alternating pattern
        
        cv_splits = [
            (np.array(range(20)), np.array(range(20, 40))),
            (np.array(range(20, 40)), np.array(range(20)))
        ]
        
        encoder = OutOfFoldTargetEncoder(alpha=10.0)
        encoded_df = encoder.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # Check that encoding values are reasonable
        encodings = encoded_df['category_SurvivalRate']
        
        # Should be between 0 and 1
        assert encodings.min() >= 0
        assert encodings.max() <= 1
        
        # Should not be constant (some variation)
        assert encodings.std() > 0
    
    def test_smoothing_parameter_effect(self):
        """Test effect of smoothing parameter on encoding."""
        X = pd.DataFrame({
            'category': ['A'] * 4 + ['B'] * 4
        })
        y = pd.Series([1, 1, 0, 0, 0, 0, 1, 1])
        
        cv_splits = [
            (np.array([0, 1, 4, 5]), np.array([2, 3, 6, 7]))
        ]
        
        # High smoothing (more regularization)
        encoder_smooth = OutOfFoldTargetEncoder(alpha=100.0)
        encoded_smooth = encoder_smooth.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # Low smoothing (less regularization)
        encoder_raw = OutOfFoldTargetEncoder(alpha=0.1)
        encoded_raw = encoder_raw.fit_transform_oof(X, y, ['category'], cv_splits)
        
        # High smoothing should be closer to global mean
        global_mean = y.mean()
        smooth_encodings = encoded_smooth['category_SurvivalRate']
        raw_encodings = encoded_raw['category_SurvivalRate']
        
        # Check that smoothing pulls toward global mean
        smooth_diff = np.abs(smooth_encodings - global_mean).mean()
        raw_diff = np.abs(raw_encodings - global_mean).mean()
        
        # Smoothed should generally be closer to global mean
        assert smooth_diff <= raw_diff + 0.1  # Allow some tolerance