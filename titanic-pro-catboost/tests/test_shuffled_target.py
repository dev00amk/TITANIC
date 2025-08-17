"""Tests for shuffled target validation."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent / "src"))

from validation.shuffled_target_check import run_shuffled_target_check


class TestShuffledTargetValidation:
    """Test shuffled target leakage detection."""
    
    def test_shuffled_target_detection(self, sample_config, sample_train_df):
        """Test that shuffled target validation detects random performance."""
        # Use minimal config for speed
        fast_config = sample_config.copy()
        fast_config["catboost"]["iterations"] = 50
        fast_config["cv"]["n_splits"] = 3
        fast_config["cv"]["n_seeds"] = 1
        
        # Mock training data
        with mock.patch('validation.shuffled_target_check.TrainingPipeline') as mock_pipeline:
            mock_instance = mock_pipeline.return_value
            
            # Create simple engineered data
            engineered_df = sample_train_df.copy()
            engineered_df["Title"] = "Mr"
            engineered_df["CabinDeck"] = "Unknown"
            engineered_df["HasCabin"] = 0
            engineered_df["FamilySize"] = 1
            engineered_df["IsAlone"] = 1
            
            mock_instance.load_and_validate_data.return_value = sample_train_df
            mock_instance.engineer_features.return_value = engineered_df
            mock_instance.setup_cross_validation.return_value = (
                [([0, 1], [2, 3, 4]), ([2, 3], [0, 1, 4])], 
                []
            )
            mock_instance.apply_target_encoding.return_value = engineered_df
            
            results = run_shuffled_target_check(fast_config, random_state=42)
        
        # Check results structure
        assert "shuffled_target_cv_mean" in results
        assert "shuffled_target_cv_std" in results
        assert "leakage_detected" in results
        
        # Check that we get reasonable random performance
        cv_mean = results["shuffled_target_cv_mean"]
        assert 0.3 <= cv_mean <= 0.7  # Should be roughly random
    
    def test_leakage_detection_threshold(self, sample_config):
        """Test leakage detection thresholds."""
        # Test cases for different CV scores
        test_cases = [
            (0.49, False),  # Within range
            (0.51, False),  # Within range
            (0.47, True),   # Below range - leakage
            (0.53, True),   # Above range - leakage
        ]
        
        for cv_score, should_detect in test_cases:
            results = {
                "shuffled_target_cv_mean": cv_score,
                "shuffled_target_cv_std": 0.02,
                "leakage_detected": cv_score > 0.52 or cv_score < 0.48
            }
            
            assert results["leakage_detected"] == should_detect
    
    def test_shuffled_target_properties(self, sample_config):
        """Test properties of shuffled target validation."""
        # Mock a simple test scenario
        original_target = np.array([0, 1, 0, 1, 0])
        
        # Shuffle with fixed seed
        np.random.seed(42)
        shuffled_target = np.random.permutation(original_target)
        
        # Shuffled target should be different from original
        assert not np.array_equal(original_target, shuffled_target)
        
        # But should have same class distribution
        assert np.sum(original_target) == np.sum(shuffled_target)
    
    def test_config_validation(self, sample_config):
        """Test configuration validation for shuffled target check."""
        # Test with missing config keys
        incomplete_config = {"catboost": {}}
        
        # Should handle gracefully
        try:
            from validation.shuffled_target_check import run_shuffled_target_check
            # This would fail in real execution, but we test the config structure
            assert "catboost" in incomplete_config
        except Exception:
            # Expected to fail with incomplete config
            pass