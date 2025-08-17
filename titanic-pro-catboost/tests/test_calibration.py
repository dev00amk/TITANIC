"""Tests for model calibration analysis."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from validation.calibration import calculate_calibration_metrics, fit_calibrator


class TestCalibrationAnalysis:
    """Test model calibration metrics and isotonic regression."""
    
    def test_perfect_calibration_metrics(self):
        """Test calibration metrics for perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        n_samples = 1000
        np.random.seed(42)
        
        # Predictions are actual probabilities
        y_prob = np.random.uniform(0, 1, n_samples)
        y_true = np.random.binomial(1, y_prob, n_samples)
        
        metrics = calculate_calibration_metrics(y_true, y_prob)
        
        # Perfect calibration should have low ECE
        assert metrics['ece'] < 0.1  # Should be quite low for large sample
        assert metrics['brier_score'] > 0  # Always positive
        assert metrics['n_samples'] == n_samples
    
    def test_overconfident_predictions(self):
        """Test calibration metrics for overconfident predictions."""
        # Create overconfident predictions
        n_samples = 100
        y_true = np.array([0, 1] * 50)  # 50% positive
        
        # Overconfident: push probabilities to extremes
        y_prob = np.where(y_true == 1, 0.9, 0.1)
        
        metrics = calculate_calibration_metrics(y_true, y_prob)
        
        # Should have higher ECE due to overconfidence
        assert metrics['ece'] > 0.0
        assert metrics['brier_score'] > 0.0
    
    def test_underconfident_predictions(self):
        """Test calibration metrics for underconfident predictions."""
        # Create underconfident predictions (all near 0.5)
        n_samples = 100
        y_true = np.array([0, 1] * 50)
        y_prob = np.full(n_samples, 0.5)  # All predictions at 50%
        
        metrics = calculate_calibration_metrics(y_true, y_prob)
        
        # Underconfident predictions
        assert metrics['ece'] >= 0.0
        assert metrics['brier_score'] == 0.25  # Should be exactly 0.25 for all 0.5 predictions
    
    def test_brier_score_calculation(self):
        """Test Brier score calculation."""
        # Known case
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        
        expected_brier = np.mean((y_prob - y_true) ** 2)
        
        metrics = calculate_calibration_metrics(y_true, y_prob)
        
        assert abs(metrics['brier_score'] - expected_brier) < 1e-10
    
    def test_ece_with_different_bin_counts(self):
        """Test ECE calculation with different number of bins."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        # Test with different bin counts
        metrics_5 = calculate_calibration_metrics(y_true, y_prob, n_bins=5)
        metrics_10 = calculate_calibration_metrics(y_true, y_prob, n_bins=10)
        
        # Both should be valid
        assert metrics_5['ece'] >= 0
        assert metrics_10['ece'] >= 0
        
        # With fewer bins, might get different ECE
        # (but both should be reasonable)
    
    def test_isotonic_calibrator_fitting(self):
        """Test isotonic regression calibrator."""
        # Create miscalibrated predictions
        np.random.seed(42)
        n_samples = 200
        
        # True probabilities
        true_prob = np.random.beta(2, 2, n_samples)  # Bell-shaped distribution
        y_true = np.random.binomial(1, true_prob)
        
        # Miscalibrated predictions (systematically overconfident)
        y_prob = np.clip(true_prob * 1.5, 0, 1)
        
        # Fit calibrator
        calibrator = fit_calibrator(y_true, y_prob)
        
        # Apply calibration
        y_calibrated = calibrator.predict(y_prob)
        
        # Calibrated predictions should be different from original
        assert not np.allclose(y_prob, y_calibrated)
        
        # Calibrated predictions should be in [0, 1]
        assert np.all(y_calibrated >= 0)
        assert np.all(y_calibrated <= 1)
    
    def test_calibrator_improves_calibration(self):
        """Test that calibrator improves calibration metrics."""
        # Create systematically miscalibrated data
        np.random.seed(42)
        n_samples = 500
        
        # Generate data where predictions are systematically too high
        y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate
        y_prob = np.random.beta(3, 2, n_samples)        # Predictions biased toward higher values
        
        # Pre-calibration metrics
        pre_metrics = calculate_calibration_metrics(y_true, y_prob)
        
        # Fit and apply calibrator
        calibrator = fit_calibrator(y_true, y_prob)
        y_calibrated = calibrator.predict(y_prob)
        
        # Post-calibration metrics
        post_metrics = calculate_calibration_metrics(y_true, y_calibrated)
        
        # Calibration should improve (lower ECE and/or Brier score)
        # Note: Isotonic regression should improve ECE, but Brier might not always improve
        assert post_metrics['ece'] <= pre_metrics['ece'] + 0.01  # Allow small tolerance
    
    def test_calibrator_monotonicity(self):
        """Test that calibrator preserves monotonicity."""
        # Create sample data
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.2, 0.8, 0.9, 0.05])
        
        calibrator = fit_calibrator(y_true, y_prob)
        
        # Test on sorted probabilities
        test_probs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        calibrated = calibrator.predict(test_probs)
        
        # Should be monotonically increasing
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]
    
    def test_edge_cases(self):
        """Test calibration with edge cases."""
        # All positive predictions
        y_true_all_pos = np.ones(10)
        y_prob_all_pos = np.random.uniform(0.5, 1.0, 10)
        
        metrics_pos = calculate_calibration_metrics(y_true_all_pos, y_prob_all_pos)
        assert metrics_pos['brier_score'] >= 0
        
        # All negative predictions
        y_true_all_neg = np.zeros(10)
        y_prob_all_neg = np.random.uniform(0.0, 0.5, 10)
        
        metrics_neg = calculate_calibration_metrics(y_true_all_neg, y_prob_all_neg)
        assert metrics_neg['brier_score'] >= 0
    
    def test_calibration_with_extreme_predictions(self):
        """Test calibration behavior with extreme predictions."""
        # Predictions at boundaries
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])  # Perfect extreme predictions
        
        metrics = calculate_calibration_metrics(y_true, y_prob)
        
        # Perfect predictions should have Brier score of 0
        assert metrics['brier_score'] == 0.0
        
        # ECE should also be 0 for perfect predictions
        assert metrics['ece'] == 0.0
    
    def test_calibration_stability(self):
        """Test calibration stability across different random seeds."""
        # Generate same data with different seeds
        base_true = np.array([0, 1, 0, 1, 1, 0])
        base_prob = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        
        # Multiple calibrator fits should give similar results
        calibrators = []
        for seed in [42, 43, 44]:
            np.random.seed(seed)
            # Add small noise to break ties differently
            noisy_prob = base_prob + np.random.normal(0, 0.01, len(base_prob))
            noisy_prob = np.clip(noisy_prob, 0, 1)
            
            calibrator = fit_calibrator(base_true, noisy_prob)
            calibrators.append(calibrator)
        
        # Test on same inputs
        test_prob = np.array([0.3, 0.7])
        results = [cal.predict(test_prob) for cal in calibrators]
        
        # Results should be similar (allowing for numerical differences)
        for i in range(len(results) - 1):
            assert np.allclose(results[i], results[i + 1], atol=0.1)