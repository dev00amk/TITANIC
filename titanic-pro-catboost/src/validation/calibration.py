"""Model calibration analysis and isotonic regression calibrator."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import pickle
import warnings

sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Calculate calibration metrics: Brier score and Expected Calibration Error (ECE)."""
    
    # Brier Score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Reliability (for additional insight)
    reliability = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            reliability += (avg_confidence_in_bin - accuracy_in_bin) ** 2 * prop_in_bin
    
    return {
        'brier_score': brier_score,
        'ece': ece,
        'reliability': reliability,
        'n_samples': len(y_true)
    }


def fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> Any:
    """Fit isotonic regression calibrator."""
    from sklearn.isotonic import IsotonicRegression
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_prob, y_true)
    
    return calibrator


def run_calibration_analysis(config: Dict[str, Any],
                           artifacts_dir: Path,
                           output_dir: Path = None) -> Dict[str, Any]:
    """Run complete calibration analysis on OOF predictions."""
    print("Running calibration analysis...")
    
    # Load OOF predictions and ground truth
    oof_path = artifacts_dir / "oof_predictions.npy"
    if not oof_path.exists():
        print("⚠️  OOF predictions not found, running training first...")
        # Run training to generate OOF predictions
        from modeling.train_cv import TrainingPipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
    
    # Load OOF predictions
    oof_predictions = np.load(oof_path)
    
    # Load training data to get ground truth
    train_path = Path(config["data"]["train_path"])
    train_df = pd.read_csv(train_path)
    y_true = train_df["Survived"].values
    
    # Ensure alignment
    if len(oof_predictions) != len(y_true):
        print(f"⚠️  Length mismatch: OOF {len(oof_predictions)} vs truth {len(y_true)}")
        min_len = min(len(oof_predictions), len(y_true))
        oof_predictions = oof_predictions[:min_len]
        y_true = y_true[:min_len]
    
    # Calculate pre-calibration metrics
    pre_metrics = calculate_calibration_metrics(y_true, oof_predictions)
    print(f"Pre-calibration: Brier {pre_metrics['brier_score']:.4f}, ECE {pre_metrics['ece']:.4f}")
    
    # Fit calibrator
    calibrator = fit_calibrator(y_true, oof_predictions)
    
    # Apply calibration
    calibrated_predictions = calibrator.predict(oof_predictions)
    
    # Calculate post-calibration metrics
    post_metrics = calculate_calibration_metrics(y_true, calibrated_predictions)
    print(f"Post-calibration: Brier {post_metrics['brier_score']:.4f}, ECE {post_metrics['ece']:.4f}")
    
    # Calibration improvement
    brier_improvement = pre_metrics['brier_score'] - post_metrics['brier_score']
    ece_improvement = pre_metrics['ece'] - post_metrics['ece']
    
    results = {
        'pre_calibration': pre_metrics,
        'post_calibration': post_metrics,
        'brier_improvement': brier_improvement,
        'ece_improvement': ece_improvement,
        'calibration_improved': (brier_improvement > 0) and (ece_improvement > 0)
    }
    
    print(f"Improvement: Brier Δ{brier_improvement:+.4f}, ECE Δ{ece_improvement:+.4f}")
    
    if results['calibration_improved']:
        print("✅ Calibration improved model reliability")
    else:
        print("⚠️  Calibration did not improve (model may already be well-calibrated)")
    
    # Save calibrator
    calibrator_path = artifacts_dir / "calibrator.pkl"
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {calibrator_path}")
    
    # Save calibration results
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        
        # Save pre-calibration metrics
        import json
        with open(output_dir / "calibration_pre.json", 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in pre_metrics.items()}, f, indent=2)
        
        # Save post-calibration metrics
        with open(output_dir / "calibration_post.json", 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in post_metrics.items()}, f, indent=2)
        
        print(f"Calibration metrics saved to {output_dir}")
    
    return results


def apply_calibration_to_test(test_predictions: np.ndarray, 
                             calibrator_path: Path) -> np.ndarray:
    """Apply saved calibrator to test predictions."""
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)
    
    return calibrator.predict(test_predictions)


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
    output_dir = Path(__file__).parent.parent.parent / "reports"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    results = run_calibration_analysis(config, artifacts_dir, output_dir)
    print(f"Calibration analysis complete. Improved: {results['calibration_improved']}")