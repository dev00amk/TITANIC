"""Slice-based performance analysis by demographic groups."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from validation.calibration import calculate_calibration_metrics

warnings.filterwarnings('ignore')


def calculate_slice_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                          slice_mask: np.ndarray, slice_name: str) -> Dict[str, Any]:
    """Calculate performance metrics for a specific slice."""
    if not np.any(slice_mask):
        return {
            'slice': slice_name,
            'n_samples': 0,
            'accuracy': np.nan,
            'auc': np.nan,
            'brier': np.nan,
            'ece': np.nan
        }
    
    slice_y_true = y_true[slice_mask]
    slice_y_prob = y_prob[slice_mask]
    slice_y_pred = (slice_y_prob >= 0.5).astype(int)
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    accuracy = accuracy_score(slice_y_true, slice_y_pred)
    
    # AUC only if both classes present
    if len(np.unique(slice_y_true)) > 1:
        auc = roc_auc_score(slice_y_true, slice_y_prob)
    else:
        auc = np.nan
    
    # Calibration metrics
    cal_metrics = calculate_calibration_metrics(slice_y_true, slice_y_prob)
    
    return {
        'slice': slice_name,
        'n_samples': len(slice_y_true),
        'accuracy': accuracy,
        'auc': auc,
        'brier': cal_metrics['brier_score'],
        'ece': cal_metrics['ece'],
        'positive_rate': slice_y_true.mean()
    }


def run_slice_analysis(config: Dict[str, Any],
                      artifacts_dir: Path,
                      output_path: Path = None) -> pd.DataFrame:
    """Run comprehensive slice analysis by demographic groups."""
    print("Running slice-based performance analysis...")
    
    # Load OOF predictions
    oof_path = artifacts_dir / "oof_predictions.npy"
    if not oof_path.exists():
        print("⚠️  OOF predictions not found, running training first...")
        from modeling.train_cv import TrainingPipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
    
    oof_predictions = np.load(oof_path)
    
    # Load and engineer data to get slice variables
    from modeling.train_cv import TrainingPipeline
    pipeline = TrainingPipeline(config)
    df = pipeline.load_and_validate_data()
    df = pipeline.engineer_features(df)
    
    y_true = df["Survived"].values
    
    # Ensure alignment
    if len(oof_predictions) != len(y_true):
        min_len = min(len(oof_predictions), len(y_true))
        oof_predictions = oof_predictions[:min_len]
        y_true = y_true[:min_len]
        df = df.iloc[:min_len]
    
    # Calculate global metrics for comparison
    global_metrics = calculate_slice_metrics(y_true, oof_predictions, 
                                           np.ones(len(y_true), dtype=bool), "Global")
    
    # Define slices
    slice_results = []
    
    # Gender slices
    if "Sex" in df.columns:
        for gender in df["Sex"].unique():
            if pd.notna(gender):
                mask = df["Sex"] == gender
                metrics = calculate_slice_metrics(y_true, oof_predictions, mask, f"Sex_{gender}")
                slice_results.append(metrics)
    
    # Class slices  
    if "Pclass" in df.columns:
        for pclass in sorted(df["Pclass"].unique()):
            if pd.notna(pclass):
                mask = df["Pclass"] == pclass
                metrics = calculate_slice_metrics(y_true, oof_predictions, mask, f"Pclass_{pclass}")
                slice_results.append(metrics)
    
    # Age group slices
    if "AgeGroup" in df.columns:
        for age_group in df["AgeGroup"].unique():
            if pd.notna(age_group):
                mask = df["AgeGroup"] == age_group
                metrics = calculate_slice_metrics(y_true, oof_predictions, mask, f"AgeGroup_{age_group}")
                slice_results.append(metrics)
    
    # Title slices (for most common titles)
    if "Title" in df.columns:
        common_titles = df["Title"].value_counts().head(5).index
        for title in common_titles:
            mask = df["Title"] == title
            metrics = calculate_slice_metrics(y_true, oof_predictions, mask, f"Title_{title}")
            slice_results.append(metrics)
    
    # Embarked slices
    if "Embarked" in df.columns:
        for embarked in df["Embarked"].unique():
            if pd.notna(embarked):
                mask = df["Embarked"] == embarked
                metrics = calculate_slice_metrics(y_true, oof_predictions, mask, f"Embarked_{embarked}")
                slice_results.append(metrics)
    
    # Family size slices
    if "FamilySize" in df.columns:
        # Group family sizes for better statistics
        df["FamilySizeGroup"] = df["FamilySize"].apply(
            lambda x: "Alone" if x == 1 else ("Small" if x <= 3 else ("Medium" if x <= 6 else "Large"))
        )
        for size_group in df["FamilySizeGroup"].unique():
            mask = df["FamilySizeGroup"] == size_group
            metrics = calculate_slice_metrics(y_true, oof_predictions, mask, f"FamilySize_{size_group}")
            slice_results.append(metrics)
    
    # Add global metrics
    slice_results.insert(0, global_metrics)
    
    # Convert to DataFrame
    slice_df = pd.DataFrame(slice_results)
    
    # Calculate performance gaps from global
    global_accuracy = global_metrics['accuracy']
    global_ece = global_metrics['ece']
    
    slice_df['accuracy_gap'] = slice_df['accuracy'] - global_accuracy
    slice_df['ece_gap'] = slice_df['ece'] - global_ece
    
    # Flag concerning slices
    slice_df['concerning_accuracy'] = slice_df['accuracy_gap'] < -0.05
    slice_df['concerning_ece'] = slice_df['ece_gap'] > 0.05
    
    # Sort by accuracy gap (worst first)
    slice_df = slice_df.sort_values('accuracy_gap')
    
    # Print results
    print(f"\nSlice Analysis Results (vs Global Accuracy: {global_accuracy:.4f}):")
    print("=" * 80)
    
    concerning_slices = slice_df[slice_df['concerning_accuracy'] | slice_df['concerning_ece']]
    if len(concerning_slices) > 0:
        print("⚠️  Concerning performance gaps detected:")
        for _, row in concerning_slices.iterrows():
            print(f"  {row['slice']}: Acc {row['accuracy']:.3f} (Δ{row['accuracy_gap']:+.3f}), "
                  f"ECE {row['ece']:.3f} (Δ{row['ece_gap']:+.3f})")
    else:
        print("✅ No significant performance gaps detected")
    
    print(f"\nTop 5 slices by accuracy:")
    top_slices = slice_df.nlargest(5, 'accuracy')[['slice', 'n_samples', 'accuracy', 'ece']]
    print(top_slices.to_string(index=False, float_format='%.3f'))
    
    print(f"\nWorst 5 slices by accuracy:")
    worst_slices = slice_df.nsmallest(5, 'accuracy')[['slice', 'n_samples', 'accuracy', 'ece']]
    print(worst_slices.to_string(index=False, float_format='%.3f'))
    
    # Save results
    if output_path:
        output_path.parent.mkdir(exist_ok=True)
        slice_df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\nSlice metrics saved to {output_path}")
    
    return slice_df


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
    output_path = Path(__file__).parent.parent.parent / "reports" / "slice_metrics.csv"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    slice_df = run_slice_analysis(config, artifacts_dir, output_path)
    
    min_acc = slice_df[slice_df['slice'] != 'Global']['accuracy'].min()
    global_acc = slice_df[slice_df['slice'] == 'Global']['accuracy'].iloc[0]
    print(f"\nSummary: Min slice accuracy: {min_acc:.4f}, Global: {global_acc:.4f}, Gap: {min_acc - global_acc:+.4f}")