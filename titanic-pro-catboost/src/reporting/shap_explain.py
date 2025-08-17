"""SHAP explainability analysis for model interpretability."""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP analysis for CatBoost model interpretability."""
    
    def __init__(self, models: List[Any], feature_names: List[str], config: Dict[str, Any]):
        self.models = models
        self.feature_names = feature_names
        self.config = config
        self.explainers = []
        self.shap_values_list = []
        
    def create_explainers(self, X_background: np.ndarray, max_background_size: int = 100) -> None:
        """Create SHAP explainers for each model."""
        print("Creating SHAP explainers...")
        
        # Sample background data if too large
        if len(X_background) > max_background_size:
            background_indices = np.random.choice(
                len(X_background), max_background_size, replace=False
            )
            X_background_sample = X_background[background_indices]
        else:
            X_background_sample = X_background
        
        self.explainers = []
        for i, model in enumerate(self.models):
            try:
                # Use TreeExplainer for CatBoost models
                explainer = shap.TreeExplainer(model)
                self.explainers.append(explainer)
                print(f"Created explainer for model {i+1}/{len(self.models)}")
            except Exception as e:
                print(f"Warning: Could not create explainer for model {i}: {e}")
                continue
        
        print(f"Successfully created {len(self.explainers)} SHAP explainers")
    
    def calculate_shap_values(self, X_explain: np.ndarray, max_samples: int = 500) -> None:
        """Calculate SHAP values for explanation dataset."""
        print("Calculating SHAP values...")
        
        # Sample explanation data if too large
        if len(X_explain) > max_samples:
            explain_indices = np.random.choice(
                len(X_explain), max_samples, replace=False
            )
            X_explain_sample = X_explain[explain_indices]
        else:
            X_explain_sample = X_explain
        
        self.shap_values_list = []
        for i, explainer in enumerate(self.explainers):
            try:
                shap_values = explainer.shap_values(X_explain_sample)
                
                # For binary classification, CatBoost returns SHAP values for positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
                
                self.shap_values_list.append(shap_values)
                print(f"Calculated SHAP values for model {i+1}/{len(self.explainers)}")
            except Exception as e:
                print(f"Warning: Could not calculate SHAP values for model {i}: {e}")
                continue
        
        print(f"Successfully calculated SHAP values for {len(self.shap_values_list)} models")
    
    def aggregate_shap_values(self) -> np.ndarray:
        """Aggregate SHAP values across all models."""
        if not self.shap_values_list:
            raise ValueError("No SHAP values calculated")
        
        # Average SHAP values across models
        aggregated_shap = np.mean(self.shap_values_list, axis=0)
        print(f"Aggregated SHAP values shape: {aggregated_shap.shape}")
        return aggregated_shap
    
    def create_global_importance_plot(self, shap_values: np.ndarray, X_data: np.ndarray, 
                                    output_path: Path, top_n: int = 20) -> None:
        """Create global feature importance plot."""
        plt.figure(figsize=(12, 8))
        
        # Create summary plot
        shap.summary_plot(
            shap_values, 
            X_data, 
            feature_names=self.feature_names,
            max_display=top_n,
            show=False
        )
        
        plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "shap_global_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Global importance plot saved to {output_path / 'shap_global_importance.png'}")
    
    def create_detailed_summary_plot(self, shap_values: np.ndarray, X_data: np.ndarray,
                                   output_path: Path, top_n: int = 20) -> None:
        """Create detailed SHAP summary plot with feature values."""
        plt.figure(figsize=(12, 10))
        
        shap.summary_plot(
            shap_values, 
            X_data,
            feature_names=self.feature_names,
            plot_type="dot",
            max_display=top_n,
            show=False
        )
        
        plt.title("SHAP Summary Plot (Feature Values)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "shap_detailed_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed summary plot saved to {output_path / 'shap_detailed_summary.png'}")
    
    def create_waterfall_plots(self, shap_values: np.ndarray, X_data: np.ndarray,
                              output_path: Path, n_examples: int = 5) -> None:
        """Create waterfall plots for individual predictions."""
        print("Creating waterfall plots for individual examples...")
        
        # Select diverse examples
        indices = np.linspace(0, len(X_data) - 1, n_examples, dtype=int)
        
        with PdfPages(output_path / "shap_waterfall_plots.pdf") as pdf:
            for i, idx in enumerate(indices):
                plt.figure(figsize=(12, 8))
                
                # Create waterfall plot
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[idx],
                        base_values=np.mean(shap_values),
                        data=X_data[idx],
                        feature_names=self.feature_names
                    ),
                    max_display=15,
                    show=False
                )
                
                plt.title(f"SHAP Waterfall Plot - Example {i+1}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save to PDF
                pdf.savefig(bbox_inches='tight', dpi=300)
                plt.close()
        
        print(f"Waterfall plots saved to {output_path / 'shap_waterfall_plots.pdf'}")
    
    def create_dependence_plots(self, shap_values: np.ndarray, X_data: np.ndarray,
                               output_path: Path, top_features: int = 10) -> None:
        """Create partial dependence plots for top features."""
        print("Creating SHAP dependence plots...")
        
        # Get top features by mean absolute SHAP value
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_feature_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
        
        with PdfPages(output_path / "shap_dependence_plots.pdf") as pdf:
            for i, feature_idx in enumerate(top_feature_indices):
                plt.figure(figsize=(10, 6))
                
                # Find best interaction feature
                interaction_idx = None
                if len(self.feature_names) > 1:
                    # Calculate correlation with other features' SHAP values
                    correlations = []
                    for j in range(len(self.feature_names)):
                        if j != feature_idx:
                            corr = np.corrcoef(shap_values[:, feature_idx], shap_values[:, j])[0, 1]
                            correlations.append((abs(corr), j))
                    
                    if correlations:
                        correlations.sort(reverse=True)
                        interaction_idx = correlations[0][1]
                
                # Create dependence plot
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    X_data,
                    feature_names=self.feature_names,
                    interaction_index=interaction_idx,
                    show=False
                )
                
                plt.title(f"SHAP Dependence Plot - {self.feature_names[feature_idx]}", 
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save to PDF
                pdf.savefig(bbox_inches='tight', dpi=300)
                plt.close()
        
        print(f"Dependence plots saved to {output_path / 'shap_dependence_plots.pdf'}")
    
    def create_force_plots(self, shap_values: np.ndarray, X_data: np.ndarray,
                          output_path: Path, n_examples: int = 5) -> None:
        """Create force plots for individual predictions."""
        print("Creating SHAP force plots...")
        
        # Select examples with different prediction strengths
        base_value = np.mean(shap_values.sum(axis=1))
        prediction_strengths = np.abs(shap_values.sum(axis=1) - base_value)
        
        # Get examples with high, medium, and low prediction strength
        sorted_indices = np.argsort(prediction_strengths)
        selected_indices = [
            sorted_indices[0],  # Lowest
            sorted_indices[len(sorted_indices)//4],  # Low-medium
            sorted_indices[len(sorted_indices)//2],  # Medium
            sorted_indices[3*len(sorted_indices)//4],  # High-medium
            sorted_indices[-1]  # Highest
        ][:n_examples]
        
        for i, idx in enumerate(selected_indices):
            try:
                # Create force plot
                shap_plot = shap.force_plot(
                    base_value,
                    shap_values[idx],
                    X_data[idx],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                
                plt.title(f"SHAP Force Plot - Example {i+1}", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_path / f"shap_force_plot_example_{i+1}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not create force plot for example {i+1}: {e}")
                continue
        
        print(f"Force plots saved to {output_path}")
    
    def calculate_feature_interactions(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Calculate and analyze feature interactions."""
        print("Analyzing feature interactions...")
        
        interaction_data = []
        n_features = len(self.feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Calculate correlation between SHAP values
                corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                
                # Calculate interaction strength (sum of absolute cross-products)
                interaction_strength = np.mean(np.abs(shap_values[:, i] * shap_values[:, j]))
                
                interaction_data.append({
                    "feature_1": self.feature_names[i],
                    "feature_2": self.feature_names[j],
                    "shap_correlation": corr,
                    "interaction_strength": interaction_strength
                })
        
        interactions_df = pd.DataFrame(interaction_data)
        interactions_df = interactions_df.sort_values("interaction_strength", ascending=False)
        
        return interactions_df
    
    def generate_shap_report(self, shap_values: np.ndarray, X_data: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive SHAP analysis report."""
        print("Generating SHAP analysis report...")
        
        # Feature importance metrics
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        mean_shap = np.mean(shap_values, axis=0)
        std_shap = np.std(shap_values, axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "std_shap": std_shap,
            "importance_rank": range(1, len(self.feature_names) + 1)
        })
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
        importance_df["importance_rank"] = range(1, len(importance_df) + 1)
        
        # Calculate feature interactions
        interactions_df = self.calculate_feature_interactions(shap_values)
        
        # Summary statistics
        summary_stats = {
            "total_samples": len(X_data),
            "total_features": len(self.feature_names),
            "mean_prediction_impact": np.mean(np.abs(shap_values.sum(axis=1))),
            "max_prediction_impact": np.max(np.abs(shap_values.sum(axis=1))),
            "top_5_features": importance_df.head(5)["feature"].tolist(),
            "most_positive_feature": importance_df.loc[importance_df["mean_shap"].idxmax(), "feature"],
            "most_negative_feature": importance_df.loc[importance_df["mean_shap"].idxmin(), "feature"]
        }
        
        return {
            "feature_importance": importance_df,
            "feature_interactions": interactions_df,
            "summary_stats": summary_stats,
            "shap_values": shap_values
        }
    
    def save_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Save SHAP analysis report to files."""
        print("Saving SHAP analysis report...")
        
        # Save feature importance
        report["feature_importance"].to_csv(
            output_path / "shap_feature_importance.csv", index=False
        )
        
        # Save feature interactions
        report["feature_interactions"].to_csv(
            output_path / "shap_feature_interactions.csv", index=False
        )
        
        # Save SHAP values
        np.save(output_path / "shap_values.npy", report["shap_values"])
        
        # Save summary report
        with open(output_path / "shap_summary_report.txt", 'w') as f:
            stats = report["summary_stats"]
            f.write("SHAP Explainability Analysis Report\n")
            f.write("===================================\n\n")
            f.write(f"Dataset Summary:\n")
            f.write(f"  Total samples analyzed: {stats['total_samples']}\n")
            f.write(f"  Total features: {stats['total_features']}\n\n")
            f.write(f"Prediction Impact:\n")
            f.write(f"  Mean absolute prediction impact: {stats['mean_prediction_impact']:.4f}\n")
            f.write(f"  Maximum absolute prediction impact: {stats['max_prediction_impact']:.4f}\n\n")
            f.write(f"Top 5 Most Important Features:\n")
            for i, feature in enumerate(stats['top_5_features'], 1):
                f.write(f"  {i}. {feature}\n")
            f.write(f"\nMost positive impact feature: {stats['most_positive_feature']}\n")
            f.write(f"Most negative impact feature: {stats['most_negative_feature']}\n")
        
        print(f"SHAP report saved to {output_path}")
    
    def run_complete_analysis(self, X_data: np.ndarray, output_dir: Path) -> Dict[str, Any]:
        """Run complete SHAP explainability analysis."""
        print("Starting comprehensive SHAP analysis...")
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Create explainers
        self.create_explainers(X_data)
        
        if not self.explainers:
            raise ValueError("No SHAP explainers could be created")
        
        # Calculate SHAP values
        self.calculate_shap_values(X_data)
        
        if not self.shap_values_list:
            raise ValueError("No SHAP values could be calculated")
        
        # Aggregate SHAP values
        aggregated_shap = self.aggregate_shap_values()
        
        # Generate visualizations
        print("Creating SHAP visualizations...")
        self.create_global_importance_plot(aggregated_shap, X_data, output_dir)
        self.create_detailed_summary_plot(aggregated_shap, X_data, output_dir)
        self.create_waterfall_plots(aggregated_shap, X_data, output_dir)
        self.create_dependence_plots(aggregated_shap, X_data, output_dir)
        self.create_force_plots(aggregated_shap, X_data, output_dir)
        
        # Generate comprehensive report
        report = self.generate_shap_report(aggregated_shap, X_data)
        
        # Save report
        self.save_report(report, output_dir)
        
        print("SHAP analysis completed successfully!")
        return report


def run_shap_analysis_from_artifacts(artifacts_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run SHAP analysis using saved model artifacts."""
    import pickle
    
    # Load models
    models_path = artifacts_dir / "models.pkl"
    if not models_path.exists():
        raise FileNotFoundError(f"Models not found at {models_path}")
    
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    # Load feature names
    feature_names_path = artifacts_dir / "feature_names.csv"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    feature_names_df = pd.read_csv(feature_names_path)
    feature_names = feature_names_df["feature"].tolist()
    
    # Load training data for SHAP analysis
    # This would need to be modified based on your data paths
    config = {"data": {"train_path": "data/train.csv"}}
    
    # Create SHAP explainer
    explainer = SHAPExplainer(models, feature_names, config)
    
    # Load data for explanation (you might want to load from artifacts)
    # For now, using a placeholder
    X_data = np.random.randn(100, len(feature_names))  # Replace with actual data
    
    # Run analysis
    report = explainer.run_complete_analysis(X_data, output_dir)
    
    return report


if __name__ == "__main__":
    # Example usage
    artifacts_dir = Path("artifacts")
    output_dir = Path("shap_analysis")
    
    try:
        report = run_shap_analysis_from_artifacts(artifacts_dir, output_dir)
        print("SHAP analysis completed successfully!")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")