"""Model performance report and model card generation."""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelCardGenerator:
    """Generate comprehensive model performance reports and model cards."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.report_data = {}
        
    def load_artifacts(self, artifacts_dir: Path) -> Dict[str, Any]:
        """Load all model artifacts for analysis."""
        artifacts = {}
        
        # Load OOF predictions
        oof_path = artifacts_dir / "oof_predictions.npy"
        if oof_path.exists():
            artifacts["oof_predictions"] = np.load(oof_path)
        
        # Load feature importance
        importance_path = artifacts_dir / "feature_importance.csv"
        if importance_path.exists():
            artifacts["feature_importance"] = pd.read_csv(importance_path)
        
        # Load feature names
        features_path = artifacts_dir / "feature_names.csv"
        if features_path.exists():
            artifacts["feature_names"] = pd.read_csv(features_path)["feature"].tolist()
        
        # Load SHAP results if available
        shap_importance_path = artifacts_dir / "shap_feature_importance.csv"
        if shap_importance_path.exists():
            artifacts["shap_importance"] = pd.read_csv(shap_importance_path)
        
        # Load training data for ground truth
        train_path = Path(self.config.get("data", {}).get("train_path", "data/train.csv"))
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            artifacts["y_true"] = train_df["Survived"].values
        
        print(f"Loaded {len(artifacts)} artifact types")
        return artifacts
    
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    threshold: float = 0.5) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "specificity": recall_score(1 - y_true, 1 - y_pred),  # True negative rate
        }
        
        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "positive_predictive_value": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0,
        })
        
        return metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Find optimal classification threshold."""
        thresholds = np.arange(0.1, 0.91, 0.01)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:  # Avoid division by zero
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                
                threshold_metrics.append({
                    "threshold": threshold,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "balanced_accuracy": (recall + precision) / 2
                })
        
        threshold_df = pd.DataFrame(threshold_metrics)
        
        # Find optimal thresholds for different metrics
        optimal_thresholds = {
            "f1_optimal": threshold_df.loc[threshold_df["f1_score"].idxmax()],
            "balanced_optimal": threshold_df.loc[threshold_df["balanced_accuracy"].idxmax()],
            "precision_recall_balance": threshold_df.loc[
                abs(threshold_df["precision"] - threshold_df["recall"]).idxmin()
            ]
        }
        
        return {
            "threshold_analysis": threshold_df,
            "optimal_thresholds": optimal_thresholds
        }
    
    def create_performance_plots(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               output_dir: Path) -> None:
        """Create comprehensive performance visualization plots."""
        print("Creating performance plots...")
        
        # ROC Curve
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        # Prediction Distribution
        plt.subplot(2, 3, 3)
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Non-Survivors', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Survivors', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(2, 3, 4)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Survived', 'Survived'],
                   yticklabels=['Not Survived', 'Survived'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Threshold Analysis
        plt.subplot(2, 3, 5)
        threshold_analysis = self.find_optimal_threshold(y_true, y_pred_proba)
        threshold_df = threshold_analysis["threshold_analysis"]
        
        plt.plot(threshold_df["threshold"], threshold_df["f1_score"], label='F1 Score', linewidth=2)
        plt.plot(threshold_df["threshold"], threshold_df["precision"], label='Precision', linewidth=2)
        plt.plot(threshold_df["threshold"], threshold_df["recall"], label='Recall', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calibration Plot
        plt.subplot(2, 3, 6)
        from sklearn.calibration import calibration_curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
        except:
            plt.text(0.5, 0.5, 'Calibration plot\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Calibration Plot')
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {output_dir / 'performance_plots.png'}")
    
    def create_feature_importance_plots(self, feature_importance: pd.DataFrame,
                                      shap_importance: Optional[pd.DataFrame],
                                      output_dir: Path) -> None:
        """Create feature importance visualization plots."""
        print("Creating feature importance plots...")
        
        fig, axes = plt.subplots(1, 2 if shap_importance is not None else 1, 
                                figsize=(20, 8) if shap_importance is not None else (12, 8))
        
        if shap_importance is not None:
            # CatBoost Feature Importance
            axes[0].barh(range(min(20, len(feature_importance))), 
                        feature_importance.head(20)["importance"][::-1])
            axes[0].set_yticks(range(min(20, len(feature_importance))))
            axes[0].set_yticklabels(feature_importance.head(20)["feature"][::-1])
            axes[0].set_xlabel('CatBoost Feature Importance')
            axes[0].set_title('CatBoost Feature Importance (Top 20)')
            axes[0].grid(True, alpha=0.3)
            
            # SHAP Feature Importance
            axes[1].barh(range(min(20, len(shap_importance))), 
                        shap_importance.head(20)["mean_abs_shap"][::-1])
            axes[1].set_yticks(range(min(20, len(shap_importance))))
            axes[1].set_yticklabels(shap_importance.head(20)["feature"][::-1])
            axes[1].set_xlabel('Mean Absolute SHAP Value')
            axes[1].set_title('SHAP Feature Importance (Top 20)')
            axes[1].grid(True, alpha=0.3)
        else:
            # Only CatBoost Feature Importance
            if isinstance(axes, np.ndarray):
                ax = axes[0]
            else:
                ax = axes
                
            ax.barh(range(min(20, len(feature_importance))), 
                   feature_importance.head(20)["importance"][::-1])
            ax.set_yticks(range(min(20, len(feature_importance))))
            ax.set_yticklabels(feature_importance.head(20)["feature"][::-1])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Feature Importance (Top 20)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plots saved to {output_dir / 'feature_importance_plots.png'}")
    
    def generate_model_card_html(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Generate an HTML model card."""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Titanic Survival Prediction - Model Card</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; margin-top: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background-color: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
                .metric-label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
                .feature-list {{ columns: 2; column-gap: 30px; }}
                .feature-item {{ break-inside: avoid; margin-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .info {{ background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Titanic Survival Prediction Model Card</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Model Overview</h2>
                <div class="info">
                    <p><strong>Model Type:</strong> CatBoost Ensemble Classifier</p>
                    <p><strong>Task:</strong> Binary Classification (Survival Prediction)</p>
                    <p><strong>Dataset:</strong> Titanic Passenger Data</p>
                    <p><strong>Training Method:</strong> {report_data.get('training_info', {}).get('cv_folds', 'N/A')}-Fold Cross-Validation with {report_data.get('training_info', {}).get('n_seeds', 'N/A')} Seeds</p>
                </div>
                
                <h2>Performance Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{report_data.get('metrics', {}).get('roc_auc', 0):.3f}</div>
                        <div class="metric-label">ROC AUC</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report_data.get('metrics', {}).get('accuracy', 0):.3f}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report_data.get('metrics', {}).get('f1_score', 0):.3f}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report_data.get('metrics', {}).get('precision', 0):.3f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report_data.get('metrics', {}).get('recall', 0):.3f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{report_data.get('metrics', {}).get('specificity', 0):.3f}</div>
                        <div class="metric-label">Specificity</div>
                    </div>
                </div>
                
                <h2>Confusion Matrix</h2>
                <table>
                    <tr><th></th><th>Predicted: Not Survived</th><th>Predicted: Survived</th></tr>
                    <tr><th>Actual: Not Survived</th><td>{report_data.get('metrics', {}).get('true_negatives', 'N/A')}</td><td>{report_data.get('metrics', {}).get('false_positives', 'N/A')}</td></tr>
                    <tr><th>Actual: Survived</th><td>{report_data.get('metrics', {}).get('false_negatives', 'N/A')}</td><td>{report_data.get('metrics', {}).get('true_positives', 'N/A')}</td></tr>
                </table>
                
                <h2>Top Feature Importance</h2>
                <div class="feature-list">
        """
        
        # Add top features
        if "feature_importance" in report_data:
            top_features = report_data["feature_importance"].head(10)
            for _, row in top_features.iterrows():
                html_content += f'<div class="feature-item">{row["feature"]}: {row["importance"]:.4f}</div>'
        
        html_content += f"""
                </div>
                
                <h2>Model Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Iterations</td><td>{report_data.get('config', {}).get('catboost', {}).get('iterations', 'N/A')}</td></tr>
                    <tr><td>Learning Rate</td><td>{report_data.get('config', {}).get('catboost', {}).get('learning_rate', 'N/A')}</td></tr>
                    <tr><td>Depth</td><td>{report_data.get('config', {}).get('catboost', {}).get('depth', 'N/A')}</td></tr>
                    <tr><td>L2 Leaf Reg</td><td>{report_data.get('config', {}).get('catboost', {}).get('l2_leaf_reg', 'N/A')}</td></tr>
                </table>
                
                <h2>Data Information</h2>
                <div class="info">
                    <p><strong>Training Samples:</strong> {report_data.get('data_info', {}).get('n_samples', 'N/A')}</p>
                    <p><strong>Features:</strong> {report_data.get('data_info', {}).get('n_features', 'N/A')}</p>
                    <p><strong>Survival Rate:</strong> {report_data.get('data_info', {}).get('survival_rate', 'N/A'):.1%}</p>
                </div>
                
                <h2>Model Limitations and Considerations</h2>
                <div class="warning">
                    <ul>
                        <li>Model trained on historical Titanic data (1912) - may not generalize to modern scenarios</li>
                        <li>Performance metrics based on cross-validation - actual performance may vary</li>
                        <li>Feature engineering based on domain knowledge and may introduce bias</li>
                        <li>Model should be regularly retrained with new data if available</li>
                        <li>Predictions should be interpreted as probabilities, not certainties</li>
                    </ul>
                </div>
                
                <h2>Intended Use</h2>
                <div class="info">
                    <p>This model is intended for educational and research purposes to demonstrate machine learning techniques 
                    on the Titanic dataset. It should not be used for real-world life-or-death decisions.</p>
                </div>
                
                <h2>Ethical Considerations</h2>
                <div class="warning">
                    <p>The model may reflect historical biases present in the original data, including gender, class, 
                    and social status biases from 1912. These biases are historical artifacts and do not reflect 
                    modern values or appropriate decision-making criteria.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Model card HTML saved to {output_path}")
    
    def generate_comprehensive_report(self, artifacts_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive model performance report."""
        print("Generating comprehensive model performance report...")
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Load artifacts
        artifacts = self.load_artifacts(artifacts_dir)
        
        if "oof_predictions" not in artifacts or "y_true" not in artifacts:
            raise ValueError("Required artifacts (OOF predictions, ground truth) not found")
        
        y_true = artifacts["y_true"]
        y_pred_proba = artifacts["oof_predictions"]
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(y_true, y_pred_proba)
        threshold_analysis = self.find_optimal_threshold(y_true, y_pred_proba)
        
        # Create visualizations
        self.create_performance_plots(y_true, y_pred_proba, output_dir)
        
        if "feature_importance" in artifacts:
            self.create_feature_importance_plots(
                artifacts["feature_importance"],
                artifacts.get("shap_importance"),
                output_dir
            )
        
        # Compile report data
        report_data = {
            "metrics": metrics,
            "threshold_analysis": threshold_analysis,
            "feature_importance": artifacts.get("feature_importance"),
            "config": self.config,
            "data_info": {
                "n_samples": len(y_true),
                "n_features": len(artifacts.get("feature_names", [])),
                "survival_rate": y_true.mean(),
            },
            "training_info": {
                "cv_folds": self.config.get("cv", {}).get("n_splits", "N/A"),
                "n_seeds": self.config.get("cv", {}).get("n_seeds", "N/A"),
            }
        }
        
        # Save detailed report
        with open(output_dir / "model_report.json", 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_report = {}
            for key, value in report_data.items():
                if isinstance(value, dict):
                    serializable_report[key] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in value.items() if not isinstance(v, pd.DataFrame)
                    }
                elif isinstance(value, pd.DataFrame):
                    continue  # Skip DataFrames for JSON
                else:
                    serializable_report[key] = value
            
            json.dump(serializable_report, f, indent=2)
        
        # Generate model card
        self.generate_model_card_html(report_data, output_dir / "model_card.html")
        
        # Save summary report
        self.generate_summary_report(report_data, output_dir / "model_summary.txt")
        
        print(f"Comprehensive report generated in {output_dir}")
        return report_data
    
    def generate_summary_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Generate a text summary report."""
        with open(output_path, 'w') as f:
            f.write("TITANIC SURVIVAL PREDICTION MODEL REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance Summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            metrics = report_data["metrics"]
            f.write(f"ROC AUC Score:    {metrics['roc_auc']:.4f}\n")
            f.write(f"Accuracy:         {metrics['accuracy']:.4f}\n")
            f.write(f"Precision:        {metrics['precision']:.4f}\n")
            f.write(f"Recall:           {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:         {metrics['f1_score']:.4f}\n")
            f.write(f"Specificity:      {metrics['specificity']:.4f}\n\n")
            
            # Confusion Matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 16 + "\n")
            f.write(f"True Positives:   {metrics['true_positives']}\n")
            f.write(f"True Negatives:   {metrics['true_negatives']}\n")
            f.write(f"False Positives:  {metrics['false_positives']}\n")
            f.write(f"False Negatives:  {metrics['false_negatives']}\n\n")
            
            # Data Summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 12 + "\n")
            data_info = report_data["data_info"]
            f.write(f"Training Samples: {data_info['n_samples']}\n")
            f.write(f"Features:         {data_info['n_features']}\n")
            f.write(f"Survival Rate:    {data_info['survival_rate']:.1%}\n\n")
            
            # Top Features
            if report_data.get("feature_importance") is not None:
                f.write("TOP 10 FEATURES\n")
                f.write("-" * 15 + "\n")
                top_features = report_data["feature_importance"].head(10)
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    f.write(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}\n")
        
        print(f"Summary report saved to {output_path}")


def generate_model_report_from_artifacts(artifacts_dir: Path, config: Dict[str, Any], 
                                       output_dir: Path) -> Dict[str, Any]:
    """Generate model report from saved artifacts."""
    generator = ModelCardGenerator(config)
    return generator.generate_comprehensive_report(artifacts_dir, output_dir)


if __name__ == "__main__":
    # Example usage
    artifacts_dir = Path("artifacts")
    output_dir = Path("model_reports")
    config = {
        "cv": {"n_splits": 5, "n_seeds": 3},
        "catboost": {"iterations": 6000, "learning_rate": 0.035, "depth": 6, "l2_leaf_reg": 8},
        "data": {"train_path": "data/train.csv"}
    }
    
    try:
        report = generate_model_report_from_artifacts(artifacts_dir, config, output_dir)
        print("Model report generation completed successfully!")
    except Exception as e:
        print(f"Model report generation failed: {e}")