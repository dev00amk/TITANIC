"""Feature ablation analysis for understanding feature importance."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.utils import set_seed
from modeling.train_cv import TrainingPipeline


class FeatureAblationAnalyzer:
    """Analyze feature importance through systematic ablation studies."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.baseline_score = None
        self.ablation_results = {}
        
    def run_baseline_training(self) -> float:
        """Run baseline training with all features to get reference score."""
        print("Running baseline training with all features...")
        
        pipeline = TrainingPipeline(self.config)
        metrics = pipeline.run()
        
        baseline_score = metrics.get("ensemble_auc", 0.0)
        self.baseline_score = baseline_score
        
        print(f"Baseline AUC: {baseline_score:.5f}")
        return baseline_score
    
    def get_feature_groups(self, all_features: List[str]) -> Dict[str, List[str]]:
        """Define feature groups for ablation analysis."""
        feature_groups = {
            "basic_demographics": ["Sex", "Age", "Pclass"],
            "family_features": [f for f in all_features if any(x in f for x in ["Family", "SibSp", "Parch", "IsAlone", "HasSiblings", "HasParents", "HasChildren"])],
            "title_features": [f for f in all_features if "Title" in f],
            "cabin_features": [f for f in all_features if any(x in f for x in ["Cabin", "Deck", "HasCabin"])],
            "ticket_features": [f for f in all_features if any(x in f for x in ["Ticket", "TicketPrefix", "TicketHasLetters"])],
            "embarked_features": [f for f in all_features if "Embarked" in f],
            "age_features": [f for f in all_features if any(x in f for x in ["Age", "IsChild"]) and f != "Age"],
            "survival_rate_features": [f for f in all_features if "_SurvivalRate" in f],
            "interaction_features": [f for f in all_features if "_" in f and any(x in f for x in ["Sex_", "Title_", "AgeGroup_", "FamilySize_", "Embarked_"])],
            "fare_features": [f for f in all_features if "Fare" in f],
        }
        
        # Remove empty groups
        feature_groups = {k: v for k, v in feature_groups.items() if v}
        
        # Add individual high-importance features
        high_importance_individual = ["Sex", "Title", "Fare", "Age", "Pclass"]
        for feature in high_importance_individual:
            if feature in all_features:
                feature_groups[f"individual_{feature}"] = [feature]
        
        return feature_groups
    
    def run_ablation_experiment(self, features_to_remove: List[str], experiment_name: str) -> Dict[str, float]:
        """Run training experiment with specified features removed."""
        print(f"Running ablation experiment: {experiment_name}")
        print(f"Removing features: {features_to_remove}")
        
        # Create modified config with features to exclude
        modified_config = self.config.copy()
        
        # Set feature exclusion in config (this would need to be handled in the feature engineering)
        if "ablation" not in modified_config:
            modified_config["ablation"] = {}
        modified_config["ablation"]["exclude_features"] = features_to_remove
        
        try:
            # Run training pipeline with modified config
            pipeline = TrainingPipeline(modified_config)
            metrics = pipeline.run()
            
            experiment_score = metrics.get("ensemble_auc", 0.0)
            score_drop = self.baseline_score - experiment_score
            relative_drop = (score_drop / self.baseline_score) * 100 if self.baseline_score > 0 else 0
            
            result = {
                "auc": experiment_score,
                "score_drop": score_drop,
                "relative_drop_pct": relative_drop,
                "accuracy": metrics.get("ensemble_accuracy", 0.0),
                "f1": metrics.get("ensemble_f1", 0.0),
                "features_removed": len(features_to_remove)
            }
            
            print(f"  AUC: {experiment_score:.5f} (drop: {score_drop:.5f}, {relative_drop:.2f}%)")
            return result
            
        except Exception as e:
            print(f"  Experiment failed: {e}")
            return {
                "auc": 0.0,
                "score_drop": float('inf'),
                "relative_drop_pct": float('inf'),
                "accuracy": 0.0,
                "f1": 0.0,
                "features_removed": len(features_to_remove),
                "error": str(e)
            }
    
    def run_progressive_ablation(self, feature_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Run progressive ablation removing one group at a time."""
        print("\nRunning progressive ablation analysis...")
        
        results = {}
        
        # Sort groups by importance (start with least important based on naming)
        group_priority = [
            "ticket_features", "embarked_features", "cabin_features", 
            "interaction_features", "age_features", "fare_features",
            "survival_rate_features", "family_features", "title_features", 
            "basic_demographics"
        ]
        
        # Add individual features
        individual_groups = [k for k in feature_groups.keys() if k.startswith("individual_")]
        group_priority.extend(individual_groups)
        
        for group_name in group_priority:
            if group_name in feature_groups:
                features_to_remove = feature_groups[group_name]
                result = self.run_ablation_experiment(features_to_remove, f"remove_{group_name}")
                results[group_name] = result
        
        return results
    
    def run_cumulative_ablation(self, feature_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Run cumulative ablation removing groups one by one."""
        print("\nRunning cumulative ablation analysis...")
        
        results = {}
        removed_features = []
        
        # Order groups by expected importance (least to most important)
        removal_order = [
            "ticket_features", "embarked_features", "cabin_features",
            "interaction_features", "age_features", "fare_features",
            "survival_rate_features", "family_features", "title_features"
        ]
        
        for group_name in removal_order:
            if group_name in feature_groups:
                # Add current group to removal list
                removed_features.extend(feature_groups[group_name])
                
                # Run experiment with cumulative removals
                result = self.run_ablation_experiment(
                    removed_features.copy(), 
                    f"cumulative_remove_up_to_{group_name}"
                )
                results[f"cumulative_{group_name}"] = result
        
        return results
    
    def analyze_feature_interactions(self, feature_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Analyze interactions between feature groups."""
        print("\nAnalyzing feature group interactions...")
        
        results = {}
        important_groups = ["basic_demographics", "title_features", "family_features", "survival_rate_features"]
        
        # Test removing pairs of important groups
        for i, group1 in enumerate(important_groups):
            for group2 in important_groups[i+1:]:
                if group1 in feature_groups and group2 in feature_groups:
                    combined_features = feature_groups[group1] + feature_groups[group2]
                    
                    result = self.run_ablation_experiment(
                        combined_features,
                        f"remove_{group1}_and_{group2}"
                    )
                    results[f"{group1}_and_{group2}"] = result
        
        return results
    
    def generate_ablation_report(self, all_results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """Generate comprehensive ablation analysis report."""
        report_data = []
        
        for category, experiments in all_results.items():
            for experiment_name, metrics in experiments.items():
                if "error" not in metrics:
                    report_data.append({
                        "category": category,
                        "experiment": experiment_name,
                        "auc": metrics["auc"],
                        "score_drop": metrics["score_drop"],
                        "relative_drop_pct": metrics["relative_drop_pct"],
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "features_removed": metrics["features_removed"]
                    })
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by score drop (descending) to see most important features first
        report_df = report_df.sort_values("score_drop", ascending=False)
        
        return report_df
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save ablation results to files."""
        output_dir = Path(self.config.get("ablation_output_dir", "ablation_results"))
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_results_path = output_dir / "ablation_detailed_results.json"
        import json
        with open(detailed_results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for category, experiments in results.items():
                serializable_results[category] = {}
                for exp_name, metrics in experiments.items():
                    serializable_results[category][exp_name] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()
                    }
            json.dump(serializable_results, f, indent=2)
        
        # Generate and save report
        report_df = self.generate_ablation_report(results)
        report_path = output_dir / "ablation_report.csv"
        report_df.to_csv(report_path, index=False)
        
        # Save summary
        summary_path = output_dir / "ablation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Feature Ablation Analysis Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Baseline AUC: {self.baseline_score:.5f}\n\n")
            
            f.write("Top 10 Most Important Feature Groups (by AUC drop):\n")
            top_drops = report_df.nlargest(10, "score_drop")
            for _, row in top_drops.iterrows():
                f.write(f"  {row['experiment']}: {row['score_drop']:.5f} ({row['relative_drop_pct']:.2f}%)\n")
            
            f.write(f"\nTop 10 Least Important Feature Groups:\n")
            bottom_drops = report_df.nsmallest(10, "score_drop")
            for _, row in bottom_drops.iterrows():
                f.write(f"  {row['experiment']}: {row['score_drop']:.5f} ({row['relative_drop_pct']:.2f}%)\n")
        
        print(f"Ablation results saved to {output_dir}")
    
    def run_complete_ablation_study(self) -> Dict[str, Any]:
        """Run the complete ablation study."""
        print("Starting comprehensive feature ablation analysis...")
        
        # Get baseline score
        baseline_score = self.run_baseline_training()
        
        # Get all features by running a quick feature engineering step
        from modeling.train_cv import TrainingPipeline
        temp_pipeline = TrainingPipeline(self.config)
        df = temp_pipeline.load_and_validate_data()
        df = temp_pipeline.engineer_features(df)
        all_features = [col for col in df.columns if col not in ["Survived", "PassengerId"]]
        
        # Define feature groups
        feature_groups = self.get_feature_groups(all_features)
        print(f"Defined {len(feature_groups)} feature groups for ablation")
        
        # Run different types of ablation studies
        all_results = {}
        
        # Progressive ablation (one group at a time)
        all_results["progressive"] = self.run_progressive_ablation(feature_groups)
        
        # Cumulative ablation (removing groups cumulatively)
        all_results["cumulative"] = self.run_cumulative_ablation(feature_groups)
        
        # Feature interaction analysis
        all_results["interactions"] = self.analyze_feature_interactions(feature_groups)
        
        # Save results
        self.save_results(all_results)
        
        print("Ablation analysis completed!")
        return all_results


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(config: DictConfig) -> None:
    """Main ablation analysis entry point."""
    # Set seed for reproducibility
    set_seed(config.cv.random_state)
    
    # Run ablation study
    analyzer = FeatureAblationAnalyzer(config)
    results = analyzer.run_complete_ablation_study()
    
    print("Feature ablation analysis completed successfully!")


if __name__ == "__main__":
    main()