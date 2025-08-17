"""Permutation feature importance with CV impact analysis."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from modeling.cat_model import CatBoostModelFactory
from core.utils import set_seed

warnings.filterwarnings('ignore')


def calculate_permutation_impact(X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str],
                               categorical_features: List[str],
                               cv_splits: List,
                               config: Dict[str, Any],
                               n_repeats: int = 3) -> pd.DataFrame:
    """Calculate permutation feature importance with CV delta."""
    print(f"Calculating permutation impact for {len(feature_names)} features...")
    
    set_seed(42)
    
    # Baseline CV score
    baseline_scores = []
    for train_idx, val_idx in cv_splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostModelFactory.create_model(config["catboost"], categorical_features)
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X_train, y_train, X_val, y_val, categorical_features, verbose=False
        )
        
        val_pred = fitted_model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        baseline_scores.append(roc_auc_score(y_val, val_pred))
    
    baseline_cv = np.mean(baseline_scores)
    
    # Permutation impact for each feature
    impacts = []
    
    for feat_idx, feature_name in enumerate(feature_names):
        print(f"  Permuting {feature_name} ({feat_idx+1}/{len(feature_names)})")
        
        repeat_impacts = []
        
        for repeat in range(n_repeats):
            # Permute feature
            X_permuted = X.copy()
            np.random.seed(42 + repeat)
            X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
            
            # Calculate CV with permuted feature
            permuted_scores = []
            for train_idx, val_idx in cv_splits:
                X_train, X_val = X_permuted[train_idx], X_permuted[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = CatBoostModelFactory.create_model(config["catboost"], categorical_features)
                fitted_model = CatBoostModelFactory.fit_with_eval(
                    model, X_train, y_train, X_val, y_val, categorical_features, verbose=False
                )
                
                val_pred = fitted_model.predict_proba(X_val)[:, 1]
                permuted_scores.append(roc_auc_score(y_val, val_pred))
            
            permuted_cv = np.mean(permuted_scores)
            impact = baseline_cv - permuted_cv
            repeat_impacts.append(impact)
        
        avg_impact = np.mean(repeat_impacts)
        std_impact = np.std(repeat_impacts)
        
        impacts.append({
            "feature": feature_name,
            "permutation_impact": avg_impact,
            "impact_std": std_impact,
            "baseline_cv": baseline_cv,
            "relative_impact": avg_impact / baseline_cv if baseline_cv > 0 else 0
        })
    
    # Convert to DataFrame and sort by impact
    impact_df = pd.DataFrame(impacts)
    impact_df = impact_df.sort_values("permutation_impact", ascending=False)
    impact_df["rank"] = range(1, len(impact_df) + 1)
    
    return impact_df


def run_permutation_impact_analysis(config: Dict[str, Any], 
                                   output_path: Path = None) -> pd.DataFrame:
    """Run full permutation impact analysis."""
    from modeling.train_cv import TrainingPipeline
    
    # Use faster config for permutation testing
    fast_config = config.copy()
    fast_config["catboost"]["iterations"] = 300
    fast_config["cv"]["n_seeds"] = 1
    
    pipeline = TrainingPipeline(fast_config)
    
    # Prepare data
    df = pipeline.load_and_validate_data()
    df = pipeline.engineer_features(df)
    cv_splits, categorical_features = pipeline.setup_cross_validation(df)
    df_encoded = pipeline.apply_target_encoding(df, cv_splits)
    
    feature_names = [col for col in df_encoded.columns if col not in ["Survived", "PassengerId"]]
    X = df_encoded[feature_names].values
    y = df_encoded["Survived"].values
    
    # Calculate permutation impacts
    impact_df = calculate_permutation_impact(
        X, y, feature_names, categorical_features, cv_splits, fast_config
    )
    
    # Save results
    if output_path:
        output_path.parent.mkdir(exist_ok=True)
        impact_df.to_csv(output_path, index=False)
        print(f"Permutation impact saved to {output_path}")
    
    # Print top features
    print(f"\nTop 10 features by permutation impact:")
    print(impact_df[["rank", "feature", "permutation_impact", "relative_impact"]].head(10).to_string(index=False))
    
    return impact_df


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    output_path = Path(__file__).parent.parent.parent / "reports" / "feature_perm_impact.csv"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    impact_df = run_permutation_impact_analysis(config, output_path)
    print(f"Analysis complete. Impact range: {impact_df['permutation_impact'].min():.4f} to {impact_df['permutation_impact'].max():.4f}")