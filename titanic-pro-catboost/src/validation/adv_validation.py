"""Adversarial validation to detect train/test distribution shift."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings

sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


def prepare_adversarial_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare combined dataset for adversarial validation."""
    # Add source labels
    train_labeled = train_df.copy()
    train_labeled['is_test'] = 0
    
    test_labeled = test_df.copy()
    test_labeled['is_test'] = 1
    
    # Remove target from train if present
    if 'Survived' in train_labeled.columns:
        train_labeled = train_labeled.drop('Survived', axis=1)
    
    # Combine datasets
    combined = pd.concat([train_labeled, test_labeled], ignore_index=True)
    
    # Extract labels
    y = combined['is_test'].values
    X = combined.drop(['is_test', 'PassengerId'], axis=1, errors='ignore')
    
    return X, y


def run_adversarial_validation(config: Dict[str, Any], 
                             output_path: Path = None) -> Dict[str, Any]:
    """Run adversarial validation to detect train/test shift."""
    print("Running adversarial validation...")
    
    from features.build import FeatureEngineer
    
    # Load and engineer features for both train and test
    train_path = Path(config["data"]["train_path"])
    test_path = Path(config["data"]["test_path"])
    
    if not train_path.exists() or not test_path.exists():
        print("⚠️  Train or test data not found, creating synthetic test data")
        train_df = pd.read_csv(train_path) if train_path.exists() else None
        if train_df is None:
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # Create synthetic test data
        test_df = train_df.sample(n=min(200, len(train_df)//2), random_state=42).copy()
        test_df = test_df.drop('Survived', axis=1)
        test_df['PassengerId'] = range(1000, 1000 + len(test_df))
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    
    # Engineer features consistently
    engineer = FeatureEngineer(config)
    train_engineered, test_engineered = engineer.engineer_features(train_df, test_df)
    
    # Prepare adversarial dataset
    X, y = prepare_adversarial_data(train_engineered, test_engineered)
    
    # Handle categorical columns
    categorical_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical_features.append(col)
    
    # Convert to numeric
    X_numeric = X.copy()
    for col in categorical_features:
        X_numeric[col] = pd.Categorical(X_numeric[col]).codes
    
    # Train adversarial classifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import catboost as cb
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    feature_importances = []
    
    for train_idx, val_idx in cv.split(X_numeric, y):
        X_train, X_val = X_numeric.iloc[train_idx], X_numeric.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train CatBoost classifier
        model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_seed=42
        )
        
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        cv_scores.append(auc)
        
        # Get feature importance
        importance = model.get_feature_importance()
        feature_importances.append(importance)
    
    # Calculate average metrics
    avg_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    
    # Average feature importance
    avg_importance = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_numeric.columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    # Identify top shift features
    top_shift_features = feature_importance_df.head(10)
    
    results = {
        'adversarial_auc': avg_auc,
        'auc_std': std_auc,
        'shift_detected': avg_auc > 0.6,
        'top_shift_features': top_shift_features.to_dict('records'),
        'n_train': len(train_engineered),
        'n_test': len(test_engineered),
        'n_features': len(X_numeric.columns)
    }
    
    # Print results
    print(f"Adversarial AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    if results['shift_detected']:
        print(f"⚠️  Distribution shift detected (AUC > 0.6)")
        print("Top features driving shift:")
        for feat in top_shift_features.head(5).itertuples():
            print(f"  {feat.feature}: {feat.importance:.4f}")
    else:
        print(f"✅ No significant distribution shift detected")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(exist_ok=True)
        import json
        with open(output_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, (np.floating, np.integer)):
                    serializable_results[k] = float(v)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    serializable_results[k] = [
                        {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                         for kk, vv in item.items()} for item in v
                    ]
                else:
                    serializable_results[k] = v
            json.dump(serializable_results, f, indent=2)
        print(f"Adversarial validation results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    output_path = Path(__file__).parent.parent.parent / "reports" / "adv_val.json"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    results = run_adversarial_validation(config, output_path)
    print(f"Shift detected: {results['shift_detected']}, AUC: {results['adversarial_auc']:.4f}")