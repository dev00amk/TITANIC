"""Shuffled target validation to detect leakage."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from modeling.train_cv import TrainingPipeline
from core.utils import set_seed

warnings.filterwarnings('ignore')


def run_shuffled_target_check(config: Dict[str, Any], random_state: int = 42) -> Dict[str, float]:
    """
    Train with shuffled targets and verify CV score is near random (0.48-0.52).
    If CV >> 0.52, indicates leakage.
    """
    print("Running shuffled target validation...")
    
    set_seed(random_state)
    
    # Create pipeline with shuffled config
    shuffled_config = config.copy()
    shuffled_config["catboost"]["iterations"] = 200  # Faster for validation
    shuffled_config["cv"]["n_seeds"] = 2  # Reduce for speed
    
    pipeline = TrainingPipeline(shuffled_config)
    
    # Load and engineer features normally
    df = pipeline.load_and_validate_data()
    df = pipeline.engineer_features(df)
    
    # SHUFFLE THE TARGET - this is the key validation
    df_shuffled = df.copy()
    shuffled_target = df["Survived"].sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_shuffled["Survived"] = shuffled_target
    
    # Setup CV with shuffled target
    cv_splits, categorical_features = pipeline.setup_cross_validation(df_shuffled)
    df_encoded = pipeline.apply_target_encoding(df_shuffled, cv_splits)
    
    # Train with shuffled target
    feature_names = [col for col in df_encoded.columns if col not in ["Survived", "PassengerId"]]
    X = df_encoded[feature_names].values
    y = df_encoded["Survived"].values
    
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        from modeling.cat_model import CatBoostModelFactory
        model = CatBoostModelFactory.create_model(shuffled_config["catboost"], categorical_features)
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X_train, y_train, X_val, y_val, categorical_features, verbose=False
        )
        
        # Evaluate
        val_pred = fitted_model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(auc)
    
    mean_cv = np.mean(fold_scores)
    std_cv = np.std(fold_scores)
    
    results = {
        "shuffled_target_cv_mean": mean_cv,
        "shuffled_target_cv_std": std_cv,
        "min_cv": np.min(fold_scores),
        "max_cv": np.max(fold_scores),
        "leakage_detected": mean_cv > 0.52 or mean_cv < 0.48
    }
    
    print(f"Shuffled target CV: {mean_cv:.4f} ± {std_cv:.4f}")
    
    if results["leakage_detected"]:
        print(f"⚠️  LEAKAGE DETECTED: CV {mean_cv:.4f} outside [0.48, 0.52]")
    else:
        print(f"✅ No leakage detected: CV {mean_cv:.4f} in valid range")
    
    return results


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    results = run_shuffled_target_check(config)
    print(f"Results: {results}")