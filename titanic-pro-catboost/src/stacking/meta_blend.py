"""Meta-learner for blending base model predictions."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from core.utils import set_seed

warnings.filterwarnings('ignore')


def train_meta_learner(oof_matrix: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Train L2-regularized logistic regression meta-learner."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    set_seed(42)
    
    # Prepare data
    feature_cols = ['catboost', 'tfdf', 'lgbm', 'logistic']
    X_meta = oof_matrix[feature_cols].values
    y_meta = oof_matrix['target'].values
    
    print(f"Meta-learner training on {X_meta.shape[0]} samples with {X_meta.shape[1]} base models")
    
    # Cross-validation for meta-learner evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # Train final meta-learner on full data
    meta_learner = LogisticRegression(
        C=1.0,  # L2 regularization
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    # CV evaluation of meta-learner
    meta_oof = np.zeros(len(y_meta))
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_meta)):
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y_meta[train_idx], y_meta[val_idx]
        
        # Train meta-learner
        fold_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        fold_meta.fit(X_train, y_train)
        
        # Predict on validation
        val_pred = fold_meta.predict_proba(X_val)[:, 1]
        meta_oof[val_idx] = val_pred
        
        # Calculate fold score
        fold_auc = roc_auc_score(y_val, val_pred)
        cv_scores.append(fold_auc)
    
    # Train final meta-learner on all data
    meta_learner.fit(X_meta, y_meta)
    
    # Calculate meta-learner performance
    meta_cv_score = np.mean(cv_scores)
    meta_cv_std = np.std(cv_scores)
    
    # Compare to individual models
    individual_scores = {}
    for col in feature_cols:
        score = roc_auc_score(y_meta, oof_matrix[col])
        individual_scores[col] = score
    
    best_individual = max(individual_scores.values())
    meta_improvement = meta_cv_score - best_individual
    
    results = {
        'meta_cv_auc': meta_cv_score,
        'meta_cv_std': meta_cv_std,
        'best_individual_auc': best_individual,
        'meta_improvement': meta_improvement,
        'individual_scores': individual_scores,
        'meta_weights': dict(zip(feature_cols, meta_learner.coef_[0])),
        'meta_intercept': meta_learner.intercept_[0]
    }
    
    print(f"Meta-learner CV AUC: {meta_cv_score:.5f} ± {meta_cv_std:.5f}")
    print(f"Best individual AUC: {best_individual:.5f}")
    print(f"Meta improvement: {meta_improvement:+.5f}")
    print("Meta-learner weights:")
    for feature, weight in results['meta_weights'].items():
        print(f"  {feature}: {weight:.4f}")
    
    return meta_learner, results


def generate_blend_submission(meta_learner: Any, test_matrix: pd.DataFrame, 
                            output_path: Path) -> pd.DataFrame:
    """Generate blended submission using meta-learner."""
    
    feature_cols = ['catboost', 'tfdf', 'lgbm', 'logistic']
    X_test_meta = test_matrix[feature_cols].values
    
    # Generate meta predictions
    blend_proba = meta_learner.predict_proba(X_test_meta)[:, 1]
    blend_pred = (blend_proba >= 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_matrix['PassengerId'],
        'Survived': blend_pred
    })
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"Blended submission saved to {output_path}")
    
    # Also save probabilities for analysis
    prob_path = output_path.parent / f"{output_path.stem}_probabilities.csv"
    prob_submission = pd.DataFrame({
        'PassengerId': test_matrix['PassengerId'],
        'Survived_Proba': blend_proba
    })
    prob_submission.to_csv(prob_path, index=False)
    
    return submission


def run_meta_blending(config: Dict[str, Any], artifacts_dir: Path, 
                     output_dir: Path) -> Dict[str, Any]:
    """Run complete meta-blending pipeline."""
    print("Running meta-blending pipeline...")
    
    # Load OOF and test matrices
    oof_path = artifacts_dir / "oof_matrix.parquet"
    test_path = artifacts_dir / "test_matrix.parquet"
    
    if not oof_path.exists():
        print("⚠️  OOF matrix not found, building first...")
        from stacking.build_oof_matrix import build_oof_matrix
        oof_matrix, test_matrix = build_oof_matrix(config, artifacts_dir)
    else:
        oof_matrix = pd.read_parquet(oof_path)
        test_matrix = pd.read_parquet(test_path)
        print(f"Loaded OOF matrix: {oof_matrix.shape}")
        print(f"Loaded test matrix: {test_matrix.shape}")
    
    # Train meta-learner
    meta_learner, results = train_meta_learner(oof_matrix, config)
    
    # Generate submissions
    output_dir.mkdir(exist_ok=True)
    
    # Blended submission
    blend_submission_path = output_dir / "submission_blend.csv"
    blend_submission = generate_blend_submission(meta_learner, test_matrix, blend_submission_path)
    
    # Also generate base model submissions for comparison
    feature_cols = ['catboost', 'tfdf', 'lgbm', 'logistic']
    
    for model_name in feature_cols:
        base_pred = (test_matrix[model_name] >= 0.5).astype(int)
        base_submission = pd.DataFrame({
            'PassengerId': test_matrix['PassengerId'],
            'Survived': base_pred
        })
        base_path = output_dir / f"submission_{model_name}.csv"
        base_submission.to_csv(base_path, index=False)
    
    # Save meta-learner and results
    import pickle
    meta_path = artifacts_dir / "meta_learner.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_learner, f)
    
    results_path = artifacts_dir / "meta_blending_results.json"
    import json
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, (np.floating, np.integer)):
                serializable_results[k] = float(v)
            else:
                serializable_results[k] = v
        json.dump(serializable_results, f, indent=2)
    
    print(f"Meta-learner saved to {meta_path}")
    print(f"Results saved to {results_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("BLENDING SUMMARY")
    print("="*60)
    print(f"Meta-learner AUC: {results['meta_cv_auc']:.5f}")
    print(f"Best base model:   {results['best_individual_auc']:.5f}")
    print(f"Improvement:       {results['meta_improvement']:+.5f}")
    print(f"Submissions generated:")
    print(f"  Base models: submission_{{catboost,tfdf,lgbm,logistic}}.csv")
    print(f"  Blended:     submission_blend.csv")
    
    return results


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
    output_dir = Path(__file__).parent.parent.parent
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    results = run_meta_blending(config, artifacts_dir, output_dir)
    print(f"Meta-blending complete. Final improvement: {results['meta_improvement']:+.5f}")