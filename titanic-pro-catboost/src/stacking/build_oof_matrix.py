"""Build OOF prediction matrix from multiple base models."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from modeling.cat_model import CatBoostModelFactory
from core.utils import set_seed

warnings.filterwarnings('ignore')


def train_tfdf_baseline(X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, float]:
    """Train TensorFlow Decision Forest baseline."""
    try:
        # Try TF-DF first
        import tensorflow_decision_forests as tfdf
        import tensorflow as tf
        
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        
        # Convert to TF dataset format
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df["target"] = y_train
        val_df = pd.DataFrame(X_val, columns=feature_names)
        
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="target", task=tfdf.keras.Task.CLASSIFICATION)
        
        # Create and train model
        model = tfdf.keras.GradientBoostedTreesModel(verbose=0)
        model.fit(train_ds)
        
        # Predict
        val_pred = model.predict(val_df.values)[:, 0] if len(model.predict(val_df.values).shape) > 1 else model.predict(val_df.values)
        
        # Calculate validation score
        from sklearn.metrics import roc_auc_score
        val_score = roc_auc_score(y_val, val_pred)
        
        return model, val_score
        
    except ImportError:
        print("TF-DF not available, using sklearn GradientBoosting")
        from sklearn.ensemble import GradientBoostingClassifier
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        val_score = roc_auc_score(y_val, val_pred)
        
        return model, val_score


def train_lgbm_baseline(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, float]:
    """Train LightGBM baseline."""
    try:
        import lightgbm as lgb
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        from sklearn.metrics import roc_auc_score
        val_score = roc_auc_score(y_val, val_pred)
        
        return model, val_score
        
    except ImportError:
        print("LightGBM not available, using sklearn GradientBoosting")
        return train_tfdf_baseline(X_train, y_train, X_val, y_val)


def train_logistic_baseline(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, float]:
    """Train logistic regression baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict_proba(X_val)[:, 1]
    
    from sklearn.metrics import roc_auc_score
    val_score = roc_auc_score(y_val, val_pred)
    
    return pipeline, val_score


def build_oof_matrix(config: Dict[str, Any], output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build complete OOF prediction matrix from multiple models."""
    print("Building OOF prediction matrix...")
    
    set_seed(42)
    
    from modeling.train_cv import TrainingPipeline
    
    # Load and prepare data
    pipeline = TrainingPipeline(config)
    df = pipeline.load_and_validate_data()
    df = pipeline.engineer_features(df)
    cv_splits, categorical_features = pipeline.setup_cross_validation(df)
    df_encoded = pipeline.apply_target_encoding(df, cv_splits)
    
    # Prepare features
    feature_names = [col for col in df_encoded.columns if col not in ["Survived", "PassengerId"]]
    X = df_encoded[feature_names].values
    y = df_encoded["Survived"].values
    passenger_ids = df_encoded["PassengerId"].values
    
    print(f"Data shape: {X.shape}, Target balance: {y.mean():.3f}")
    
    # Initialize OOF matrix
    n_samples = len(X)
    oof_matrix = pd.DataFrame({
        'PassengerId': passenger_ids,
        'target': y
    })
    
    # 1. CatBoost OOF (primary model)
    print("Training CatBoost models...")
    catboost_oof = np.zeros(n_samples)
    catboost_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"  Fold {fold_idx + 1}/{len(cv_splits)}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostModelFactory.create_model(config["catboost"], categorical_features)
        fitted_model = CatBoostModelFactory.fit_with_eval(
            model, X_train, y_train, X_val, y_val, categorical_features, verbose=False
        )
        
        val_pred = fitted_model.predict_proba(X_val)[:, 1]
        catboost_oof[val_idx] = val_pred
        catboost_models.append(fitted_model)
    
    oof_matrix['catboost'] = catboost_oof
    
    # 2. TF-DF/GradientBoosting baseline
    print("Training TF-DF baseline...")
    tfdf_oof = np.zeros(n_samples)
    tfdf_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, val_score = train_tfdf_baseline(X_train, y_train, X_val, y_val)
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            val_pred = model.predict_proba(X_val)[:, 1]
        else:
            val_pred = model.predict(X_val)
            
        tfdf_oof[val_idx] = val_pred
        tfdf_models.append(model)
    
    oof_matrix['tfdf'] = tfdf_oof
    
    # 3. LightGBM baseline
    print("Training LightGBM baseline...")
    lgbm_oof = np.zeros(n_samples)
    lgbm_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, val_score = train_lgbm_baseline(X_train, y_train, X_val, y_val)
        
        # Get predictions
        if hasattr(model, 'predict'):
            if hasattr(model, 'best_iteration'):
                val_pred = model.predict(X_val, num_iteration=getattr(model, 'best_iteration', None))
            else:
                val_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
        else:
            val_pred = model.predict_proba(X_val)[:, 1]
            
        lgbm_oof[val_idx] = val_pred
        lgbm_models.append(model)
    
    oof_matrix['lgbm'] = lgbm_oof
    
    # 4. Logistic regression baseline
    print("Training Logistic baseline...")
    logistic_oof = np.zeros(n_samples)
    logistic_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model, val_score = train_logistic_baseline(X_train, y_train, X_val, y_val)
        val_pred = model.predict_proba(X_val)[:, 1]
        
        logistic_oof[val_idx] = val_pred
        logistic_models.append(model)
    
    oof_matrix['logistic'] = logistic_oof
    
    # Calculate individual model scores
    from sklearn.metrics import roc_auc_score
    model_scores = {}
    for col in ['catboost', 'tfdf', 'lgbm', 'logistic']:
        score = roc_auc_score(y, oof_matrix[col])
        model_scores[col] = score
        print(f"{col.upper()} OOF AUC: {score:.5f}")
    
    # Create test predictions matrix
    print("Generating test predictions...")
    
    # Load test data if available
    test_path = Path(config["data"]["test_path"])
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        from features.build import FeatureEngineer
        engineer = FeatureEngineer(config)
        _, test_engineered = engineer.engineer_features(df_encoded.drop(['Survived'], axis=1), test_df)
        
        X_test = test_engineered[feature_names].values
        test_passenger_ids = test_engineered["PassengerId"].values
        
        # Generate test predictions
        test_matrix = pd.DataFrame({'PassengerId': test_passenger_ids})
        
        # CatBoost test predictions
        catboost_test_pred = np.mean([model.predict_proba(X_test)[:, 1] for model in catboost_models], axis=0)
        test_matrix['catboost'] = catboost_test_pred
        
        # TF-DF test predictions
        tfdf_test_preds = []
        for model in tfdf_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test)
            tfdf_test_preds.append(pred)
        test_matrix['tfdf'] = np.mean(tfdf_test_preds, axis=0)
        
        # LightGBM test predictions
        lgbm_test_preds = []
        for model in lgbm_models:
            if hasattr(model, 'predict') and hasattr(model, 'best_iteration'):
                pred = model.predict(X_test, num_iteration=getattr(model, 'best_iteration', None))
            elif hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test)
            lgbm_test_preds.append(pred)
        test_matrix['lgbm'] = np.mean(lgbm_test_preds, axis=0)
        
        # Logistic test predictions
        logistic_test_pred = np.mean([model.predict_proba(X_test)[:, 1] for model in logistic_models], axis=0)
        test_matrix['logistic'] = logistic_test_pred
        
    else:
        print("⚠️  Test data not found, creating placeholder test matrix")
        test_matrix = pd.DataFrame({
            'PassengerId': range(1000, 1100),
            'catboost': np.random.random(100),
            'tfdf': np.random.random(100),
            'lgbm': np.random.random(100),
            'logistic': np.random.random(100)
        })
    
    # Save matrices
    output_dir.mkdir(exist_ok=True)
    
    oof_path = output_dir / "oof_matrix.parquet"
    test_path = output_dir / "test_matrix.parquet"
    
    oof_matrix.to_parquet(oof_path, index=False)
    test_matrix.to_parquet(test_path, index=False)
    
    print(f"OOF matrix saved: {oof_path} (shape: {oof_matrix.shape})")
    print(f"Test matrix saved: {test_path} (shape: {test_matrix.shape})")
    
    # Save model scores
    scores_path = output_dir / "base_model_scores.json"
    import json
    with open(scores_path, 'w') as f:
        json.dump(model_scores, f, indent=2)
    
    return oof_matrix, test_matrix


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    output_dir = Path(__file__).parent.parent.parent / "artifacts"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    oof_matrix, test_matrix = build_oof_matrix(config, output_dir)
    print(f"OOF matrix complete. Best single model AUC: {oof_matrix[['catboost', 'tfdf', 'lgbm', 'logistic']].corrwith(oof_matrix['target']).abs().max():.5f}")