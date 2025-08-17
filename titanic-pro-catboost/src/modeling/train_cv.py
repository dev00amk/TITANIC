"""Main training pipeline with cross-validation and MLflow tracking."""

import os
import sys
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.contracts import RawTrainSchema, EngineeredFeaturesSchema
from core.utils import set_seed, create_family_groups, stratified_group_split
from features.build import FeatureEngineer
from features.target_encoding import OutOfFoldTargetEncoder
from modeling.cat_model import CatBoostModelFactory

warnings.filterwarnings('ignore')


class TrainingPipeline:
    """Complete training pipeline with MLflow tracking."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: List[Any] = []
        self.oof_predictions = None
        self.feature_importance_agg = {}
        self.cv_scores = []
        
    def setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        mlflow_config = self.config.get("mlflow", {})
        
        if mlflow_config.get("enabled", True):
            experiment_name = mlflow_config.get("experiment_name", "titanic-catboost")
            mlflow.set_experiment(experiment_name)
            
            # Set tracking URI if specified
            if "tracking_uri" in mlflow_config:
                mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate raw training data."""
        data_path = Path(self.config.data.train_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Validate with Pandera schema
        try:
            validated_df = RawTrainSchema.validate(df)
            print(f"Data validation passed. Shape: {validated_df.shape}")
            return validated_df
        except Exception as e:
            print(f"Data validation failed: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering pipeline."""
        engineer = FeatureEngineer(self.config)
        
        # Apply feature engineering
        df_engineered, _ = engineer.engineer_features(df)
        
        # Validate engineered features
        try:
            validated_df = EngineeredFeaturesSchema.validate(df_engineered)
            print(f"Feature engineering completed. Shape: {validated_df.shape}")
            return validated_df
        except Exception as e:
            print(f"Feature validation failed: {e}")
            raise
    
    def setup_cross_validation(self, df: pd.DataFrame) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str]]:
        """Setup stratified group K-fold cross-validation."""
        cv_config = self.config.cv
        
        # Create family groups for group-aware CV
        family_groups = create_family_groups(df)
        
        # Setup CV splits
        cv_splits = stratified_group_split(
            df, 
            family_groups,
            target_col="Survived",
            n_splits=cv_config.n_splits,
            random_state=cv_config.random_state
        )
        
        # Get categorical features
        categorical_features = self.get_categorical_features(df)
        
        print(f"CV setup: {len(cv_splits)} folds with {len(categorical_features)} categorical features")
        return cv_splits, categorical_features
    
    def get_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """Identify categorical features for CatBoost."""
        # Exclude target and ID columns
        feature_cols = [col for col in df.columns if col not in ["Survived", "PassengerId"]]
        
        categorical_features = []
        for col in feature_cols:
            if (df[col].dtype == 'object' or 
                df[col].dtype.name == 'category' or
                col in ["Pclass", "SibSp", "Parch", "IsAlone", "HasSiblings", "HasParents", 
                       "HasChildren", "HasCabin", "TicketHasLetters", "IsChild"]):
                categorical_features.append(col)
        
        # Exclude survival rate features from categorical (they're continuous)
        categorical_features = [col for col in categorical_features if "_SurvivalRate" not in col]
        
        return categorical_features
    
    def apply_target_encoding(
        self, 
        df: pd.DataFrame, 
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """Apply out-of-fold target encoding."""
        target_encoding_config = self.config.get("target_encoding", {})
        
        if not target_encoding_config.get("enabled", True):
            return df
        
        features_to_encode = target_encoding_config.get("features", [
            "Title", "CabinDeck", "TicketPrefix", "Embarked", "FamilyID"
        ])
        
        # Filter features that exist in the dataframe
        features_to_encode = [f for f in features_to_encode if f in df.columns]
        
        if not features_to_encode:
            print("No features found for target encoding")
            return df
        
        # Apply OOF target encoding
        encoder = OutOfFoldTargetEncoder(
            alpha=target_encoding_config.get("alpha", 10.0),
            random_state=self.config.cv.random_state
        )
        
        df_encoded = encoder.fit_transform_oof(
            df, 
            df["Survived"], 
            features_to_encode,
            cv_splits
        )
        
        # Validate no leakage
        if encoder.validate_no_leakage(df, df["Survived"], df_encoded, features_to_encode, cv_splits):
            print("Target encoding validation passed - no leakage detected")
        else:
            raise ValueError("Target encoding validation failed - potential leakage detected")
        
        return df_encoded
    
    def train_single_seed(
        self, 
        df: pd.DataFrame, 
        cv_splits: List[Tuple[np.ndarray, np.ndarray]], 
        categorical_features: List[str],
        seed: int
    ) -> Tuple[List[Any], np.ndarray, Dict[str, float]]:
        """Train models for a single seed across all CV folds."""
        set_seed(seed)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ["Survived", "PassengerId"]]
        X = df[feature_cols].values
        y = df["Survived"].values
        
        models = []
        oof_pred = np.zeros(len(df))
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"  Fold {fold_idx + 1}/{len(cv_splits)}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = CatBoostModelFactory.create_model(
                self.config.catboost, 
                categorical_features
            )
            
            model = CatBoostModelFactory.fit_with_eval(
                model, X_train, y_train, X_val, y_val, 
                categorical_features, verbose=False
            )
            
            # Make predictions
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            oof_pred[val_idx] = val_pred_proba
            
            # Calculate fold metrics
            val_pred = (val_pred_proba >= 0.5).astype(int)
            fold_score = {
                "accuracy": accuracy_score(y_val, val_pred),
                "precision": precision_score(y_val, val_pred),
                "recall": recall_score(y_val, val_pred),
                "f1": f1_score(y_val, val_pred),
                "auc": roc_auc_score(y_val, val_pred_proba)
            }
            fold_scores.append(fold_score)
            
            models.append(model)
        
        # Calculate overall metrics for this seed
        overall_pred = (oof_pred >= 0.5).astype(int)
        seed_metrics = {
            "accuracy": accuracy_score(y, overall_pred),
            "precision": precision_score(y, overall_pred),
            "recall": recall_score(y, overall_pred),
            "f1": f1_score(y, overall_pred),
            "auc": roc_auc_score(y, oof_pred)
        }
        
        return models, oof_pred, seed_metrics
    
    def aggregate_feature_importance(self, models: List[Any], feature_names: List[str]) -> Dict[str, float]:
        """Aggregate feature importance across all models."""
        importance_sum = {}
        model_count = 0
        
        for model in models:
            try:
                importance = CatBoostModelFactory.get_feature_importance(model, feature_names)
                model_count += 1
                
                for feature, imp in importance.items():
                    importance_sum[feature] = importance_sum.get(feature, 0) + imp
            except Exception as e:
                print(f"Warning: Could not get feature importance from model: {e}")
                continue
        
        # Average the importance
        if model_count > 0:
            return {feature: imp / model_count for feature, imp in importance_sum.items()}
        else:
            return {}
    
    def log_to_mlflow(
        self, 
        metrics: Dict[str, float], 
        feature_importance: Dict[str, float],
        config: DictConfig
    ) -> None:
        """Log results to MLflow."""
        if not self.config.get("mlflow", {}).get("enabled", True):
            return
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "cv_folds": config.cv.n_splits,
                "n_seeds": config.cv.n_seeds,
                "catboost_iterations": config.catboost.iterations,
                "catboost_learning_rate": config.catboost.learning_rate,
                "catboost_depth": config.catboost.depth,
                "catboost_l2_leaf_reg": config.catboost.l2_leaf_reg,
            })
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log feature importance as artifacts
            if feature_importance:
                importance_df = pd.DataFrame([
                    {"feature": k, "importance": v} 
                    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                ])
                importance_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
    
    def save_artifacts(
        self, 
        models: List[Any], 
        oof_predictions: np.ndarray, 
        feature_importance: Dict[str, float],
        feature_names: List[str]
    ) -> None:
        """Save training artifacts."""
        artifacts_dir = Path(self.config.get("artifacts_dir", "artifacts"))
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save OOF predictions
        np.save(artifacts_dir / "oof_predictions.npy", oof_predictions)
        
        # Save feature importance
        importance_df = pd.DataFrame([
            {"feature": k, "importance": v} 
            for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        ])
        importance_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)
        
        # Save feature names
        pd.DataFrame({"feature": feature_names}).to_csv(artifacts_dir / "feature_names.csv", index=False)
        
        print(f"Artifacts saved to {artifacts_dir}")
    
    def run(self) -> Dict[str, float]:
        """Run the complete training pipeline."""
        print("Starting training pipeline...")
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Load and validate data
        df = self.load_and_validate_data()
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Setup cross-validation
        cv_splits, categorical_features = self.setup_cross_validation(df)
        
        # Apply target encoding
        df = self.apply_target_encoding(df, cv_splits)
        
        # Get feature names
        feature_names = [col for col in df.columns if col not in ["Survived", "PassengerId"]]
        
        # Multi-seed training
        all_models = []
        all_oof_predictions = []
        seed_metrics_list = []
        
        n_seeds = self.config.cv.get("n_seeds", 3)
        base_seed = self.config.cv.get("random_state", 42)
        
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            print(f"Training seed {seed_idx + 1}/{n_seeds} (seed={seed})")
            
            models, oof_pred, seed_metrics = self.train_single_seed(
                df, cv_splits, categorical_features, seed
            )
            
            all_models.extend(models)
            all_oof_predictions.append(oof_pred)
            seed_metrics_list.append(seed_metrics)
            
            print(f"  Seed {seed_idx + 1} CV Score: {seed_metrics['auc']:.5f}")
        
        # Ensemble OOF predictions
        self.oof_predictions = np.mean(all_oof_predictions, axis=0)
        
        # Calculate final ensemble metrics
        y_true = df["Survived"].values
        final_pred = (self.oof_predictions >= 0.5).astype(int)
        
        final_metrics = {
            "ensemble_accuracy": accuracy_score(y_true, final_pred),
            "ensemble_precision": precision_score(y_true, final_pred),
            "ensemble_recall": recall_score(y_true, final_pred),
            "ensemble_f1": f1_score(y_true, final_pred),
            "ensemble_auc": roc_auc_score(y_true, self.oof_predictions)
        }
        
        # Aggregate feature importance
        self.feature_importance_agg = self.aggregate_feature_importance(all_models, feature_names)
        
        # Log to MLflow
        self.log_to_mlflow(final_metrics, self.feature_importance_agg, self.config)
        
        # Save artifacts
        self.save_artifacts(
            all_models, self.oof_predictions, 
            self.feature_importance_agg, feature_names
        )
        
        # Store models and results
        self.models = all_models
        
        print(f"\nTraining completed!")
        print(f"Final Ensemble AUC: {final_metrics['ensemble_auc']:.5f}")
        print(f"Final Ensemble Accuracy: {final_metrics['ensemble_accuracy']:.5f}")
        
        return final_metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(config: DictConfig) -> None:
    """Main training entry point with Hydra."""
    # Set seeds
    set_seed(config.cv.random_state)
    
    # Run training pipeline
    pipeline = TrainingPipeline(config)
    metrics = pipeline.run()
    
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()