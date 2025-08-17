"""Inference pipeline for test set predictions."""

import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.contracts import RawTestSchema
from features.build import FeatureEngineer
from features.target_encoding import OutOfFoldTargetEncoder
from modeling.cat_model import CatBoostModelFactory


class InferencePipeline:
    """Inference pipeline for test predictions."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.models: List[Any] = []
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.target_encoder: Optional[OutOfFoldTargetEncoder] = None
        self.feature_names: List[str] = []
        
    def load_models(self, models_dir: Path) -> None:
        """Load trained models from artifacts directory."""
        models_path = models_dir / "models.pkl"
        
        if not models_path.exists():
            raise FileNotFoundError(f"Models file not found at {models_path}")
            
        with open(models_path, 'rb') as f:
            self.models = pickle.load(f)
            
        print(f"Loaded {len(self.models)} models")
    
    def load_feature_artifacts(self, artifacts_dir: Path) -> None:
        """Load feature engineering artifacts."""
        # Load feature names
        feature_names_path = artifacts_dir / "feature_names.csv"
        if feature_names_path.exists():
            feature_names_df = pd.read_csv(feature_names_path)
            self.feature_names = feature_names_df["feature"].tolist()
        
        # Load feature engineer if saved
        feature_engineer_path = artifacts_dir / "feature_engineer.pkl"
        if feature_engineer_path.exists():
            with open(feature_engineer_path, 'rb') as f:
                self.feature_engineer = pickle.load(f)
        else:
            # Create new feature engineer with config
            self.feature_engineer = FeatureEngineer(self.config)
        
        # Load target encoder if saved
        target_encoder_path = artifacts_dir / "target_encoder.pkl"
        if target_encoder_path.exists():
            with open(target_encoder_path, 'rb') as f:
                self.target_encoder = pickle.load(f)
        
        print("Feature artifacts loaded")
    
    def load_training_data_for_context(self) -> pd.DataFrame:
        """Load training data to provide context for feature engineering."""
        train_path = Path(self.config.data.train_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        return pd.read_csv(train_path)
    
    def load_and_validate_test_data(self) -> pd.DataFrame:
        """Load and validate test data."""
        test_path = Path(self.config.data.test_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        # Load data
        df = pd.read_csv(test_path)
        
        # Validate with Pandera schema
        try:
            validated_df = RawTestSchema.validate(df)
            print(f"Test data validation passed. Shape: {validated_df.shape}")
            return validated_df
        except Exception as e:
            print(f"Test data validation failed: {e}")
            # Continue with warning for test data
            print("Continuing with test data despite validation issues...")
            return df
    
    def engineer_test_features(self, test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to test data using training context."""
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not loaded")
        
        # Apply feature engineering with training context
        _, test_engineered = self.feature_engineer.engineer_features(train_df, test_df)
        
        if test_engineered is None:
            raise ValueError("Feature engineering returned None for test data")
        
        print(f"Test feature engineering completed. Shape: {test_engineered.shape}")
        return test_engineered
    
    def apply_target_encoding_to_test(self, test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding to test data."""
        target_encoding_config = self.config.get("target_encoding", {})
        
        if not target_encoding_config.get("enabled", True):
            return test_df
        
        features_to_encode = target_encoding_config.get("features", [
            "Title", "CabinDeck", "TicketPrefix", "Embarked", "FamilyID"
        ])
        
        # Filter features that exist in the dataframe
        features_to_encode = [f for f in features_to_encode if f in test_df.columns]
        
        if not features_to_encode:
            print("No features found for target encoding")
            return test_df
        
        # Create and fit encoder on full training data
        encoder = OutOfFoldTargetEncoder(
            alpha=target_encoding_config.get("alpha", 10.0),
            random_state=self.config.cv.random_state
        )
        
        # Fit on training data
        encoder.fit_full_data(train_df, train_df["Survived"], features_to_encode)
        
        # Transform test data
        test_encoded = encoder.transform(test_df, features_to_encode)
        
        print(f"Target encoding applied to test data")
        return test_encoded
    
    def prepare_test_features(self, test_df: pd.DataFrame) -> np.ndarray:
        """Prepare test features for model input."""
        # Ensure we have the same features as training
        if not self.feature_names:
            # If feature names not loaded, use all non-ID columns
            feature_cols = [col for col in test_df.columns if col != "PassengerId"]
        else:
            # Use saved feature names
            feature_cols = []
            for feature in self.feature_names:
                if feature in test_df.columns:
                    feature_cols.append(feature)
                else:
                    print(f"Warning: Feature {feature} not found in test data")
        
        if not feature_cols:
            raise ValueError("No valid features found for prediction")
        
        X_test = test_df[feature_cols].values
        print(f"Test features prepared. Shape: {X_test.shape}")
        return X_test
    
    def make_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """Make ensemble predictions using all trained models."""
        if not self.models:
            raise ValueError("No models loaded for prediction")
        
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions.append(pred_proba)
            except Exception as e:
                print(f"Warning: Model {i} failed to predict: {e}")
                continue
        
        if not predictions:
            raise ValueError("All models failed to make predictions")
        
        # Ensemble by averaging
        ensemble_pred = np.mean(predictions, axis=0)
        
        print(f"Ensemble predictions created from {len(predictions)} models")
        return ensemble_pred
    
    def save_predictions(self, test_df: pd.DataFrame, predictions: np.ndarray) -> None:
        """Save predictions to submission file."""
        # Create submission dataframe
        submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": (predictions >= 0.5).astype(int)
        })
        
        # Save submission
        output_path = Path(self.config.get("output_path", "submission.csv"))
        submission.to_csv(output_path, index=False)
        
        # Also save probabilities for analysis
        prob_output_path = output_path.parent / f"{output_path.stem}_probabilities.csv"
        prob_submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived_Proba": predictions
        })
        prob_submission.to_csv(prob_output_path, index=False)
        
        print(f"Submission saved to {output_path}")
        print(f"Probabilities saved to {prob_output_path}")
        
        # Print prediction statistics
        pred_binary = (predictions >= 0.5).astype(int)
        survival_rate = pred_binary.mean()
        print(f"Predicted survival rate: {survival_rate:.3f}")
        print(f"Prediction distribution: {np.bincount(pred_binary)}")
    
    def run(self) -> np.ndarray:
        """Run the complete inference pipeline."""
        print("Starting inference pipeline...")
        
        # Load artifacts
        artifacts_dir = Path(self.config.get("artifacts_dir", "artifacts"))
        
        try:
            self.load_models(artifacts_dir)
        except FileNotFoundError:
            print("Warning: Could not load saved models. Ensure training pipeline has been run.")
            raise
        
        self.load_feature_artifacts(artifacts_dir)
        
        # Load data
        train_df = self.load_training_data_for_context()
        test_df = self.load_and_validate_test_data()
        
        # Feature engineering
        test_df = self.engineer_test_features(test_df, train_df)
        
        # Target encoding
        test_df = self.apply_target_encoding_to_test(test_df, train_df)
        
        # Prepare features
        X_test = self.prepare_test_features(test_df)
        
        # Make predictions
        predictions = self.make_predictions(X_test)
        
        # Save results
        self.save_predictions(test_df, predictions)
        
        print("Inference pipeline completed successfully!")
        return predictions


def load_models_from_training() -> List[Any]:
    """Helper function to load models after training."""
    # This would be called by the training pipeline to save models
    # Implementation depends on how models are stored during training
    pass


def save_models_for_inference(models: List[Any], artifacts_dir: Path) -> None:
    """Helper function to save models for inference."""
    models_path = artifacts_dir / "models.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {models_path}")


def save_feature_artifacts(
    feature_engineer: FeatureEngineer,
    target_encoder: Optional[OutOfFoldTargetEncoder],
    feature_names: List[str],
    artifacts_dir: Path
) -> None:
    """Save feature engineering artifacts for inference."""
    # Save feature engineer
    with open(artifacts_dir / "feature_engineer.pkl", 'wb') as f:
        pickle.dump(feature_engineer, f)
    
    # Save target encoder if exists
    if target_encoder:
        with open(artifacts_dir / "target_encoder.pkl", 'wb') as f:
            pickle.dump(target_encoder, f)
    
    # Save feature names
    pd.DataFrame({"feature": feature_names}).to_csv(
        artifacts_dir / "feature_names.csv", index=False
    )
    
    print("Feature artifacts saved for inference")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(config: DictConfig) -> None:
    """Main inference entry point with Hydra."""
    pipeline = InferencePipeline(config)
    predictions = pipeline.run()
    print(f"Generated {len(predictions)} predictions")


if __name__ == "__main__":
    main()