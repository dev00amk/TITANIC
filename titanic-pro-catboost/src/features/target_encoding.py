"""Out-of-fold target encoding to prevent leakage."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold


class OutOfFoldTargetEncoder:
    """Out-of-fold target encoder with smoothing."""
    
    def __init__(self, alpha: float = 10.0, random_state: int = 42):
        """
        Initialize OOF target encoder.
        
        Args:
            alpha: Smoothing parameter for Bayesian mean encoding
            random_state: Random state for CV splits
        """
        self.alpha = alpha
        self.random_state = random_state
        self.global_mean_: Optional[float] = None
        self.encoding_maps_: Dict[str, Dict] = {}
        
    def _smooth_encode(self, counts: pd.Series, sums: pd.Series, global_mean: float) -> pd.Series:
        """Apply Bayesian mean encoding with smoothing."""
        smoothed = (sums + self.alpha * global_mean) / (counts + self.alpha)
        return smoothed
    
    def fit_transform_oof(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        features: List[str],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Fit and transform using out-of-fold encoding.
        
        Args:
            X: Feature dataframe
            y: Target series
            features: List of feature names to encode
            cv_splits: List of (train_idx, val_idx) tuples for CV
            
        Returns:
            DataFrame with OOF encoded features
        """
        X_encoded = X.copy()
        self.global_mean_ = y.mean()
        
        # Initialize OOF columns
        for feature in features:
            X_encoded[f"{feature}_SurvivalRate"] = np.nan
        
        # OOF encoding
        for train_idx, val_idx in cv_splits:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            for feature in features:
                # Calculate encoding on training fold
                encoding_stats = (
                    pd.DataFrame({"feature": X_train[feature], "target": y_train})
                    .groupby("feature")["target"]
                    .agg(["count", "sum"])
                    .reset_index()
                )
                
                # Apply smoothing
                encoding_stats["smoothed_rate"] = self._smooth_encode(
                    encoding_stats["count"], 
                    encoding_stats["sum"], 
                    self.global_mean_
                )
                
                # Create mapping
                encoding_map = dict(zip(encoding_stats["feature"], encoding_stats["smoothed_rate"]))
                
                # Apply to validation fold
                encoded_feature = f"{feature}_SurvivalRate"
                X_encoded.loc[val_idx, encoded_feature] = (
                    X.loc[val_idx, feature].map(encoding_map).fillna(self.global_mean_)
                )
        
        return X_encoded
    
    def fit_full_data(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> None:
        """
        Fit encoder on full dataset for test encoding.
        
        Args:
            X: Feature dataframe
            y: Target series
            features: List of feature names to encode
        """
        self.global_mean_ = y.mean()
        self.encoding_maps_ = {}
        
        for feature in features:
            encoding_stats = (
                pd.DataFrame({"feature": X[feature], "target": y})
                .groupby("feature")["target"]
                .agg(["count", "sum"])
                .reset_index()
            )
            
            encoding_stats["smoothed_rate"] = self._smooth_encode(
                encoding_stats["count"], 
                encoding_stats["sum"], 
                self.global_mean_
            )
            
            self.encoding_maps_[feature] = dict(
                zip(encoding_stats["feature"], encoding_stats["smoothed_rate"])
            )
    
    def transform(self, X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Transform using fitted encodings.
        
        Args:
            X: Feature dataframe
            features: List of feature names to encode
            
        Returns:
            DataFrame with encoded features
        """
        if self.global_mean_ is None:
            raise ValueError("Encoder must be fitted before transform")
        
        X_encoded = X.copy()
        
        for feature in features:
            if feature in self.encoding_maps_:
                encoded_feature = f"{feature}_SurvivalRate"
                X_encoded[encoded_feature] = (
                    X[feature].map(self.encoding_maps_[feature]).fillna(self.global_mean_)
                )
        
        return X_encoded
    
    def validate_no_leakage(
        self, 
        X_original: pd.DataFrame, 
        y: pd.Series, 
        X_encoded: pd.DataFrame, 
        features: List[str],
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """
        Validate that no leakage occurred in OOF encoding.
        
        Returns True if no leakage detected, False otherwise.
        """
        for feature in features:
            encoded_feature = f"{feature}_SurvivalRate"
            
            if encoded_feature not in X_encoded.columns:
                continue
            
            # For each fold, check that validation encodings don't match full-data encodings
            for train_idx, val_idx in cv_splits:
                # Full data encoding for validation indices
                full_encoding_stats = (
                    pd.DataFrame({"feature": X_original[feature], "target": y})
                    .groupby("feature")["target"]
                    .agg(["count", "sum"])
                    .reset_index()
                )
                
                full_encoding_stats["smoothed_rate"] = self._smooth_encode(
                    full_encoding_stats["count"], 
                    full_encoding_stats["sum"], 
                    y.mean()
                )
                
                full_encoding_map = dict(
                    zip(full_encoding_stats["feature"], full_encoding_stats["smoothed_rate"])
                )
                
                # Check validation indices
                for idx in val_idx:
                    feature_value = X_original.loc[idx, feature]
                    oof_encoded_value = X_encoded.loc[idx, encoded_feature]
                    full_encoded_value = full_encoding_map.get(feature_value, y.mean())
                    
                    # They should be different (unless it's the global mean fallback)
                    if (
                        not pd.isna(oof_encoded_value) and 
                        not pd.isna(full_encoded_value) and
                        abs(oof_encoded_value - full_encoded_value) < 1e-10 and
                        abs(full_encoded_value - y.mean()) > 1e-10
                    ):
                        print(f"Potential leakage detected in {feature} at index {idx}")
                        return False
        
        return True