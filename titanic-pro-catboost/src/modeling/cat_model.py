"""CatBoost model factory and utilities."""

import catboost as cb
from typing import Dict, Any, List, Optional
import numpy as np


class CatBoostModelFactory:
    """Factory for creating CatBoost models with config."""
    
    @staticmethod
    def create_model(config: Dict[str, Any], cat_features: Optional[List[str]] = None) -> cb.CatBoostClassifier:
        """
        Create CatBoost classifier from config.
        
        Args:
            config: CatBoost configuration dictionary
            cat_features: List of categorical feature names
            
        Returns:
            Configured CatBoostClassifier
        """
        # Base parameters
        params = {
            "iterations": config.get("iterations", 6000),
            "learning_rate": config.get("learning_rate", 0.035),
            "depth": config.get("depth", 6),
            "l2_leaf_reg": config.get("l2_leaf_reg", 8),
            "bagging_temperature": config.get("bagging_temperature", 0.2),
            "random_strength": config.get("random_strength", 0.2),
            "border_count": config.get("border_count", 254),
            "loss_function": config.get("loss_function", "Logloss"),
            "eval_metric": config.get("eval_metric", "Logloss"),
            "verbose": config.get("verbose", 200),
            "allow_writing_files": config.get("allow_writing_files", False),
            "grow_policy": config.get("grow_policy", "SymmetricTree"),
            "od_type": "Iter",
            "od_wait": config.get("od_wait", 600),
            "random_seed": config.get("random_seed", 42),
        }
        
        # Optional parameters
        if config.get("auto_class_weights"):
            params["auto_class_weights"] = config["auto_class_weights"]
        
        # Early stopping parameters for validation
        if "early_stopping_rounds" in config:
            params["early_stopping_rounds"] = config["early_stopping_rounds"]
        
        model = cb.CatBoostClassifier(**params)
        
        # Set categorical features if provided
        if cat_features:
            model.set_params(cat_features=cat_features)
        
        return model
    
    @staticmethod
    def fit_with_eval(
        model: cb.CatBoostClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[List[str]] = None,
        verbose: bool = True
    ) -> cb.CatBoostClassifier:
        """
        Fit model with optional validation set.
        
        Args:
            model: CatBoost model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cat_features: Categorical feature names
            verbose: Whether to print training progress
            
        Returns:
            Fitted model
        """
        fit_params = {
            "X": X_train,
            "y": y_train,
            "verbose": verbose,
        }
        
        if cat_features:
            fit_params["cat_features"] = cat_features
        
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = (X_val, y_val)
            fit_params["use_best_model"] = True
        
        model.fit(**fit_params)
        return model
    
    @staticmethod
    def get_feature_importance(
        model: cb.CatBoostClassifier, 
        feature_names: List[str],
        importance_type: str = "PredictionValuesChange"
    ) -> Dict[str, float]:
        """
        Get feature importance from fitted model.
        
        Args:
            model: Fitted CatBoost model
            feature_names: List of feature names
            importance_type: Type of importance to compute
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not model.is_fitted():
            raise ValueError("Model must be fitted before getting feature importance")
        
        importances = model.get_feature_importance(type=importance_type)
        
        return dict(zip(feature_names, importances))
    
    @staticmethod
    def predict_with_threshold(
        model: cb.CatBoostClassifier,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with custom threshold.
        
        Args:
            model: Fitted CatBoost model
            X: Features to predict
            threshold: Classification threshold
            
        Returns:
            Tuple of (probabilities, predictions)
        """
        if not model.is_fitted():
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return probabilities, predictions
    
    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        metric: str = "accuracy"
    ) -> tuple[float, float]:
        """
        Find optimal threshold for given metric.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metric_functions = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
        }
        
        if metric not in metric_functions:
            raise ValueError(f"Unsupported metric: {metric}")
        
        metric_func = metric_functions[metric]
        
        # Test thresholds from 0.1 to 0.9 in steps of 0.01
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            try:
                score = metric_func(y_true, y_pred)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except:
                # Skip invalid thresholds (e.g., all same class)
                continue
        
        return best_threshold, best_score