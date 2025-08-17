"""CatBoost modeling with advanced techniques and hyperparameter tuning."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  CatBoost not available. Install with: pip install catboost")
    from sklearn.ensemble import GradientBoostingClassifier
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False


class AdvancedTitanicModel:
    """Advanced modeling class with CatBoost and ensemble techniques."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.feature_encoders = {}
        self.categorical_features = []
        self.feature_importances = {}
        self.cv_scores = {}
        
    def identify_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """Identify categorical features for CatBoost."""
        categorical_features = []
        
        # Explicit categorical features
        explicit_cats = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'FamilySizeGroup', 
                        'AgeGroup', 'TicketPrefix', 'Sex_Pclass', 'AgeGroup_Pclass',
                        'Title_Pclass', 'FamilySize_Pclass', 'Embarked_Pclass', 'FareBin', 'AgeBin']
        
        for col in explicit_cats:
            if col in df.columns:
                categorical_features.append(col)
        
        # Auto-detect object type columns and categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col not in categorical_features and col not in ['Name', 'Ticket', 'Cabin']:
                categorical_features.append(col)
        
        # Auto-detect low cardinality numeric features (but exclude survival rates)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['PassengerId', 'Survived', 'Age', 'Fare', 'FarePerPerson', 'NameLength', 'TicketNumber']:
                # Exclude survival rate features (they contain floats)
                if '_SurvivalRate' not in col:
                    unique_vals = df[col].nunique()
                    if unique_vals <= 10:  # Low cardinality
                        categorical_features.append(col)
        
        self.categorical_features = categorical_features
        print(f"Identified categorical features: {categorical_features}")
        return categorical_features
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for modeling."""
        print("PREPARING FEATURES FOR MODELING")
        print("=" * 50)
        
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Remove unnecessary columns
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        
        for df in [train_df, test_df]:
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
        
        # Identify categorical features
        self.identify_categorical_features(train_df)
        
        # Handle categorical encoding if not using CatBoost
        if not CATBOOST_AVAILABLE:
            print("  Encoding categorical features for sklearn...")
            for col in self.categorical_features:
                if col in train_df.columns:
                    le = LabelEncoder()
                    
                    # Combine train and test for consistent encoding
                    combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
                    le.fit(combined_values)
                    
                    train_df[col] = le.transform(train_df[col].astype(str))
                    test_df[col] = le.transform(test_df[col].astype(str))
                    
                    self.feature_encoders[col] = le
        
        print(f"  Training features shape: {train_df.shape}")
        print(f"  Test features shape: {test_df.shape}")
        
        return train_df, test_df
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            n_iter: int = 100, cv_folds: int = 5) -> Dict:
        """Comprehensive hyperparameter tuning."""
        print(f"HYPERPARAMETER TUNING ({n_iter} iterations)")
        print("=" * 50)
        
        if CATBOOST_AVAILABLE:
            # CatBoost parameter space
            param_distributions = {
                'iterations': [500, 750, 1000, 1250, 1500],
                'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
                'depth': [4, 5, 6, 7, 8],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'border_count': [32, 64, 128, 255],
                'bagging_temperature': [0, 0.5, 1.0],
                'random_strength': [0, 0.5, 1.0]
            }
            
            base_model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                eval_metric='Accuracy',
                early_stopping_rounds=50,
                cat_features=self.categorical_features
            )
        else:
            # GradientBoosting parameter space
            param_distributions = {
                'n_estimators': [100, 200, 300, 500, 750, 1000],
                'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            base_model = GradientBoostingClassifier(random_state=self.random_state)
        
        # Stratified CV for hyperparameter search
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        print("  Running randomized search...")
        search.fit(X, y)
        
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"  Best CV score: {best_score:.4f}")
        print(f"  Best parameters: {best_params}")
        
        return best_params
    
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                         params: Optional[Dict] = None) -> object:
        """Train the final model with best parameters."""
        print("TRAINING FINAL MODEL")
        print("=" * 50)
        
        if params is None:
            # Default parameters optimized for Titanic
            if CATBOOST_AVAILABLE:
                params = {
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'border_count': 128,
                    'bagging_temperature': 0.5,
                    'random_strength': 0.5,
                    'eval_metric': 'Accuracy',
                    'random_state': self.random_state,
                    'verbose': False
                }
            else:
                params = {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'subsample': 0.9,
                    'random_state': self.random_state
                }
        
        if CATBOOST_AVAILABLE:
            model = CatBoostClassifier(**params, cat_features=self.categorical_features)
            
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                # Get best iteration and retrain
                best_iter = model.get_best_iteration()
                print(f"  Best iteration: {best_iter}")
                
                final_params = params.copy()
                final_params['iterations'] = best_iter
                final_model = CatBoostClassifier(**final_params, cat_features=self.categorical_features)
                final_model.fit(X_train, y_train, verbose=False)
            else:
                final_model = model
                final_model.fit(X_train, y_train, verbose=False)
        else:
            final_model = GradientBoostingClassifier(**params)
            final_model.fit(X_train, y_train)
        
        self.models['final'] = final_model
        
        # Feature importance
        if hasattr(final_model, 'feature_importances_'):
            feature_names = X_train.columns
            importances = final_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.feature_importances['final'] = feature_importance_df
            
            print("  Top 10 feature importances:")
            for _, row in feature_importance_df.head(10).iterrows():
                print(f"     {row['feature']:20s}: {row['importance']:.4f}")
        
        return final_model
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5, params: Optional[Dict] = None) -> Dict:
        """Perform stratified cross-validation."""
        print(f"STRATIFIED CROSS-VALIDATION ({cv_folds} folds)")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        if params is None:
            # Use default parameters
            if CATBOOST_AVAILABLE:
                model = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    random_state=self.random_state,
                    verbose=False,
                    cat_features=self.categorical_features
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=self.random_state
                )
        else:
            if CATBOOST_AVAILABLE:
                model = CatBoostClassifier(**params, cat_features=self.categorical_features)
            else:
                model = GradientBoostingClassifier(**params)
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        cv_results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores,
            'cv_folds': cv_folds
        }
        
        self.cv_scores['final'] = cv_results
        
        print(f"  CV Accuracy: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"  Individual scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        return cv_results
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      n_models: int = 5) -> List[object]:
        """Train ensemble of models with different random seeds."""
        print(f"TRAINING ENSEMBLE ({n_models} models)")
        print("=" * 50)
        
        ensemble_models = []
        
        for i in range(n_models):
            print(f"  Training model {i+1}/{n_models}...")
            
            if CATBOOST_AVAILABLE:
                model = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3,
                    random_state=self.random_state + i,
                    verbose=False,
                    cat_features=self.categorical_features
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=self.random_state + i
                )
            
            model.fit(X_train, y_train)
            ensemble_models.append(model)
        
        self.models['ensemble'] = ensemble_models
        print(f"  Ensemble training complete!")
        
        return ensemble_models
    
    def predict_ensemble(self, X: pd.DataFrame, models: List[object]) -> np.ndarray:
        """Make ensemble predictions by averaging probabilities."""
        predictions = []
        
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            else:
                pred_proba = model.predict(X)
            predictions.append(pred_proba)
        
        # Average predictions
        ensemble_proba = np.mean(predictions, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        if 'final' not in self.feature_importances:
            print("No feature importance data available.")
            return
        
        importance_df = self.feature_importances['final'].head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def analyze_with_shap(self, X: pd.DataFrame, model: object = None):
        """SHAP analysis for model interpretation."""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return
        
        if model is None:
            if 'final' not in self.models:
                print("No trained model available.")
                return
            model = self.models['final']
        
        print("SHAP ANALYSIS")
        print("=" * 50)
        
        try:
            # Create SHAP explainer
            if CATBOOST_AVAILABLE and isinstance(model, CatBoostClassifier):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use subset for speed)
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")


def run_advanced_modeling_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """Run the complete advanced modeling pipeline."""
    
    print("ADVANCED CATBOOST MODELING PIPELINE")
    print("=" * 70)
    
    # Initialize model
    model_manager = AdvancedTitanicModel()
    
    # Prepare features
    X_train_processed, X_test_processed = model_manager.prepare_features(train_df, test_df)
    
    # Extract target
    y_train = train_df['Survived']
    
    # Remove target from features if present
    if 'Survived' in X_train_processed.columns:
        X_train_processed = X_train_processed.drop('Survived', axis=1)
    
    # Cross-validation
    cv_results = model_manager.cross_validate_model(X_train_processed, y_train)
    
    # Hyperparameter tuning (optional - comment out for speed)
    print("\nSkipping hyperparameter tuning for speed...")
    # best_params = model_manager.hyperparameter_tuning(X_train_processed, y_train, n_iter=50)
    best_params = None
    
    # Train final model
    final_model = model_manager.train_final_model(X_train_processed, y_train, params=best_params)
    
    # Train ensemble
    ensemble_models = model_manager.train_ensemble(X_train_processed, y_train)
    
    # Make predictions
    print("\nMAKING PREDICTIONS")
    print("=" * 50)
    
    # Single model predictions
    if hasattr(final_model, 'predict_proba'):
        single_proba = final_model.predict_proba(X_test_processed)[:, 1]
    else:
        single_proba = final_model.predict(X_test_processed)
    single_pred = (single_proba > 0.5).astype(int)
    
    # Ensemble predictions
    ensemble_pred, ensemble_proba = model_manager.predict_ensemble(X_test_processed, ensemble_models)
    
    print(f"  → Single model predictions: {single_pred}")
    print(f"  → Ensemble predictions: {ensemble_pred}")
    print(f"  → Ensemble survival rate: {ensemble_pred.mean():.3f}")
    
    # Feature importance analysis
    model_manager.plot_feature_importance()
    
    # SHAP analysis
    if len(X_train_processed) <= 1000:  # Only for smaller datasets
        model_manager.analyze_with_shap(X_train_processed[:100])
    
    results = {
        'cv_results': cv_results,
        'single_predictions': single_pred,
        'ensemble_predictions': ensemble_pred,
        'ensemble_probabilities': ensemble_proba,
        'model_manager': model_manager,
        'feature_importance': model_manager.feature_importances.get('final'),
        'processed_features': X_train_processed.columns.tolist()
    }
    
    return ensemble_pred, results


if __name__ == "__main__":
    # Example usage
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Assuming features are already engineered
    predictions, results = run_advanced_modeling_pipeline(train_df, test_df)