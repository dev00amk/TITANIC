"""Test script for the advanced pipeline."""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from src.features import engineer_advanced_features
from src.catboost_model import AdvancedTitanicModel

def test_feature_engineering():
    """Test the advanced feature engineering."""
    print("Testing feature engineering...")
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    
    # Apply feature engineering
    train_eng, test_eng = engineer_advanced_features(train_df, test_df)
    
    print(f"Engineered train shape: {train_eng.shape}")
    print(f"Engineered test shape: {test_eng.shape}")
    
    # Show new features
    new_features = set(train_eng.columns) - set(train_df.columns)
    print(f"New features ({len(new_features)}): {sorted(new_features)}")
    
    return train_eng, test_eng

def test_modeling(train_df, test_df):
    """Test the CatBoost modeling."""
    print("\nTesting CatBoost modeling...")
    
    # Initialize model
    model = AdvancedTitanicModel()
    
    # Prepare features
    X_train, X_test = model.prepare_features(train_df, test_df)
    y_train = train_df['Survived']
    
    # Remove target from features
    if 'Survived' in X_train.columns:
        X_train = X_train.drop('Survived', axis=1)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Quick cross-validation
    cv_results = model.cross_validate_model(X_train, y_train, cv_folds=3)
    
    # Train final model
    final_model = model.train_final_model(X_train, y_train)
    
    # Make predictions
    if hasattr(final_model, 'predict_proba'):
        test_proba = final_model.predict_proba(X_test)[:, 1]
    else:
        test_proba = final_model.predict(X_test)
    
    test_pred = (test_proba > 0.5).astype(int)
    
    print(f"Predictions shape: {test_pred.shape}")
    print(f"Survival rate: {test_pred.mean():.3f}")
    
    return test_pred

def create_submission(test_df, predictions):
    """Create submission file."""
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    os.makedirs('outputs', exist_ok=True)
    submission.to_csv('outputs/advanced_submission.csv', index=False)
    
    print(f"\nSubmission created: outputs/advanced_submission.csv")
    print(f"Shape: {submission.shape}")
    print(submission.head())

if __name__ == "__main__":
    print("ADVANCED TITANIC PIPELINE TEST")
    print("=" * 50)
    
    try:
        # Test feature engineering
        train_eng, test_eng = test_feature_engineering()
        
        # Test modeling
        predictions = test_modeling(train_eng, test_eng)
        
        # Create submission
        test_df = pd.read_csv('data/test.csv')
        create_submission(test_df, predictions)
        
        print("\nTEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()