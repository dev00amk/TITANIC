"""Inference script for ensemble prediction."""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.datasets import load_train_test, apply_imputers
from src.features import add_derived_features


def main():
    parser = argparse.ArgumentParser(description='Run ensemble inference')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Models directory')
    parser.add_argument('--seeds', type=str, default='13,37,71,97,123', help='Comma-separated seeds')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"Using ensemble of {len(seeds)} models with seeds: {seeds}")
    
    # Load test data
    print(f"Loading test data from {args.data_dir}")
    _, test_df = load_train_test(args.data_dir)
    
    # Store original PassengerId for output
    passenger_ids = test_df['PassengerId'].copy()
    
    # Load preprocessing artifacts from first model (they should be identical across seeds)
    first_model_dir = os.path.join(args.models_dir, f"run_{seeds[0]}")
    
    print("Loading preprocessing artifacts")
    with open(os.path.join(first_model_dir, "imputers.pkl"), "rb") as f:
        imputers = pickle.load(f)
    
    with open(os.path.join(first_model_dir, "name_vocab.pkl"), "rb") as f:
        name_vocab = pickle.load(f)
    
    with open(os.path.join(first_model_dir, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    
    # Preprocess test data
    print("Preprocessing test data")
    test_df, _ = add_derived_features(test_df, vocab=name_vocab)
    test_df = apply_imputers(test_df, imputers)
    
    # Prepare features for sklearn
    feature_cols = [col for col in test_df.columns if col not in ['PassengerId']]
    X_test = test_df[feature_cols].copy()
    
    # Apply label encoders
    for col in X_test.columns:
        if col in label_encoders:
            le = label_encoders[col]
            X_test[col] = X_test[col].astype(str)
            # Handle unseen categories
            X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            X_test[col] = le.transform(X_test[col])
    
    # Load models and make predictions
    all_predictions = []
    
    for seed in seeds:
        model_dir = os.path.join(args.models_dir, f"run_{seed}")
        model_path = os.path.join(model_dir, "model.pkl")
        
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        print(f"Making predictions with seed {seed} model")
        # Get probability of positive class (class 1)
        probs = model.predict_proba(X_test)[:, 1]
        
        all_predictions.append(probs)
    
    # Ensemble predictions by averaging probabilities
    print("Ensembling predictions")
    ensemble_probs = np.mean(all_predictions, axis=0)
    
    # Apply threshold to get binary predictions
    ensemble_predictions = (ensemble_probs > 0.5).astype(int)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': ensemble_predictions
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save submission
    print(f"Saving submission to {args.output}")
    submission_df.to_csv(args.output, index=False)
    
    print(f"Submission saved with {len(submission_df)} predictions")
    print(f"Survival rate: {ensemble_predictions.mean():.3f}")


if __name__ == "__main__":
    main()