"""Training script for Gradient Boosting model."""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from src.datasets import load_train_test, fit_imputers, apply_imputers
from src.features import add_derived_features


def prepare_features_for_sklearn(df):
    """Prepare features for sklearn training."""
    # Exclude non-feature columns
    feature_cols = [col for col in df.columns if col not in ['PassengerId', 'Survived']]
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Encode categorical features
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    return X, label_encoders


def main():
    parser = argparse.ArgumentParser(description='Train Gradient Boosting model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model')
    parser.add_argument('--seed', type=int, default=13, help='Random seed')
    parser.add_argument('--trees', type=int, default=2000, help='Number of trees')
    parser.add_argument('--shrinkage', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--min_examples', type=int, default=1, help='Min examples in leaf')
    parser.add_argument('--sparse_oblique', type=str, default='true', help='Use sparse oblique splits')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}")
    train_df, _ = load_train_test(args.data_dir)
    
    print("Adding derived features")
    train_df, name_vocab = add_derived_features(train_df)
    
    print("Fitting imputers")
    imputers = fit_imputers(train_df)
    
    print("Applying imputation")
    train_df = apply_imputers(train_df, imputers)
    
    print("Preparing features for sklearn")
    X, label_encoders = prepare_features_for_sklearn(train_df)
    y = train_df['Survived']
    
    print("Creating Gradient Boosting model")
    model = GradientBoostingClassifier(
        n_estimators=args.trees,
        learning_rate=args.shrinkage,
        min_samples_leaf=args.min_examples,
        random_state=args.seed,
        verbose=1
    )
    
    print(f"Training model with {args.trees} trees, seed {args.seed}")
    model.fit(X, y)
    
    print(f"Saving model to {args.output_dir}")
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # Save preprocessing artifacts
    with open(os.path.join(args.output_dir, "imputers.pkl"), "wb") as f:
        pickle.dump(imputers, f)
    
    with open(os.path.join(args.output_dir, "name_vocab.pkl"), "wb") as f:
        pickle.dump(name_vocab, f)
    
    with open(os.path.join(args.output_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    
    # Export feature importances
    try:
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        importance_df.to_csv(os.path.join(args.output_dir, "feature_importances.csv"), index=False)
        print("Feature importances saved")
    except Exception as e:
        print(f"Could not save feature importances: {e}")
    
    print("Training completed successfully")


if __name__ == "__main__":
    main()