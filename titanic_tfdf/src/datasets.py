"""Data loading and preprocessing utilities."""

import os
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from typing import Dict, Optional, Tuple


def load_train_test(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test datasets."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def fit_imputers(train_df: pd.DataFrame) -> Dict:
    """Fit imputers on training data."""
    imputers = {}
    
    # Numeric columns - use median
    numeric_cols = ['Age', 'Fare']
    for col in numeric_cols:
        if col in train_df.columns:
            imputer = SimpleImputer(strategy='median')
            imputer.fit(train_df[[col]])
            imputers[col] = imputer
    
    # Categorical columns - use most frequent (mode)
    categorical_cols = ['Embarked']
    for col in categorical_cols:
        if col in train_df.columns:
            imputer = SimpleImputer(strategy='most_frequent')
            imputer.fit(train_df[[col]])
            imputers[col] = imputer
    
    # For missing categorical features, we'll use 'UNK' as default
    # This is handled in the feature extraction functions
    
    return imputers


def apply_imputers(df: pd.DataFrame, imputers: Dict) -> pd.DataFrame:
    """Apply fitted imputers to dataframe."""
    df = df.copy()
    
    for col, imputer in imputers.items():
        if col in df.columns:
            df[col] = imputer.transform(df[[col]]).ravel()
    
    return df


def to_tf_dataset(df: pd.DataFrame, label_col: Optional[str] = None, batch_size: int = 1024) -> tf.data.Dataset:
    """Convert pandas DataFrame to TensorFlow Dataset."""
    
    # Prepare feature columns
    feature_cols = [col for col in df.columns if col != label_col and col not in ['PassengerId']]
    
    # Create feature dictionary
    features = {}
    for col in feature_cols:
        if df[col].dtype == 'object':
            # String features
            features[col] = df[col].fillna('UNK').astype(str).values
        else:
            # Numeric features
            features[col] = df[col].fillna(0.0).astype('float32').values
    
    if label_col and label_col in df.columns:
        # Training dataset with labels
        labels = df[label_col].astype('int32').values
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        # Inference dataset without labels
        dataset = tf.data.Dataset.from_tensor_slices(features)
    
    return dataset.batch(batch_size)