"""Core utilities for the pipeline."""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
import logging


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def create_family_id(df: pd.DataFrame) -> pd.Series:
    """Create family ID from surname and ticket prefix."""
    # Extract surname
    surname = df["Name"].str.extract(r"^([^,]+),", expand=False).str.strip()
    
    # Extract ticket prefix
    ticket_prefix = df["Ticket"].str.extract(r"^([A-Za-z./]+)", expand=False)
    ticket_prefix = ticket_prefix.fillna("")
    
    # Combine surname with ticket prefix, fallback to surname only
    family_id = surname + "_" + ticket_prefix
    family_id = family_id.str.replace("_$", "", regex=True)  # Remove trailing underscore
    
    return family_id


def create_stratified_group_splits(
    df: pd.DataFrame, 
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified CV splits that respect group boundaries.
    
    For small datasets like Titanic, we use StratifiedKFold as base
    and check for group disjointness, preferring group separation
    when conflicts arise.
    """
    # Basic stratified split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(df, df[target_col]))
    
    # Check group disjointness and adjust if needed
    adjusted_splits = []
    groups = df[group_col].values
    
    for train_idx, val_idx in splits:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        
        # Check for overlap
        overlap = train_groups.intersection(val_groups)
        
        if overlap:
            # Move overlapping groups to training set (conservative)
            overlap_mask = df[group_col].isin(overlap)
            overlap_indices = df[overlap_mask].index.values
            
            # Remove overlap from validation set
            val_idx_clean = np.setdiff1d(val_idx, overlap_indices)
            train_idx_expanded = np.union1d(train_idx, overlap_indices)
            
            adjusted_splits.append((train_idx_expanded, val_idx_clean))
        else:
            adjusted_splits.append((train_idx, val_idx))
    
    return adjusted_splits


def save_cv_splits(
    splits: List[Tuple[np.ndarray, np.ndarray]], 
    output_dir: Path,
    seeds: List[int]
) -> None:
    """Save CV splits to disk for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            np.save(seed_dir / f"fold_{fold_idx}_train.npy", train_idx)
            np.save(seed_dir / f"fold_{fold_idx}_val.npy", val_idx)


def load_cv_splits(
    input_dir: Path, 
    seed: int, 
    n_folds: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load CV splits from disk."""
    seed_dir = input_dir / f"seed_{seed}"
    splits = []
    
    for fold_idx in range(n_folds):
        train_idx = np.load(seed_dir / f"fold_{fold_idx}_train.npy")
        val_idx = np.load(seed_dir / f"fold_{fold_idx}_val.npy")
        splits.append((train_idx, val_idx))
    
    return splits


def get_categorical_features(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Identify categorical features for CatBoost."""
    exclude = exclude or ["PassengerId", "Survived"]
    
    categorical_features = []
    
    # String/object columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col not in exclude:
            categorical_features.append(col)
    
    # Low cardinality numeric columns (excluding continuous features)
    numeric_exclude = [
        "Age", "Fare", "SibSp", "Parch", "FamilySize", 
        "PassengerId", "Survived"
    ] + [col for col in df.columns if "SurvivalRate" in col]
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in numeric_exclude and col not in exclude:
            unique_vals = df[col].nunique()
            if unique_vals <= 10:
                categorical_features.append(col)
    
    return sorted(categorical_features)


def calculate_cv_metrics(fold_metrics: List[Dict]) -> Dict[str, float]:
    """Calculate mean and std of CV metrics."""
    metrics = {}
    
    if not fold_metrics:
        return metrics
    
    # Get all metric names
    metric_names = fold_metrics[0].keys()
    
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        metrics[f"{metric_name}_mean"] = np.mean(values)
        metrics[f"{metric_name}_std"] = np.std(values)
        metrics[f"{metric_name}_values"] = values
    
    return metrics