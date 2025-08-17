"""Family wall validation to ensure no family leakage across folds."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.utils import create_family_groups, stratified_group_split


def check_family_wall(df: pd.DataFrame, cv_splits: List[Tuple[np.ndarray, np.ndarray]], 
                     family_groups: pd.Series) -> Dict[str, Any]:
    """
    Verify that family groups don't span across train/validation folds.
    """
    print("Checking family wall integrity...")
    
    violations = []
    fold_family_stats = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Get families in train and validation sets
        train_families = set(family_groups.iloc[train_idx].unique())
        val_families = set(family_groups.iloc[val_idx].unique())
        
        # Check for overlap
        overlap = train_families.intersection(val_families)
        
        if overlap:
            violations.append({
                "fold": fold_idx,
                "overlapping_families": list(overlap),
                "n_overlap": len(overlap)
            })
        
        fold_stats = {
            "fold": fold_idx,
            "n_train_families": len(train_families),
            "n_val_families": len(val_families),
            "n_overlap": len(overlap),
            "train_size": len(train_idx),
            "val_size": len(val_idx)
        }
        fold_family_stats.append(fold_stats)
    
    # Overall statistics
    unique_families = family_groups.nunique()
    total_violations = sum(len(v["overlapping_families"]) for v in violations)
    
    results = {
        "wall_intact": len(violations) == 0,
        "total_families": unique_families,
        "total_violations": total_violations,
        "violations": violations,
        "fold_stats": fold_family_stats
    }
    
    if results["wall_intact"]:
        print(f"✅ Family wall intact: {unique_families} families, no cross-fold leakage")
    else:
        print(f"⚠️  Family wall breached: {total_violations} violations across {len(violations)} folds")
        for violation in violations:
            print(f"   Fold {violation['fold']}: {violation['n_overlap']} families overlap")
    
    return results


def run_family_wall_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete family wall validation."""
    from modeling.train_cv import TrainingPipeline
    
    pipeline = TrainingPipeline(config)
    
    # Load and engineer features
    df = pipeline.load_and_validate_data()
    df = pipeline.engineer_features(df)
    
    # Create family groups
    family_groups = create_family_groups(df)
    
    # Setup CV splits
    cv_splits = stratified_group_split(
        df, 
        family_groups,
        target_col="Survived",
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"]
    )
    
    # Check family wall
    results = check_family_wall(df, cv_splits, family_groups)
    
    # Additional family statistics
    family_sizes = family_groups.value_counts()
    results.update({
        "avg_family_size": family_sizes.mean(),
        "max_family_size": family_sizes.max(),
        "single_member_families": (family_sizes == 1).sum(),
        "large_families": (family_sizes >= 5).sum()
    })
    
    print(f"Family statistics: {family_sizes.describe()}")
    
    return results


if __name__ == "__main__":
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    results = run_family_wall_check(config)
    print(f"Wall check results: {results['wall_intact']}")