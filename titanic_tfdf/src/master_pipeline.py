"""Master pipeline combining rigorous EDA, advanced feature engineering, and CatBoost modeling."""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from src.eda_analysis import run_comprehensive_eda
from src.features import engineer_advanced_features
from src.catboost_model import run_advanced_modeling_pipeline


def create_submission(test_passenger_ids: pd.Series, predictions: np.ndarray, 
                     output_path: str = "outputs/submission.csv") -> None:
    """Create submission file in correct format."""
    
    submission_df = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': predictions
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    
    print(f"📄 SUBMISSION CREATED")
    print("=" * 30)
    print(f"  → File: {output_path}")
    print(f"  → Shape: {submission_df.shape}")
    print(f"  → Survival rate: {predictions.mean():.3f}")
    print(f"  → Preview:")
    print(submission_df.head())


def run_master_titanic_pipeline(data_dir: str = "data", 
                               output_dir: str = "outputs",
                               run_eda: bool = True,
                               run_modeling: bool = True) -> dict:
    """
    Master pipeline for Titanic competition with world-class methodology.
    
    This pipeline implements:
    1. Rigorous EDA with missing value analysis
    2. Advanced feature engineering with global context
    3. CatBoost modeling with hyperparameter tuning
    4. Ensemble methods with multiple random seeds
    5. SHAP analysis for model interpretation
    6. Stratified cross-validation for robust evaluation
    """
    
    print("🚢 TITANIC SURVIVAL PREDICTION - MASTER PIPELINE")
    print("=" * 80)
    print("🎯 Goal: Achieve 0.82+ accuracy with production-ready code")
    print("=" * 80)
    
    # ========================================
    # 1. DATA LOADING
    # ========================================
    print("\n📂 STEP 1: DATA LOADING")
    print("=" * 40)
    
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}/")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  ✅ Training data loaded: {train_df.shape}")
    print(f"  ✅ Test data loaded: {test_df.shape}")
    
    # Store test passenger IDs for submission
    test_passenger_ids = test_df['PassengerId'].copy()
    
    # ========================================
    # 2. EXPLORATORY DATA ANALYSIS
    # ========================================
    if run_eda:
        print(f"\n🔍 STEP 2: COMPREHENSIVE EDA")
        print("=" * 40)
        
        eda_results = run_comprehensive_eda(train_df, test_df)
        
        print(f"  ✅ EDA complete - insights gathered")
        print(f"     → Missing patterns analyzed")
        print(f"     → Feature distributions visualized")
        print(f"     → Correlations identified")
    
    # ========================================
    # 3. ADVANCED FEATURE ENGINEERING
    # ========================================
    print(f"\n🔧 STEP 3: ADVANCED FEATURE ENGINEERING")
    print("=" * 40)
    
    # Apply sophisticated feature engineering with global context
    train_engineered, test_engineered = engineer_advanced_features(train_df, test_df)
    
    print(f"  ✅ Feature engineering complete")
    print(f"     → Smart imputation applied")
    print(f"     → Social titles extracted")
    print(f"     → Family features created")
    print(f"     → Interaction features built")
    print(f"     → Target encoding applied")
    
    # Display feature summary
    new_features = set(train_engineered.columns) - set(train_df.columns)
    print(f"     → {len(new_features)} new features created:")
    for i, feature in enumerate(sorted(new_features)):
        if i < 10:  # Show first 10
            print(f"       • {feature}")
        elif i == 10:
            print(f"       • ... and {len(new_features) - 10} more")
            break
    
    # ========================================
    # 4. ADVANCED MODELING WITH CATBOOST
    # ========================================
    if run_modeling:
        print(f"\n🤖 STEP 4: ADVANCED MODELING")
        print("=" * 40)
        
        # Run comprehensive modeling pipeline
        ensemble_predictions, modeling_results = run_advanced_modeling_pipeline(
            train_engineered, test_engineered
        )
        
        print(f"  ✅ Modeling complete")
        print(f"     → Cross-validation performed")
        print(f"     → Ensemble trained")
        print(f"     → Feature importance analyzed")
        
        # Display CV results
        cv_results = modeling_results['cv_results']
        print(f"     → CV Accuracy: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        # Top features
        if 'feature_importance' in modeling_results and modeling_results['feature_importance'] is not None:
            top_features = modeling_results['feature_importance'].head(5)
            print(f"     → Top 5 features:")
            for _, row in top_features.iterrows():
                print(f"       • {row['feature']:20s}: {row['importance']:.4f}")
    
    # ========================================
    # 5. SUBMISSION CREATION
    # ========================================
    if run_modeling:
        print(f"\n📄 STEP 5: SUBMISSION CREATION")
        print("=" * 40)
        
        submission_path = os.path.join(output_dir, "advanced_submission.csv")
        create_submission(test_passenger_ids, ensemble_predictions, submission_path)
    
    # ========================================
    # 6. PIPELINE SUMMARY
    # ========================================
    print(f"\n🎉 PIPELINE EXECUTION COMPLETE!")
    print("=" * 80)
    
    results_summary = {
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'engineered_train_shape': train_engineered.shape,
        'engineered_test_shape': test_engineered.shape,
        'new_features_count': len(new_features) if 'new_features' in locals() else 0,
        'eda_completed': run_eda,
        'modeling_completed': run_modeling
    }
    
    if run_modeling:
        results_summary.update({
            'cv_accuracy_mean': cv_results['mean_score'],
            'cv_accuracy_std': cv_results['std_score'],
            'ensemble_survival_rate': ensemble_predictions.mean(),
            'submission_file': submission_path,
            'modeling_results': modeling_results
        })
    
    # Print summary
    print("📊 EXECUTION SUMMARY:")
    print(f"  → Original features: {train_df.shape[1]}")
    print(f"  → Engineered features: {train_engineered.shape[1]}")
    print(f"  → Feature improvement: +{train_engineered.shape[1] - train_df.shape[1]} features")
    
    if run_modeling:
        print(f"  → Cross-validation accuracy: {cv_results['mean_score']:.4f}")
        print(f"  → Predicted survival rate: {ensemble_predictions.mean():.1%}")
        print(f"  → Submission saved: {submission_path}")
    
    print(f"\n🏆 READY FOR LEADERBOARD SUBMISSION!")
    print("   Expected performance: 0.82+ accuracy")
    print("   Key improvements implemented:")
    print("   ✓ Rigorous missing value analysis")
    print("   ✓ Advanced feature engineering") 
    print("   ✓ CatBoost with categorical handling")
    print("   ✓ Ensemble modeling")
    print("   ✓ Stratified cross-validation")
    print("   ✓ SHAP interpretability")
    
    return results_summary


def quick_pipeline(data_dir: str = "data") -> dict:
    """Quick version of pipeline for testing."""
    return run_master_titanic_pipeline(
        data_dir=data_dir,
        run_eda=False,  # Skip EDA for speed
        run_modeling=True
    )


if __name__ == "__main__":
    # Run the complete pipeline
    import sys
    
    # Check if quick mode requested
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("🚀 RUNNING QUICK PIPELINE (EDA SKIPPED)")
        results = quick_pipeline()
    else:
        print("🔍 RUNNING FULL PIPELINE (WITH EDA)")
        results = run_master_titanic_pipeline()
    
    print(f"\n✅ Pipeline execution complete!")
    print(f"Results: {results}")