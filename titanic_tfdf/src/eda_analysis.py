"""Comprehensive EDA and Missing Value Analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def analyze_missing_patterns(df: pd.DataFrame, title: str = "Dataset") -> Dict:
    """Comprehensive missing value analysis."""
    print(f"\n{'='*60}")
    print(f"MISSING VALUE ANALYSIS - {title}")
    print(f"{'='*60}")
    
    missing_stats = {}
    total_rows = len(df)
    
    print(f"Total rows: {total_rows}")
    print(f"\nMissing value summary:")
    print("-" * 40)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows) * 100
        missing_stats[col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
        
        if missing_count > 0:
            print(f"{col:15s}: {missing_count:4d} ({missing_pct:5.1f}%)")
    
    # Missing value patterns
    print(f"\nMissing value patterns:")
    print("-" * 40)
    missing_pattern = df.isnull().sum(axis=1).value_counts().sort_index()
    for pattern, count in missing_pattern.items():
        if pattern > 0:
            pct = (count / total_rows) * 100
            print(f"Rows with {pattern:2d} missing values: {count:4d} ({pct:5.1f}%)")
    
    return missing_stats


def plot_missing_heatmap(df: pd.DataFrame, title: str = "Missing Values Heatmap"):
    """Visualize missing value patterns."""
    plt.figure(figsize=(12, 8))
    
    # Missing values heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
    plt.title(f"{title}")
    plt.xticks(rotation=45)
    
    # Missing values bar plot
    plt.subplot(2, 2, 2)
    missing_counts = df.isnull().sum()
    missing_counts[missing_counts > 0].plot(kind='bar', color='coral')
    plt.title("Missing Values Count")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    
    # Missing values percentage
    plt.subplot(2, 2, 3)
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct[missing_pct > 0].plot(kind='bar', color='lightblue')
    plt.title("Missing Values Percentage")
    plt.xticks(rotation=45)
    plt.ylabel("Percentage (%)")
    
    # Correlation of missing values
    plt.subplot(2, 2, 4)
    missing_corr = df.isnull().corr()
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Missing Values Correlation")
    
    plt.tight_layout()
    plt.show()


def analyze_feature_distributions(df: pd.DataFrame, target_col: str = 'Survived'):
    """Analyze feature distributions and relationships with target."""
    print(f"\n{'='*60}")
    print(f"FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")
    
    # Target distribution
    if target_col in df.columns:
        print(f"\nTarget distribution ({target_col}):")
        target_dist = df[target_col].value_counts()
        for val, count in target_dist.items():
            pct = (count / len(df)) * 100
            print(f"  {val}: {count} ({pct:.1f}%)")
    
    return numeric_cols, categorical_cols


def plot_feature_distributions(df: pd.DataFrame, numeric_cols: List[str], 
                              categorical_cols: List[str], target_col: str = 'Survived'):
    """Create comprehensive distribution plots."""
    
    # Numeric features distribution
    if numeric_cols:
        n_numeric = len(numeric_cols)
        fig, axes = plt.subplots(nrows=(n_numeric + 1) // 2, ncols=2, 
                                figsize=(15, 4 * ((n_numeric + 1) // 2)))
        if n_numeric == 1:
            axes = [axes]
        elif (n_numeric + 1) // 2 == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            row = i // 2
            col_idx = i % 2
            
            if (n_numeric + 1) // 2 == 1:
                ax = axes[col_idx]
            else:
                ax = axes[row, col_idx]
            
            # Distribution by target
            if target_col in df.columns:
                for target_val in df[target_col].unique():
                    if not pd.isna(target_val):
                        subset = df[df[target_col] == target_val][col].dropna()
                        ax.hist(subset, alpha=0.6, label=f'{target_col}={target_val}', bins=20)
                ax.legend()
            else:
                ax.hist(df[col].dropna(), bins=20, alpha=0.7)
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        # Hide empty subplot if odd number of features
        if n_numeric % 2 == 1 and n_numeric > 1:
            if (n_numeric + 1) // 2 == 1:
                axes[-1].set_visible(False)
            else:
                axes[n_numeric // 2, 1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # Categorical features distribution
    if categorical_cols:
        n_categorical = len(categorical_cols)
        fig, axes = plt.subplots(nrows=(n_categorical + 1) // 2, ncols=2, 
                                figsize=(15, 4 * ((n_categorical + 1) // 2)))
        if n_categorical == 1:
            axes = [axes]
        elif (n_categorical + 1) // 2 == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols):
            row = i // 2
            col_idx = i % 2
            
            if (n_categorical + 1) // 2 == 1:
                ax = axes[col_idx]
            else:
                ax = axes[row, col_idx]
            
            # Count plot by target
            if target_col in df.columns:
                cross_tab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
                cross_tab.plot(kind='bar', ax=ax, stacked=True)
                ax.set_title(f'{col} vs {target_col} (%)')
            else:
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'{col} Distribution')
            
            ax.set_xlabel(col)
            ax.set_ylabel('Percentage' if target_col in df.columns else 'Count')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Hide empty subplot if odd number of features
        if n_categorical % 2 == 1 and n_categorical > 1:
            if (n_categorical + 1) // 2 == 1:
                axes[-1].set_visible(False)
            else:
                axes[n_categorical // 2, 1].set_visible(False)
        
        plt.tight_layout()
        plt.show()


def correlation_analysis(df: pd.DataFrame, target_col: str = 'Survived'):
    """Analyze correlations between features and target."""
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Numeric correlations
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Target correlations
        if target_col in correlation_matrix.columns:
            target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
            print(f"\nCorrelations with {target_col}:")
            print("-" * 30)
            for feature, corr in target_corr.items():
                if feature != target_col:
                    print(f"{feature:15s}: {corr:6.3f}")


def run_comprehensive_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Run complete EDA analysis."""
    
    print("🔍 COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Basic info
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Missing value analysis
    train_missing = analyze_missing_patterns(train_df, "Training Set")
    test_missing = analyze_missing_patterns(test_df, "Test Set")
    
    # Feature distributions
    numeric_cols, categorical_cols = analyze_feature_distributions(train_df)
    
    # Correlation analysis
    correlation_analysis(train_df)
    
    # Visualizations
    print(f"\n📊 Generating visualizations...")
    plot_missing_heatmap(train_df, "Training Set - Missing Values")
    plot_feature_distributions(train_df, numeric_cols, categorical_cols)
    
    return {
        'train_missing': train_missing,
        'test_missing': test_missing,
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols
    }


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Run EDA
    eda_results = run_comprehensive_eda(train_df, test_df)