"""Tests for family wall validation."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).parent.parent / "src"))

from validation.family_wall_check import check_family_wall, run_family_wall_check


class TestFamilyWallValidation:
    """Test family wall integrity checking."""
    
    def test_family_wall_intact(self):
        """Test family wall with no violations."""
        # Create sample data with distinct families per fold
        df = pd.DataFrame({
            'PassengerId': range(1, 11),
            'FamilyID': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']
        })
        
        family_groups = pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'])
        
        # Create CV splits that respect family boundaries
        cv_splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),  # Families A,B vs C,D
            (np.array([0, 1, 4, 5]), np.array([2, 3, 6, 7])),  # Families A,C vs B,D
        ]
        
        results = check_family_wall(df, cv_splits, family_groups)
        
        assert results['wall_intact'] == True
        assert results['total_violations'] == 0
        assert len(results['violations']) == 0
    
    def test_family_wall_violation(self):
        """Test family wall with violations."""
        df = pd.DataFrame({
            'PassengerId': range(1, 7),
            'FamilyID': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        
        family_groups = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
        
        # Create CV splits that violate family boundaries
        cv_splits = [
            (np.array([0, 2, 4]), np.array([1, 3, 5])),  # Family A split across folds
        ]
        
        results = check_family_wall(df, cv_splits, family_groups)
        
        assert results['wall_intact'] == False
        assert results['total_violations'] > 0
        assert len(results['violations']) > 0
    
    def test_single_member_families(self):
        """Test handling of single-member families."""
        df = pd.DataFrame({
            'PassengerId': range(1, 6),
            'FamilyID': ['A', 'B', 'C', 'D', 'E']  # All single-member families
        })
        
        family_groups = pd.Series(['A', 'B', 'C', 'D', 'E'])
        
        cv_splits = [
            (np.array([0, 1, 2]), np.array([3, 4])),
        ]
        
        results = check_family_wall(df, cv_splits, family_groups)
        
        # Should always be intact for single-member families
        assert results['wall_intact'] == True
    
    def test_fold_statistics(self):
        """Test fold statistics calculation."""
        df = pd.DataFrame({
            'PassengerId': range(1, 9),
            'FamilyID': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
        })
        
        family_groups = pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'])
        
        cv_splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
        ]
        
        results = check_family_wall(df, cv_splits, family_groups)
        
        fold_stats = results['fold_stats'][0]
        assert fold_stats['n_train_families'] == 2  # Families A, B
        assert fold_stats['n_val_families'] == 2    # Families C, D
        assert fold_stats['train_size'] == 4
        assert fold_stats['val_size'] == 4
    
    def test_edge_cases(self):
        """Test edge cases in family wall checking."""
        # Empty family groups
        empty_df = pd.DataFrame({'PassengerId': []})
        empty_groups = pd.Series([], dtype='object')
        empty_splits = []
        
        # Should handle gracefully
        try:
            results = check_family_wall(empty_df, empty_splits, empty_groups)
            assert results['wall_intact'] == True
        except (IndexError, ValueError):
            # Expected for empty data
            pass
    
    def test_family_wall_integration(self, sample_config):
        """Test integration with full pipeline."""
        # Mock the training pipeline
        with mock.patch('validation.family_wall_check.TrainingPipeline') as mock_pipeline:
            mock_instance = mock_pipeline.return_value
            
            # Create mock data with families
            sample_df = pd.DataFrame({
                'PassengerId': range(1, 11),
                'Survived': [0, 1] * 5,
                'Name': [f"Family_{i//2}, Mr. Person" for i in range(10)],
                'Ticket': [f"TICKET_{i//2}" for i in range(10)]
            })
            
            mock_instance.load_and_validate_data.return_value = sample_df
            mock_instance.engineer_features.return_value = sample_df
            
            # Mock family groups and CV splits
            with mock.patch('validation.family_wall_check.create_family_groups') as mock_families, \
                 mock.patch('validation.family_wall_check.stratified_group_split') as mock_cv:
                
                mock_families.return_value = pd.Series([f"Family_{i//2}" for i in range(10)])
                mock_cv.return_value = [
                    (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
                    (np.array([0, 1, 4, 5]), np.array([2, 3, 8, 9]))
                ]
                
                results = run_family_wall_check(sample_config)
                
                assert 'wall_intact' in results
                assert 'total_families' in results
    
    def test_large_family_handling(self):
        """Test handling of large families."""
        # Create data with one very large family
        df = pd.DataFrame({
            'PassengerId': range(1, 21),
            'FamilyID': ['LargeFamily'] * 15 + ['Small1'] * 2 + ['Small2'] * 2 + ['Single']
        })
        
        family_groups = pd.Series(df['FamilyID'])
        
        # Try to split - large family might cause issues
        cv_splits = [
            (np.array(range(10)), np.array(range(10, 20))),
        ]
        
        results = check_family_wall(df, cv_splits, family_groups)
        
        # Large family should cause violations if split
        if 'LargeFamily' in family_groups.iloc[:10].values and 'LargeFamily' in family_groups.iloc[10:].values:
            assert results['wall_intact'] == False