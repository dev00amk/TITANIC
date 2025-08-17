"""Unit tests for feature extraction functions."""

import pytest
import pandas as pd
from src.features import (
    extract_title, ticket_prefix, cabin_prefix, tokenize_name,
    build_name_vocab, name_bow, add_derived_features
)


def test_extract_title():
    """Test title extraction from names."""
    assert extract_title("Smith, Mr. John") == "Mr"
    assert extract_title("Johnson, Mrs. Mary") == "Mrs"
    assert extract_title("Brown, Miss. Sarah") == "Miss"
    assert extract_title("Wilson, Master. Tommy") == "Master"
    assert extract_title("Davis, Dr. Robert") == "Rare"
    assert extract_title("Miller, Rev. William") == "Rare"
    assert extract_title("NoTitleName") == "Rare"


def test_ticket_prefix():
    """Test ticket prefix extraction."""
    assert ticket_prefix("A/5 21171") == "A/"
    assert ticket_prefix("PC 17599") == "PC"
    assert ticket_prefix("STON/O2. 3101282") == "STON/O."
    assert ticket_prefix("113803") == "NUMERIC"
    assert ticket_prefix(None) == "UNK"
    assert ticket_prefix(pd.NA) == "UNK"


def test_cabin_prefix():
    """Test cabin prefix extraction."""
    assert cabin_prefix("C85") == "C"
    assert cabin_prefix("C123 C456") == "C"
    assert cabin_prefix("E46") == "E"
    assert cabin_prefix(None) == "UNK"
    assert cabin_prefix(pd.NA) == "UNK"
    assert cabin_prefix("123") == "UNK"


def test_tokenize_name():
    """Test name tokenization."""
    tokens = tokenize_name("Smith, Mr. John William")
    assert "smith" in tokens
    assert "john" in tokens
    assert "william" in tokens
    assert "mr" not in tokens  # Should filter out titles
    
    tokens = tokenize_name("Johnson-Brown, Mrs. Mary Elizabeth")
    assert "johnson" in tokens
    assert "brown" in tokens
    assert "mary" in tokens
    assert "elizabeth" in tokens
    
    assert tokenize_name(None) == []
    assert tokenize_name("") == []


def test_build_name_vocab():
    """Test vocabulary building."""
    names = pd.Series([
        "Smith, Mr. John",
        "Smith, Mrs. Mary",
        "Johnson, Miss. Sarah",
        "Brown, Master. Tommy"
    ])
    
    vocab = build_name_vocab(names, k=5)
    assert "smith" in vocab  # Most frequent
    assert len(vocab) <= 5


def test_name_bow():
    """Test name bag-of-words conversion."""
    vocab = {"john", "smith", "mary"}
    
    bow = name_bow("Smith, Mr. John William", vocab)
    assert "john" in bow
    assert "smith" in bow
    assert "william" not in bow  # Not in vocab
    
    bow = name_bow("Johnson, Mrs. Mary", vocab)
    assert "mary" in bow
    assert "johnson" not in bow  # Not in vocab


def test_add_derived_features():
    """Test adding all derived features to dataframe."""
    # Create sample dataframe
    df = pd.DataFrame({
        'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary'],
        'Ticket': ['A123', '456789'],
        'Cabin': ['C85', None],
        'SibSp': [1, 0],
        'Parch': [0, 1]
    })
    
    result_df, vocab = add_derived_features(df)
    
    # Check that all new columns exist
    expected_cols = ['Title', 'TicketPrefix', 'CabinPrefix', 'FamilySize', 'IsAlone', 'NameBOW']
    for col in expected_cols:
        assert col in result_df.columns
    
    # Check some specific values
    assert result_df.loc[0, 'Title'] == 'Mr'
    assert result_df.loc[1, 'Title'] == 'Mrs'
    assert result_df.loc[0, 'FamilySize'] == 2  # 1 + 0 + 1
    assert result_df.loc[1, 'FamilySize'] == 2  # 0 + 1 + 1
    assert result_df.loc[0, 'IsAlone'] == 0
    assert result_df.loc[1, 'IsAlone'] == 0
    
    # Check vocab is returned
    assert isinstance(vocab, set)
    assert len(vocab) > 0


def test_add_derived_features_with_existing_vocab():
    """Test using existing vocabulary."""
    df = pd.DataFrame({
        'Name': ['Smith, Mr. John'],
        'Ticket': ['A123'],
        'Cabin': ['C85'],
        'SibSp': [0],
        'Parch': [0]
    })
    
    existing_vocab = {'john', 'smith'}
    result_df, returned_vocab = add_derived_features(df, vocab=existing_vocab)
    
    # Should return the same vocab that was passed in
    assert returned_vocab == existing_vocab