"""Data contracts using Pandera schemas."""

import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Dict, Any


def get_raw_train_schema() -> DataFrameSchema:
    """Schema for raw training data."""
    return DataFrameSchema(
        {
            "PassengerId": Column(pa.Int64, Check.ge(1)),
            "Survived": Column(pa.Int64, Check.isin([0, 1])),
            "Pclass": Column(pa.Int64, Check.isin([1, 2, 3])),
            "Name": Column(pa.String, Check.str_length(min_val=1)),
            "Sex": Column(pa.String, Check.isin(["male", "female"])),
            "Age": Column(pa.Float64, Check.ge(0) & Check.le(100), nullable=True),
            "SibSp": Column(pa.Int64, Check.ge(0)),
            "Parch": Column(pa.Int64, Check.ge(0)),
            "Ticket": Column(pa.String, nullable=False),
            "Fare": Column(pa.Float64, Check.ge(0), nullable=True),
            "Cabin": Column(pa.String, nullable=True),
            "Embarked": Column(pa.String, Check.isin(["C", "Q", "S"]), nullable=True),
        },
        strict=True,
        coerce=True,
    )


def get_raw_test_schema() -> DataFrameSchema:
    """Schema for raw test data (no Survived column)."""
    schema_dict = get_raw_train_schema().columns.copy()
    del schema_dict["Survived"]
    return DataFrameSchema(schema_dict, strict=True, coerce=True)


def get_engineered_features_schema() -> DataFrameSchema:
    """Schema for engineered features."""
    return DataFrameSchema(
        {
            "PassengerId": Column(pa.Int64, Check.ge(1)),
            "Pclass": Column(pa.Int64, Check.isin([1, 2, 3])),
            "PclassStr": Column(pa.String, Check.isin(["1", "2", "3"])),
            "Sex": Column(pa.String, Check.isin(["male", "female"])),
            "Age": Column(pa.Float64, Check.ge(0) & Check.le(100)),
            "SibSp": Column(pa.Int64, Check.ge(0)),
            "Parch": Column(pa.Int64, Check.ge(0)),
            "Fare": Column(pa.Float64, Check.ge(0)),
            "Embarked": Column(pa.String, Check.isin(["C", "Q", "S"])),
            "Title": Column(
                pa.String,
                Check.isin([
                    "Mr", "Mrs", "Miss", "Master", 
                    "Nobility", "Officer", "Professional", "Rare"
                ])
            ),
            "CabinDeck": Column(
                pa.String, 
                Check.isin(["Upper", "Middle", "Lower", "Other", "Unknown"])
            ),
            "HasCabin": Column(pa.Int64, Check.isin([0, 1])),
            "TicketPrefix": Column(pa.String, nullable=False),
            "TicketHasLetters": Column(pa.Int64, Check.isin([0, 1])),
            "FamilySize": Column(pa.Int64, Check.ge(1)),
            "IsAlone": Column(pa.Int64, Check.isin([0, 1])),
            "HasSiblings": Column(pa.Int64, Check.isin([0, 1])),
            "HasParents": Column(pa.Int64, Check.isin([0, 1])),
            "HasChildren": Column(pa.Int64, Check.isin([0, 1])),
            "FamilyID": Column(pa.String, nullable=False),
            "AgeGroup": Column(
                pa.String,
                Check.isin(["Child", "Teen", "YoungAdult", "Adult", "Senior", "Elderly"])
            ),
            "AgeBin": Column(pa.String, nullable=False),
            "IsChild": Column(pa.Int64, Check.isin([0, 1])),
            "Sex_Pclass": Column(pa.String, nullable=False),
            "Title_Pclass": Column(pa.String, nullable=False),
            "AgeGroup_Pclass": Column(pa.String, nullable=False),
            "FamilySize_Pclass": Column(pa.String, nullable=False),
            "Embarked_Pclass": Column(pa.String, nullable=False),
        },
        strict=False,  # Allow additional columns for target encoding
        coerce=True,
    )


def validate_dataframe(df: pa.DataFrame, schema: DataFrameSchema) -> pa.DataFrame:
    """Validate dataframe against schema."""
    try:
        validated_df = schema.validate(df, lazy=True)
        return validated_df
    except pa.errors.SchemaErrors as e:
        print(f"Schema validation failed: {e}")
        raise