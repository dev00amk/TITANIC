"""Feature engineering pipeline."""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class FeatureEngineer:
    """Feature engineering pipeline for Titanic dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_config = config.get("features", {})
        self.imputation_values: Dict[str, Any] = {}
        
    def extract_title(self, name: str) -> str:
        """Extract and normalize title from passenger name."""
        if pd.isna(name):
            return "Rare"
        
        title_search = re.search(r" ([A-Za-z]+)\.", str(name))
        if not title_search:
            return "Rare"
        
        title = title_search.group(1)
        
        # Apply title mapping from config
        title_mapping = self.feature_config.get("title_mapping", {})
        
        if title in title_mapping.get("standard", []):
            return title
        elif title in ["Mme", "Ms", "Mlle"]:
            return "Miss" if title in ["Ms", "Mlle"] else "Mrs"
        elif title in title_mapping.get("nobility", []):
            return "Nobility"
        elif title in title_mapping.get("officer", []):
            return "Officer"
        elif title in title_mapping.get("professional", []):
            return "Professional"
        else:
            return "Rare"
    
    def extract_cabin_deck(self, cabin: str) -> str:
        """Extract deck from cabin with luxury tier mapping."""
        if pd.isna(cabin) or cabin == "":
            return "Unknown"
        
        deck = str(cabin).strip()[0].upper()
        if not deck.isalpha():
            return "Unknown"
        
        deck_mapping = self.feature_config.get("deck_mapping", {})
        
        for tier, decks in deck_mapping.items():
            if deck in decks:
                return tier.title()  # Upper, Middle, Lower, Other
        
        return "Other"
    
    def extract_ticket_prefix(self, ticket: str) -> str:
        """Extract prefix from ticket."""
        if pd.isna(ticket):
            return "NUMERIC"
        
        # Remove digits and spaces, keep only letters
        prefix = re.sub(r"[0-9\s/.]", "", str(ticket)).upper()
        
        if not prefix:
            return "NUMERIC"
        
        # Group common prefixes
        prefix_groups = {
            "PC": "FIRST_CLASS",
            "PARIS": "FIRST_CLASS", 
            "SC": "SOUTHAMPTON",
            "SOTON": "SOUTHAMPTON",
            "STON": "SOUTHAMPTON",
            "CA": "CALIFORNIA",
            "CASOTON": "CALIFORNIA",
        }
        
        return prefix_groups.get(prefix, prefix)
    
    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create family-related features."""
        df = df.copy()
        
        # Basic family size
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        
        # Family composition
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasSiblings"] = (df["SibSp"] > 0).astype(int)
        df["HasParents"] = (df["Parch"] > 0).astype(int)
        df["HasChildren"] = (df["Parch"] > 2).astype(int)  # Likely has children
        
        # Family ID for grouping
        surname = df["Name"].str.extract(r"^([^,]+),", expand=False).str.strip()
        ticket_prefix = df["Ticket"].str.extract(r"^([A-Za-z./]+)", expand=False)
        ticket_prefix = ticket_prefix.fillna("")
        
        df["FamilyID"] = (surname + "_" + ticket_prefix).str.replace("_$", "", regex=True)
        
        return df
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features."""
        df = df.copy()
        
        # Age groups
        age_bins = self.feature_config.get("age_bins", [0, 12, 18, 30, 50, 65, 100])
        age_labels = self.feature_config.get("age_labels", 
                                           ["Child", "Teen", "YoungAdult", "Adult", "Senior", "Elderly"])
        
        df["AgeBin"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, include_lowest=True)
        df["AgeBin"] = df["AgeBin"].astype(str)
        
        # Age group mapping
        def categorize_age(age: float) -> str:
            if pd.isna(age):
                return "Adult"  # Default for missing ages
            elif age < 12:
                return "Child"
            elif age < 18:
                return "Teen"
            elif age < 30:
                return "YoungAdult"
            elif age < 50:
                return "Adult"
            elif age < 65:
                return "Senior"
            else:
                return "Elderly"
        
        df["AgeGroup"] = df["Age"].apply(categorize_age)
        df["IsChild"] = ((df["Age"] < 18) | (df["Title"] == "Master")).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()
        
        # Convert Pclass to string for interactions
        df["PclassStr"] = df["Pclass"].astype(str)
        
        # Core interactions
        df["Sex_Pclass"] = df["Sex"] + "_" + df["PclassStr"]
        df["Title_Pclass"] = df["Title"] + "_" + df["PclassStr"]
        df["AgeGroup_Pclass"] = df["AgeGroup"] + "_" + df["PclassStr"]
        
        # Family size groups for interactions
        def family_size_group(size: int) -> str:
            if size == 1:
                return "Alone"
            elif size <= 3:
                return "Small"
            elif size <= 6:
                return "Medium"
            else:
                return "Large"
        
        df["FamilySizeGroup"] = df["FamilySize"].apply(family_size_group)
        df["FamilySize_Pclass"] = df["FamilySizeGroup"] + "_" + df["PclassStr"]
        df["Embarked_Pclass"] = df["Embarked"] + "_" + df["PclassStr"]
        
        return df
    
    def smart_imputation(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Intelligent imputation using group-based methods."""
        train_df = train_df.copy()
        if test_df is not None:
            test_df = test_df.copy()
            combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        else:
            combined_df = train_df
        
        # Age imputation by Title and Pclass groups
        age_groups = combined_df.groupby(["Title", "Pclass"])["Age"].median()
        title_age_groups = combined_df.groupby("Title")["Age"].median()
        global_age_median = combined_df["Age"].median()
        
        def impute_age(row):
            if pd.isna(row["Age"]):
                # Try Title + Pclass
                if (row["Title"], row["Pclass"]) in age_groups.index:
                    return age_groups.loc[(row["Title"], row["Pclass"])]
                # Fallback to Title only
                elif row["Title"] in title_age_groups.index:
                    return title_age_groups.loc[row["Title"]]
                # Final fallback
                else:
                    return global_age_median
            return row["Age"]
        
        # Apply imputation
        for df in [train_df] + ([test_df] if test_df is not None else []):
            df["Age"] = df.apply(impute_age, axis=1)
            
            # Fare imputation
            fare_groups = combined_df.groupby(["Pclass", "Embarked"])["Fare"].median()
            pclass_fare_groups = combined_df.groupby("Pclass")["Fare"].median()
            global_fare_median = combined_df["Fare"].median()
            
            def impute_fare(row):
                if pd.isna(row["Fare"]):
                    embarked = row["Embarked"] if pd.notna(row["Embarked"]) else "S"
                    if (row["Pclass"], embarked) in fare_groups.index:
                        return fare_groups.loc[(row["Pclass"], embarked)]
                    elif row["Pclass"] in pclass_fare_groups.index:
                        return pclass_fare_groups.loc[row["Pclass"]]
                    else:
                        return global_fare_median
                return row["Fare"]
            
            df["Fare"] = df.apply(impute_fare, axis=1)
            
            # Embarked imputation
            most_common_embarked = combined_df["Embarked"].mode()[0] if not combined_df["Embarked"].mode().empty else "S"
            df["Embarked"] = df["Embarked"].fillna(most_common_embarked)
        
        # Store imputation values for later use
        self.imputation_values = {
            "age_groups": age_groups.to_dict(),
            "title_age_groups": title_age_groups.to_dict(),
            "global_age_median": global_age_median,
            "fare_groups": fare_groups.to_dict(),
            "pclass_fare_groups": pclass_fare_groups.to_dict(),
            "global_fare_median": global_fare_median,
            "most_common_embarked": most_common_embarked,
        }
        
        return train_df, test_df
    
    def apply_validation_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply validation constraints to data."""
        df = df.copy()
        
        validation_config = self.config.get("validation", {})
        
        # Age constraints
        age_range = validation_config.get("age_range", [0, 100])
        df["Age"] = df["Age"].clip(lower=age_range[0], upper=age_range[1])
        
        # Fare constraints
        fare_min = validation_config.get("fare_min", 0)
        df["Fare"] = df["Fare"].clip(lower=fare_min)
        
        return df
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Main feature engineering pipeline."""
        # Copy dataframes
        train_processed = train_df.copy()
        test_processed = test_df.copy() if test_df is not None else None
        
        # Process both datasets together for global context
        datasets = [train_processed] + ([test_processed] if test_processed is not None else [])
        
        for df in datasets:
            # Basic feature extraction
            df["Title"] = df["Name"].apply(self.extract_title)
            df["CabinDeck"] = df["Cabin"].apply(self.extract_cabin_deck)
            df["HasCabin"] = (~df["Cabin"].isna()).astype(int)
            df["TicketPrefix"] = df["Ticket"].apply(self.extract_ticket_prefix)
            df["TicketHasLetters"] = df["Ticket"].str.contains(r"[A-Za-z]", na=False).astype(int)
        
        # Smart imputation
        train_processed, test_processed = self.smart_imputation(train_processed, test_processed)
        
        # Continue feature engineering
        for df in [train_processed] + ([test_processed] if test_processed is not None else []):
            # Family features
            df = self.create_family_features(df)
            
            # Age features
            df = self.create_age_features(df)
            
            # Interaction features
            df = self.create_interaction_features(df)
            
            # Apply validation constraints
            df = self.apply_validation_constraints(df)
        
        # Reorder columns for deterministic output
        column_order = self.feature_config.get("column_order", [])
        if column_order:
            # Only reorder columns that exist
            available_columns = train_processed.columns.tolist()
            ordered_columns = [col for col in column_order if col in available_columns]
            remaining_columns = [col for col in available_columns if col not in ordered_columns]
            final_order = ordered_columns + remaining_columns
            
            train_processed = train_processed[final_order]
            if test_processed is not None:
                test_processed = test_processed[final_order]
        
        return train_processed, test_processed