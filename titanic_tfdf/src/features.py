"""Advanced feature engineering for Titanic dataset."""

import re
import pandas as pd
import numpy as np
from typing import Optional, Set, Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder


def extract_title(name: str) -> str:
    """Extract and normalize title from passenger name with advanced grouping."""
    if pd.isna(name):
        return 'Unknown'
    
    title_search = re.search(r' ([A-Za-z]+)\.', str(name))
    if not title_search:
        return 'Unknown'
    
    title = title_search.group(1)
    
    # Advanced title grouping based on social status and age
    title_mapping = {
        # Standard titles
        'Mr': 'Mr',
        'Mrs': 'Mrs', 
        'Miss': 'Miss',
        'Master': 'Master',  # Young boys
        
        # Married women variations
        'Mme': 'Mrs',
        'Ms': 'Miss',
        'Mlle': 'Miss',
        
        # Nobility and high status
        'Lady': 'Nobility',
        'Sir': 'Nobility',
        'Countess': 'Nobility',
        'Capt': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        
        # Professional titles
        'Dr': 'Professional',
        'Rev': 'Professional',
        
        # Rare titles
        'Don': 'Rare',
        'Dona': 'Rare',
        'Jonkheer': 'Rare'
    }
    
    return title_mapping.get(title, 'Rare')


def extract_deck_from_cabin(cabin: str) -> str:
    """Extract deck letter from cabin with intelligent grouping."""
    if pd.isna(cabin) or cabin == '':
        return 'Unknown'
    
    # Extract first letter (deck)
    deck = str(cabin).strip()[0].upper()
    
    if not deck.isalpha():
        return 'Unknown'
    
    # Group decks by proximity and luxury level
    deck_mapping = {
        'A': 'Upper',    # Top deck, most luxurious
        'B': 'Upper',    # High deck
        'C': 'Upper',    # Upper middle
        'D': 'Middle',   # Middle deck
        'E': 'Middle',   # Middle deck
        'F': 'Lower',    # Lower deck
        'G': 'Lower',    # Lower deck
        'T': 'Other'     # Tank top
    }
    
    return deck_mapping.get(deck, 'Other')


def extract_ticket_info(ticket: str) -> Tuple[str, bool, int]:
    """Extract comprehensive ticket information."""
    if pd.isna(ticket):
        return 'Unknown', False, 0
    
    ticket_str = str(ticket).strip()
    
    # Check if ticket has letters (indicating class/type)
    has_letters = bool(re.search(r'[A-Za-z]', ticket_str))
    
    # Extract numeric part
    numbers = re.findall(r'\d+', ticket_str)
    ticket_number = int(numbers[-1]) if numbers else 0
    
    # Extract prefix
    prefix = re.sub(r'[\d\s/\.]', '', ticket_str).upper()
    
    # Group common prefixes
    if prefix in ['PC', 'PARIS']:
        prefix_group = 'First_Class'
    elif prefix in ['SC', 'SOTON', 'STON']:
        prefix_group = 'Southampton' 
    elif prefix in ['CA', 'CASOTON']:
        prefix_group = 'California'
    elif prefix == '':
        prefix_group = 'Numeric_Only'
    else:
        prefix_group = 'Other'
    
    return prefix_group, has_letters, ticket_number


def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sophisticated family-related features."""
    df = df.copy()
    
    # Basic family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Family size grouping based on survival patterns
    def categorize_family_size(size):
        if size == 1:
            return 'Alone'
        elif size <= 3:
            return 'Small'
        elif size <= 6:
            return 'Medium'
        else:
            return 'Large'
    
    df['FamilySizeGroup'] = df['FamilySize'].apply(categorize_family_size)
    
    # Family type analysis
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasSiblings'] = (df['SibSp'] > 0).astype(int)
    df['HasParents'] = (df['Parch'] > 0).astype(int)
    df['HasChildren'] = (df['Parch'] > 2).astype(int)  # More than 2 indicates likely children
    
    # Create family surname for potential family group analysis
    df['Surname'] = df['Name'].str.extract(r'^([^,]+),', expand=False).str.strip()
    
    return df


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age-related features with intelligent grouping."""
    df = df.copy()
    
    # Age grouping
    def categorize_age(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 12:
            return 'Child'
        elif age < 18:
            return 'Teenager'
        elif age < 30:
            return 'Young_Adult'
        elif age < 50:
            return 'Middle_Adult'
        elif age < 65:
            return 'Senior_Adult'
        else:
            return 'Elderly'
    
    df['AgeGroup'] = df['Age'].apply(categorize_age)
    
    # Create child indicator
    df['IsChild'] = ((df['Age'] < 18) | (df['Title'] == 'Master')).astype(int)
    
    # Create age bins for modeling
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 65, 100], 
                         labels=['Child', 'Teen', 'Young', 'Middle', 'Senior', 'Elderly'])
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between important variables."""
    df = df.copy()
    
    # Gender-class interaction (might already exist)
    if 'Sex_Pclass' not in df.columns:
        df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
    
    # Age-class interaction
    if 'AgeGroup' in df.columns:
        df['AgeGroup_Pclass'] = df['AgeGroup'] + '_' + df['Pclass'].astype(str)
    
    # Title-class interaction (social status indicator)
    df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
    
    # Family-class interaction
    df['FamilySize_Pclass'] = df['FamilySizeGroup'] + '_' + df['Pclass'].astype(str)
    
    # Embarked-class (port of departure by class) (might already exist)
    if 'Embarked_Pclass' not in df.columns:
        df['Embarked_Pclass'] = df['Embarked'].fillna('Unknown') + '_' + df['Pclass'].astype(str)
    
    # Special combinations
    df['Young_Female'] = ((df['Sex'] == 'female') & (df['Age'] < 30)).astype(int)
    df['Third_Class_Male'] = ((df['Sex'] == 'male') & (df['Pclass'] == 3)).astype(int)
    
    # Child with parents (check if IsChild exists)
    if 'IsChild' in df.columns:
        df['Child_With_Parents'] = (df['IsChild'] & (df['Parch'] > 0)).astype(int)
    
    return df


def smart_imputation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Intelligent missing value imputation using group-based methods."""
    
    # Combine for consistent imputation
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Age imputation by Title and Pclass groups
    age_groups = pd.concat([train_df, test_df]).groupby(['Title', 'Pclass'])['Age'].median()
    
    for df in [train_df, test_df]:
        for idx, row in df.iterrows():
            if pd.isna(row['Age']):
                try:
                    imputed_age = age_groups.loc[(row['Title'], row['Pclass'])]
                    if not pd.isna(imputed_age):
                        df.loc[idx, 'Age'] = imputed_age
                    else:
                        # Fallback to overall median by title
                        df.loc[idx, 'Age'] = pd.concat([train_df, test_df]).groupby('Title')['Age'].median().get(row['Title'], 29)
                except KeyError:
                    df.loc[idx, 'Age'] = 29  # Overall median fallback
    
    # Fare imputation by Pclass and Embarked
    fare_groups = pd.concat([train_df, test_df]).groupby(['Pclass', 'Embarked'])['Fare'].median()
    
    for df in [train_df, test_df]:
        for idx, row in df.iterrows():
            if pd.isna(row['Fare']):
                try:
                    embarked = row['Embarked'] if not pd.isna(row['Embarked']) else 'S'
                    imputed_fare = fare_groups.loc[(row['Pclass'], embarked)]
                    if not pd.isna(imputed_fare):
                        df.loc[idx, 'Fare'] = imputed_fare
                    else:
                        df.loc[idx, 'Fare'] = pd.concat([train_df, test_df]).groupby('Pclass')['Fare'].median().get(row['Pclass'], 15)
                except KeyError:
                    df.loc[idx, 'Fare'] = 15  # Fallback
    
    # Embarked imputation (mode)
    most_common_embarked = pd.concat([train_df, test_df])['Embarked'].mode()[0]
    for df in [train_df, test_df]:
        df['Embarked'] = df['Embarked'].fillna(most_common_embarked)
    
    return train_df, test_df


def engineer_advanced_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Master feature engineering function with global context."""
    
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 50)
    
    # Combine datasets for global feature engineering
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Mark datasets
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Combine
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # 1. Extract titles
    print("  Extracting social titles...")
    combined_df['Title'] = combined_df['Name'].apply(extract_title)
    
    # 2. Cabin features
    print("  Engineering cabin features...")
    combined_df['Deck'] = combined_df['Cabin'].apply(extract_deck_from_cabin)
    combined_df['HasCabin'] = (~combined_df['Cabin'].isna()).astype(int)
    combined_df['CabinCount'] = combined_df['Cabin'].fillna('').apply(lambda x: len(x.split()) if x else 0)
    
    # 3. Ticket features
    print("  Analyzing ticket information...")
    ticket_info = combined_df['Ticket'].apply(extract_ticket_info)
    combined_df['TicketPrefix'] = [info[0] for info in ticket_info]
    combined_df['TicketHasLetters'] = [info[1] for info in ticket_info]
    combined_df['TicketNumber'] = [info[2] for info in ticket_info]
    
    # 4. Family features
    print("  Creating family features...")
    combined_df = create_family_features(combined_df)
    
    # 5. Create basic interaction features before imputation
    print("  Building basic interaction features...")
    combined_df['Sex_Pclass'] = combined_df['Sex'] + '_' + combined_df['Pclass'].astype(str)
    combined_df['Embarked_Pclass'] = combined_df['Embarked'].fillna('Unknown') + '_' + combined_df['Pclass'].astype(str)
    
    # Split back for smart imputation
    train_part = combined_df[combined_df['is_train'] == 1].copy()
    test_part = combined_df[combined_df['is_train'] == 0].copy()
    
    # 6. Smart imputation
    print("  Performing intelligent imputation...")
    train_part, test_part = smart_imputation(train_part, test_part)
    
    # Recombine after imputation
    combined_df = pd.concat([train_part, test_part], ignore_index=True)
    
    # 7. Age features (after imputation)
    print("  Engineering age features...")
    combined_df = create_age_features(combined_df)
    
    # 8. Fare features
    print("  Creating fare features...")
    combined_df['FarePerPerson'] = combined_df['Fare'] / combined_df['FamilySize']
    combined_df['FareBin'] = pd.qcut(combined_df['Fare'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    # 9. Complete interaction features
    print("  Completing interaction features...")
    combined_df = create_interaction_features(combined_df)
    
    # 10. Advanced features
    print("  Creating advanced derived features...")
    
    # Name length as proxy for social status
    combined_df['NameLength'] = combined_df['Name'].str.len()
    
    # Survival likelihood by group (using training data only)
    if 'Survived' in train_part.columns:
        survival_rates = {}
        
        # Title survival rates
        title_survival = train_part.groupby('Title')['Survived'].mean()
        survival_rates['Title_SurvivalRate'] = combined_df['Title'].map(title_survival)
        
        # Deck survival rates  
        deck_survival = train_part.groupby('Deck')['Survived'].mean()
        survival_rates['Deck_SurvivalRate'] = combined_df['Deck'].map(deck_survival)
        
        # Sex-Class survival rates
        sexclass_survival = train_part.groupby('Sex_Pclass')['Survived'].mean()
        survival_rates['SexClass_SurvivalRate'] = combined_df['Sex_Pclass'].map(sexclass_survival)
        
        for feature, values in survival_rates.items():
            combined_df[feature] = values.fillna(values.mean())
    
    # Split back to train and test
    train_engineered = combined_df[combined_df['is_train'] == 1].copy()
    test_engineered = combined_df[combined_df['is_train'] == 0].copy()
    
    # Remove helper columns
    for df in [train_engineered, test_engineered]:
        if 'is_train' in df.columns:
            df.drop('is_train', axis=1, inplace=True)
        if 'Surname' in df.columns:
            df.drop('Surname', axis=1, inplace=True)
    
    print(f"Feature engineering complete!")
    print(f"  Training set: {train_engineered.shape}")
    print(f"  Test set: {test_engineered.shape}")
    print(f"  New features: {set(train_engineered.columns) - set(train_df.columns)}")
    
    return train_engineered, test_engineered


# Legacy compatibility functions
def add_derived_features(df: pd.DataFrame, vocab: Optional[Set[str]] = None) -> Tuple[pd.DataFrame, Set[str]]:
    """Legacy function for backward compatibility."""
    df = df.copy()
    
    # Basic feature engineering
    df['Title'] = df['Name'].apply(extract_title)
    df['Deck'] = df['Cabin'].apply(extract_deck_from_cabin)
    df = create_family_features(df)
    
    # Dummy vocabulary for compatibility
    if vocab is None:
        vocab = set(['basic', 'features'])
    
    return df, vocab