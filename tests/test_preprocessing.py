import pandas as pd
import pytest
from src.data_processing import preprocessor
from src.config import config

@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    data = {
        'survivor_age': [25, 40],
        'survivor_sex': ['Female', 'Male'],
        'employment_status_main': ['Unemployed', 'Self employed'],
        'employment_status_victim_main': ['Unemployed', 'Self employed'],
        'widow': [1, 0],
        'out_of_school_child': [0, 0],
        'minor': [0, 0],
        'household_help': [0, 1],
        'child_apprentice': [0, 0],
        'orphans': [0, 0],
        'educational_status': ['Primary', 'Secondary'],
        'IDP': [1, 0],
        'marital_status': ['Widowed', 'Married'],
        'victim_lives_with': ['Alone', 'Spouse/Partner'],
        'female_sex_worker': [0, 0], 
        'PLHIV': [0, 1], 'PLWD': [1, 0], 
        'drug_user': [0, 0],
        'vulnerability_target': [1, 0]
    }
    # Add columns for other engineered features to avoid KeyErrors
    data['financial_decision_autonomy'] = [0, 1]
    data['has_savings'] = [0, 1]
    data['recent_job_loss'] = [1, 0]
    data['is_income_regular'] = [0, 1]
    data['risk_of_eviction'] = [1, 0]
    data['safe_shelter_access'] = [0, 1]
    data['social_network_size'] = [2, 10]
    data['community_participation'] = [0, 1]
    data['perception_of_community_support'] = [0, 1]
    data['access_to_credit'] = [0, 1]
    data['owns_property_assets'] = [0, 1]
    return pd.DataFrame(data)

def test_engineer_gbv_risk_features(sample_dataframe):
    """Tests if feature engineering function creates the expected columns."""
    engineered_df = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    
    # Check if a few key engineered features are created
    assert 'economic_dependency_score' in engineered_df.columns
    assert 'social_isolation_score' in engineered_df.columns
    assert 'community_connection_score' in engineered_df.columns
    assert 'who_survivor/victim_stay_with' in engineered_df.columns # Check mapping
    
    # Check if scores are numeric
    assert pd.api.types.is_numeric_dtype(engineered_df['economic_dependency_score'])
    assert engineered_df.shape[0] == 2 # Ensure no rows were dropped

def test_encode_categorical_features(sample_dataframe):
    """Tests the categorical encoding function."""
    df_train = sample_dataframe.copy()
    df_test = sample_dataframe.copy()

    # Select only the features that will be used for modeling
    df_train_X = df_train[config.TOP_FEATURES]
    df_test_X = df_test[config.TOP_FEATURES]

    # Map the living situation column to match TOP_FEATURES
    df_train_X = df_train_X.rename(columns={'victim_lives_with': 'who_survivor/victim_stay_with'})
    df_test_X = df_test_X.rename(columns={'victim_lives_with': 'who_survivor/victim_stay_with'})
    
    _, _, encoders = preprocessor.encode_categorical_features(df_train_X, df_test_X)
    
    assert 'survivor_sex' in encoders
    assert 'marital_status' in encoders
    assert len(encoders['survivor_sex'].classes_) >= 2