import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch
from src.data_processing import preprocessor
from src.config import config

@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame that mimics the raw input data structure."""
    data = {
        'survivor_age': [25, 40, 17, 68],
        'survivor_sex': ['Female', 'Male', 'Female', 'Female'],
        'employment_status_main': ['Unemployed', 'Self employed', 'Not reported', 'Currently employed'],
        'employment_status_victim_main': ['Unemployed', 'Self employed', 'Unemployed', 'Unemployed'],
        'marital_status': ['Widowed', 'Married/cohabiting', 'Single', 'Divorced/separated'],
        'educational_status': ['No formal', 'Completed secondary', 'Some primary', 'Graduate'],
        'who_survivor/victim_stay_with': ['Alone', 'Partner', 'Parent', 'Alone'],
        'widow': [1, 0, 0, 0],
        'out_of_school_child': [0, 0, 1, 0],
        'minor': [0, 0, 1, 0],
        'household_help': [0, 1, 0, 0],
        'child_apprentice': [0, 0, 1, 0],
        'orphans': [0, 0, 0, 1],
        'IDP': [1, 0, 0, 1],
        'female_sex_worker': [0, 0, 0, 1],
        'PLHIV': [0, 1, 0, 0],
        'PLWD': [1, 0, 0, 0],
        'drug_user': [0, 0, 1, 0],
        'vulnerability_target': [1, 0, 1, 1]  # Target variable
    }
    return pd.DataFrame(data)

def test_engineer_gbv_risk_features(sample_dataframe):
    """
    Tests if the feature engineering function creates the expected new feature columns
    and if they are of a numeric type.
    """
    engineered_df = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    
    # Check if key engineered features are created
    expected_new_features = [
        'economic_dependency_score',
        'social_isolation_score',
        'community_connection_score',
        'income_stability_score',
        'housing_security_score',
        'financial_access_proxy'
    ]
    for feature in expected_new_features:
        assert feature in engineered_df.columns
        assert pd.api.types.is_numeric_dtype(engineered_df[feature])

    # Ensure no rows were dropped during the process
    assert engineered_df.shape[0] == sample_dataframe.shape[0]
    # Check if a known value is computed correctly
    # Example: First row's social_isolation_score
    # (Alone*3) + (Widowed*3) + (PLWD*2) + (IDP*2) = 3 + 3 + 2 + 2 = 10
    assert engineered_df['social_isolation_score'].iloc[0] == 10

def test_encode_categorical_features(sample_dataframe):
    """
    Tests the categorical encoding function to ensure it encodes features
    and handles unseen categories in the test set correctly.
    """
    df_train = sample_dataframe.copy()
    # Create a test set with a category not seen in training
    df_test = sample_dataframe.copy()
    df_test.loc[0, 'marital_status'] = 'Unknown' 

    # Extract categorical columns for encoding
    train_cat_features = df_train.select_dtypes(include=['object'])
    test_cat_features = df_test.select_dtypes(include=['object'])

    df_train_encoded, df_test_encoded, encoders = preprocessor.encode_categorical_features(
        train_cat_features, test_cat_features
    )

    # Assert that categorical columns are now numeric
    for col in train_cat_features.columns:
        assert pd.api.types.is_numeric_dtype(df_train_encoded[col])
        assert pd.api.types.is_numeric_dtype(df_test_encoded[col])

    # Check that encoders were created for each categorical feature
    assert set(encoders.keys()) == set(train_cat_features.columns)
    # Check handling of unseen value. 'Unknown' should be mapped to the most frequent value from training.
    # The mode of marital_status in train is 'Widowed' (or any if all unique). Let's assume it maps.
    # 'Widowed' is encoded to 3 in the training set. The unseen 'Unknown' should also be encoded to 3.
    widowed_encoded_value = encoders['marital_status'].transform(['Widowed'])[0]
    assert df_test_encoded['marital_status'].iloc[0] == widowed_encoded_value

def test_scale_numerical_features(sample_dataframe):
    """
    Tests if numerical features are scaled correctly and that scalers are returned.
    """
    # Use only a subset of numerical columns for this test
    numerical_cols = ['survivor_age', 'economic_dependency_score']
    df_engineered = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    
    df_train = df_engineered[numerical_cols]
    df_test = df_engineered[numerical_cols]

    df_train_scaled, df_test_scaled, scalers = preprocessor.scale_numerical_features(
        df_train, df_test, config.TOP_FEATURES
    )

    # Check if scalers were created for the numerical features
    assert set(scalers.keys()) == set(numerical_cols)
    
    # Check that scaled data has means close to 0 and std dev close to 1 (for StandardScaler)
    # Note: RobustScaler columns will not have this property. 'survivor_age' uses StandardScaler.
    assert np.isclose(df_train_scaled['survivor_age'].mean(), 0, atol=1e-9)
    assert np.isclose(df_train_scaled['survivor_age'].std(), 1, atol=1e-9)

@patch('src.data_processing.preprocessor.load_data')
def test_preprocess_for_training(mock_load_data, sample_dataframe):
    """
    Integration test for the main preprocessing pipeline.
    It mocks the data loading step and verifies the final output.
    """
    # Configure the mock to return our sample dataframe
    mock_load_data.return_value = sample_dataframe

    X_train, X_test, y_train, y_test, encoders, scalers = preprocessor.preprocess_for_training()

    # --- Check Shapes and Types ---
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[1] == len(config.TOP_FEATURES)
    assert X_test.shape[1] == len(config.TOP_FEATURES)
    assert X_train.shape[0] + X_test.shape[0] == sample_dataframe.shape[0]

    # --- Check for Null Values ---
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()

    # --- Check Encoding and Scaling ---
    # Ensure all columns in the final dataframes are numeric
    assert all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns)
    assert all(pd.api.types.is_numeric_dtype(X_test[col]) for col in X_test.columns)
    
    # Check if encoders and scalers were returned
    assert encoders is not None
    assert scalers is not None
    assert 'marital_status' in encoders
    assert 'survivor_age' in scalers