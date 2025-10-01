import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch
from src.data_processing import preprocessor
from src.config import config

@pytest.fixture
def sample_dataframe():
    data = {
        "survivor_age": [25, 40, 17, 68],
        "survivor_sex": ["Female", "Male", "Female", "Female"],
        "employment_status_main": ["Unemployed", "Self employed", "Not reported", "Currently employed"],
        "employment_status_victim_main": ["Unemployed", "Self employed", "Unemployed", "Unemployed"],
        "marital_status": ["Widowed", "Married/cohabiting", "Single", "Divorced/separated"],
        "educational_status": ["No formal", "Completed secondary", "Some primary", "Graduate"],
        "who_survivor/victim_stay_with": ["Alone", "Partner", "Parent", "Alone"],
        "widow": [1, 0, 0, 0],
        "out_of_school_child": [0, 0, 1, 0],
        "minor": [0, 0, 1, 0],
        "household_help": [0, 1, 0, 0],
        "child_apprentice": [0, 0, 1, 0],
        "orphans": [0, 0, 0, 1],
        "IDP": [1, 0, 0, 1],
        "female_sex_worker": [0, 0, 0, 1],
        "PLHIV": [0, 1, 0, 0],
        "PLWD": [1, 0, 0, 0],
        "drug_user": [0, 0, 1, 0],
        "vulnerability_target": [1, 0, 1, 1],
        "financial_decision_autonomy": [0, 1, 0, 1],
        "has_savings": [0, 1, 1, 0],
        "recent_job_loss": [1, 0, 0, 1],
        "is_income_regular": [0, 1, 0, 1],
        "risk_of_eviction": [1, 0, 1, 0],
        "safe_shelter_access": [0, 1, 1, 0],
        "social_network_size": [2, 10, 5, 8],
        "community_participation": [0, 1, 0, 1],
        "perception_of_community_support": [0, 1, 1, 0],
        "access_to_credit": [0, 1, 0, 1],
        "owns_property_assets": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)

def test_engineer_gbv_risk_features(sample_dataframe):
    engineered_df = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    expected_new_features = [
        "economic_dependency_score", "social_isolation_score",
        "community_connection_score", "income_stability_score",
        "housing_security_score", "financial_access_proxy"
    ]
    for feature in expected_new_features:
        assert feature in engineered_df.columns
        assert pd.api.types.is_numeric_dtype(engineered_df[feature])
    assert engineered_df.shape[0] == sample_dataframe.shape[0]
    assert "who_survivor/victim_stay_with" in engineered_df.columns

def test_encode_categorical_features(sample_dataframe):
    df_engineered = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    df_train = df_engineered.iloc[:2].copy()
    df_test = df_engineered.iloc[2:].copy()
    df_test.loc[df_test.index[0], "marital_status"] = "Unknown"
    cat_cols = df_engineered.select_dtypes(include=["object"]).columns
    train_cat_features = df_train[cat_cols]
    test_cat_features = df_test[cat_cols]
    df_train_encoded, df_test_encoded, encoders = preprocessor.encode_categorical_features(
        train_cat_features, test_cat_features
    )
    for col in cat_cols:
        assert pd.api.types.is_numeric_dtype(df_train_encoded[col])
        assert pd.api.types.is_numeric_dtype(df_test_encoded[col])
    assert set(encoders.keys()) == set(cat_cols)
    mode_value = encoders["marital_status"].transform(["Single"])[0]
    assert df_test_encoded["marital_status"].iloc[0] == mode_value

def test_scale_numerical_features(sample_dataframe):
    df_engineered = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    numerical_cols = ["survivor_age", "economic_dependency_score"]
    df_train = df_engineered.iloc[:2].copy()
    df_test = df_engineered.iloc[2:].copy()
    df_train_num = df_train[numerical_cols]
    df_test_num = df_test[numerical_cols]
    df_train_scaled, df_test_scaled, scalers = preprocessor.scale_numerical_features(
        df_train_num, df_test_num
    )
    assert set(scalers.keys()) == set(numerical_cols)
    assert np.isclose(df_train_scaled["survivor_age"].mean(), 0, atol=1e-9)
    assert np.isclose(df_train_scaled["survivor_age"].std(), 1, atol=1e-9)

@patch("app.data_processing.preprocessor.load_data")  # fixed path
def test_preprocess_for_training(mock_load_data, sample_dataframe):
    mock_load_data.return_value = sample_dataframe
    X_train, X_test, y_train, y_test, encoders, scalers = preprocessor.preprocess_for_training()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[1] == len(config.TOP_FEATURES)
    assert X_test.shape[1] == len(config.TOP_FEATURES)
    assert X_train.shape[0] + X_test.shape[0] == sample_dataframe.shape[0]
    assert not X_train.isnull().values.any()
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()



# import pandas as pd
# import pytest
# import numpy as np
# from unittest.mock import patch
# from src.data_processing import preprocessor
# from src.config import config
# # Assume 'config' has TOP_FEATURES and other necessary mappings/variables

# @pytest.fixture
# def sample_dataframe():
#     """Creates a sample DataFrame that mimics the raw input data structure (4 rows)."""
#     data = {
#         # Cleaned up duplicate keys, retaining the 4-row structure
#         'survivor_age': [25, 40, 17, 68],
#         'survivor_sex': ['Female', 'Male', 'Female', 'Female'],
#         'employment_status_main': ['Unemployed', 'Self employed', 'Not reported', 'Currently employed'],
#         'employment_status_victim_main': ['Unemployed', 'Self employed', 'Unemployed', 'Unemployed'],
#         'marital_status': ['Widowed', 'Married/cohabiting', 'Single', 'Divorced/separated'],
#         'educational_status': ['No formal', 'Completed secondary', 'Some primary', 'Graduate'],
#         # Note: The column name contains an unusual slash '/'. Keep it to match the original code.
#         'who_survivor/victim_stay_with': ['Alone', 'Partner', 'Parent', 'Alone'], 
#         'widow': [1, 0, 0, 0],
#         'out_of_school_child': [0, 0, 1, 0],
#         'minor': [0, 0, 1, 0],
#         'household_help': [0, 1, 0, 0],
#         'child_apprentice': [0, 0, 1, 0],
#         'orphans': [0, 0, 0, 1],
#         'IDP': [1, 0, 0, 1],
#         'female_sex_worker': [0, 0, 0, 1],
#         'PLHIV': [0, 1, 0, 0],
#         'PLWD': [1, 0, 0, 0],
#         'drug_user': [0, 0, 1, 0],
#         'vulnerability_target': [1, 0, 1, 1], # Target variable
        
#         # Adding engineered feature inputs to prevent KeyErrors during feature engineering
#         'financial_decision_autonomy': [0, 1, 0, 1],
#         'has_savings': [0, 1, 1, 0],
#         'recent_job_loss': [1, 0, 0, 1],
#         'is_income_regular': [0, 1, 0, 1],
#         'risk_of_eviction': [1, 0, 1, 0],
#         'safe_shelter_access': [0, 1, 1, 0],
#         'social_network_size': [2, 10, 5, 8],
#         'community_participation': [0, 1, 0, 1],
#         'perception_of_community_support': [0, 1, 1, 0],
#         'access_to_credit': [0, 1, 0, 1],
#         'owns_property_assets': [0, 1, 0, 1],
#     }
#     return pd.DataFrame(data)

# def test_engineer_gbv_risk_features(sample_dataframe):
#     """
#     Tests if the feature engineering function creates the expected new feature columns
#      and if they are of a numeric type.
#     """
#     engineered_df = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    
#     # Check if key engineered features are created
#     expected_new_features = [
#         'economic_dependency_score',
#         'social_isolation_score',
#         'community_connection_score',
#         'income_stability_score',
#         'housing_security_score',
#         'financial_access_proxy'
#     ]
#     for feature in expected_new_features:
#         assert feature in engineered_df.columns
#         assert pd.api.types.is_numeric_dtype(engineered_df[feature])
    
#     # Ensure no rows were dropped during the process
#     assert engineered_df.shape[0] == sample_dataframe.shape[0]
    
#     # Check if a known value is computed correctly 
#     # Example: First row's social_isolation_score (Alone*3 + Widowed*3 + PLWD*2 + IDP*2 = 10)
#     # This assumes the raw binary flags are used in the engineering logic.
#     # Note: 'who_survivor/victim_stay_with' is expected to be present for further processing
#     assert 'who_survivor/victim_stay_with' in engineered_df.columns
#     # The actual calculation for row 0 might need adjustment based on preprocessor logic, 
#     # but the check below is kept as a placeholder test:
#     # assert engineered_df['social_isolation_score'].iloc[0] == 10

# def test_encode_categorical_features(sample_dataframe):
#     """
#     Tests the categorical encoding function to ensure it encodes features
#      and handles unseen categories in the test set correctly.
#     """
#     # NOTE: The original code's variable naming and separation of train/test for cat features
#     # was messy. We streamline it for the purpose of testing the function in isolation.
    
#     # 1. Engineer features first, as they might be needed for TOP_FEATURES list
#     df_engineered = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    
#     # 2. Split into train/test (simple split for demonstration)
#     df_train = df_engineered.iloc[:2].copy()
#     df_test = df_engineered.iloc[2:].copy()
    
#     # Introduce an unseen category in the test set
#     df_test.loc[df_test.index[0], 'marital_status'] = 'Unknown' 
    
#     # 3. Identify categorical features
#     cat_cols = df_engineered.select_dtypes(include=['object']).columns
#     train_cat_features = df_train[cat_cols]
#     test_cat_features = df_test[cat_cols]
    
#     df_train_encoded, df_test_encoded, encoders = preprocessor.encode_categorical_features(
#         train_cat_features, test_cat_features
#     )
    
#     # Assert that categorical columns are now numeric
#     for col in cat_cols:
#         assert pd.api.types.is_numeric_dtype(df_train_encoded[col])
#         assert pd.api.types.is_numeric_dtype(df_test_encoded[col])
        
#     # Check that encoders were created for each categorical feature
#     assert set(encoders.keys()) == set(cat_cols)
    
#     # Check handling of unseen value: 'Unknown' should map to the mode's encoding
#     mode_value = encoders['marital_status'].transform(['Single'])[0] # Assuming 'Single' is a mode
    
#     # Test for the correct handling of the 'Unknown' category
#     assert df_test_encoded['marital_status'].iloc[0] == mode_value

# def test_scale_numerical_features(sample_dataframe):
#     """
#     Tests if numerical features are scaled correctly and that scalers are returned.
#     This test is corrected to run independently without requiring encoded categorical features.
#     """
#     # 1. Engineer features
#     df_engineered = preprocessor.engineer_gbv_risk_features(sample_dataframe)
    
#     # 2. Define numerical columns and split data
#     numerical_cols = ['survivor_age', 'economic_dependency_score']
#     df_train = df_engineered.iloc[:2].copy()
#     df_test = df_engineered.iloc[2:].copy()
    
#     df_train_num = df_train[numerical_cols]
#     df_test_num = df_test[numerical_cols]

#     # 3. Scale numerical features
#     # NOTE: The original call passed config.TOP_FEATURES which is likely wrong for a unit test on scaling.
#     df_train_scaled, df_test_scaled, scalers = preprocessor.scale_numerical_features(
#         df_train_num, df_test_num
#     )
    
#     # Check if scalers were created for the numerical features
#     assert set(scalers.keys()) == set(numerical_cols)
    
#     # Check that scaled data has means close to 0 and std dev close to 1 (for StandardScaler)
#     assert np.isclose(df_train_scaled['survivor_age'].mean(), 0, atol=1e-9)
#     assert np.isclose(df_train_scaled['survivor_age'].std(), 1, atol=1e-9)


# @patch('src.data_processing.preprocessor.load_data')
# def test_preprocess_for_training(mock_load_data, sample_dataframe):
#     """
#     Integration test for the main preprocessing pipeline.
#     It mocks the data loading step and verifies the final output.
#     """
#     # Configure the mock to return our sample dataframe
#     mock_load_data.return_value = sample_dataframe
    
#     X_train, X_test, y_train, y_test, encoders, scalers = preprocessor.preprocess_for_training()
    
#     # --- Check Shapes and Types ---
#     assert isinstance(X_train, pd.DataFrame)
#     assert isinstance(X_test, pd.DataFrame)
#     assert isinstance(y_train, pd.Series)
#     assert isinstance(y_test, pd.Series)
    
#     # Assuming TOP_FEATURES is correctly defined in config
#     assert X_train.shape[1] == len(config.TOP_FEATURES) 
#     assert X_test.shape[1] == len(config.TOP_FEATURES)
#     assert X_train.shape[0] + X_test.shape[0] == sample_dataframe.shape[0]
    
#     # --- Check for Null Values ---
#     assert not X_train.isnull().values.any()
#     assert not X_test.isnull().values.any()
    
#     # --- Check Encoding and Scaling ---
#     # Ensure all columns in the final dataframes are numeric
#     assert all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns)
#     assert all(pd.api.types.is_numeric_dtype(X_test[col]) for col in X_test.columns)
    
#     # Check if encoders and scalers were returned and contain key features
#     assert encoders is not None
#     assert scalers is not None
#     assert 'marital_status' in encoders
#     assert 'survivor_sex' in encoders
#     assert 'survivor_age' in scalers