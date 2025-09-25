# src/data_processing/preprocessor.py: Data loading, feature engineering, encoding, and scaling for training the GBV vulnerability prediction model.
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from src.config import config
import warnings

# Ignore SettingWithCopyWarning, as we are intentionally modifying copies
warnings.filterwarnings('ignore', message='A value is trying to be set on a copy of a slice from a DataFrame')

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at: {file_path}")
        return None

def engineer_gbv_risk_features(df):
    """
    Engineer predictive features for GBV vulnerability assessment using
    vectorized operations.
    """
    df_features = df.copy()
    print("⚙️ Engineering features for training (vectorized)...")

    # --- Vectorized Feature Engineering ---
    
    # Economic Dependency Score
    employment_main_lower = df_features['employment_status_main'].str.strip().str.lower()
    employment_victim_lower = df_features['employment_status_victim_main'].str.strip().str.lower()
    
    df_features['economic_dependency_score'] = (
        (employment_main_lower == 'unemployed').astype(int) * 3 +
        (employment_main_lower == 'self employed').astype(int) * 1 +
        (employment_main_lower.isin(['unknown', 'not reported'])).astype(int) * 1 +
        (employment_victim_lower == 'unemployed').astype(int) * 2 +
        (employment_victim_lower == 'self employed').astype(int) * 1 +
        (employment_victim_lower.isin(['unknown', 'not reported'])).astype(int) * 1 +
        ((df_features['survivor_age'] < 18) | (df_features['survivor_age'] > 65)).astype(int) * 2 +
        df_features[['widow', 'out_of_school_child', 'minor', 'household_help', 'child_apprentice', 'orphans']].sum(axis=1)
    )
    df_features['economic_dependency_score'] = df_features['economic_dependency_score'].clip(upper=10)

    # Financial Access Proxy
    df_features['financial_access_proxy'] = 5
    df_features.loc[df_features['employment_status_main'].str.contains('formal', case=False, na=False), 'financial_access_proxy'] += 3
    df_features.loc[df_features['employment_status_main'].str.contains('self-employed', case=False, na=False), 'financial_access_proxy'] += 1
    
    education_map = {'tertiary': 3, 'university': 3, 'secondary': 2, 'primary': 1}
    df_features['financial_access_proxy'] += df_features['educational_status'].str.lower().map(education_map).fillna(0)
    
    df_features['financial_access_proxy'] -= df_features[['PLWD', 'PLHIV', 'female_sex_worker', 'IDP', 'drug_user', 'minor']].sum(axis=1) * 2
    df_features['financial_access_proxy'] = df_features['financial_access_proxy'].clip(lower=0)

    # Income Stability Score
    df_features['income_stability_score'] = 5
    marital_lower = df_features['marital_status'].str.lower().fillna('')
    df_features.loc[marital_lower.str.contains('married', na=False), 'income_stability_score'] += 2
    df_features.loc[marital_lower.str.contains('divorced|separated', regex=True, na=False), 'income_stability_score'] -= 2
    df_features.loc[marital_lower.str.contains('single', na=False), 'income_stability_score'] -= 1
    
    employment_lower = df_features['employment_status_main'].str.lower().fillna('')
    df_features.loc[employment_lower.str.contains('permanent|formal', regex=True, na=False), 'income_stability_score'] += 3
    df_features.loc[employment_lower.str.contains('casual|informal', regex=True, na=False), 'income_stability_score'] -= 1
    df_features.loc[employment_lower.str.contains('unemployed', na=False), 'income_stability_score'] -= 3
    
    df_features.loc[df_features['survivor_age'].between(25, 50), 'income_stability_score'] += 1
    df_features.loc[df_features['survivor_age'] < 18, 'income_stability_score'] -= 2
    df_features['income_stability_score'] = df_features['income_stability_score'].clip(lower=0)
    
# Housing Security Score
    df_features['housing_security_score'] = 5
    
    # Vectorized check for living arrangements
    living_lower = df_features['who_survivor/victim_stay_with'].str.lower().fillna('')
    df_features.loc[living_lower.str.contains('spouse|family', regex=True, na=False), 'housing_security_score'] += 2
    df_features.loc[living_lower.str.contains('alone', na=False), 'housing_security_score'] -= 1
    df_features.loc[living_lower.str.contains('friends', na=False), 'housing_security_score'] += 1

    # Vectorized subtraction for vulnerability factors
    housing_risk_factors = ['IDP', 'orphans', 'PLWD', 'PLHIV', 'female_sex_worker', 'drug_user', 'minor']
    df_features['housing_security_score'] -= df_features[housing_risk_factors].sum(axis=1) * 3

    # Vectorized check for employment status
    employment_lower = df_features['employment_status_victim_main'].str.lower().fillna('')
    df_features.loc[employment_lower.str.contains('unemployed', na=False), 'housing_security_score'] -= 2

    # Clip score to be non-negative
    df_features['housing_security_score'] = df_features['housing_security_score'].clip(lower=0)

    # Social Isolation Score
    df_features['social_isolation_score'] = (
        (df_features['who_survivor/victim_stay_with'].str.lower().str.contains('alone', na=False)).astype(int) * 3 +
        ((df_features['survivor_age'] > 65) | (df_features['survivor_age'] < 18)).astype(int) * 1 +
        df_features[['widow', 'PLHIV', 'PLWD', 'drug_user', 'female_sex_worker', 'orphans', 'IDP']].sum(axis=1) * 2
    )
    df_features.loc[marital_lower.str.contains('divorced|separated', regex=True, na=False), 'social_isolation_score'] += 2
    df_features.loc[marital_lower.str.contains('widowed', na=False), 'social_isolation_score'] += 3

    # Community Connection Score
    df_features['community_connection_score'] = 5
    df_features.loc[marital_lower.str.contains('married', na=False), 'community_connection_score'] += 2
    df_features.loc[~employment_lower.str.contains('unemployed', na=False), 'community_connection_score'] += 2
    
    df_features.loc[df_features['educational_status'].str.lower().str.contains('tertiary', na=False), 'community_connection_score'] += 3
    df_features.loc[df_features['educational_status'].str.lower().str.contains('secondary', na=False), 'community_connection_score'] += 2
    
    df_features['community_connection_score'] -= df_features[['IDP', 'drug_user', 'out_of_school_child']].sum(axis=1) * 2
    df_features['community_connection_score'] = df_features['community_connection_score'].clip(lower=0)
    
    # Map 'who_survivor/victim_stay_with' to the expected column name
    if 'victim_lives_with' in df_features.columns:
        df_features['who_survivor/victim_stay_with'] = df_features['victim_lives_with']

    print("✅ Feature engineering completed (vectorized)!")
    return df_features

def encode_categorical_features(df_train, df_test):
    """Encodes categorical features using LabelEncoder."""
    encoders = {}
    categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
    
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    
    for feature in categorical_features:
        encoder = LabelEncoder()
        df_train_encoded[feature] = encoder.fit_transform(df_train[feature].astype(str))
        
        test_categories = df_test[feature].astype(str)
        most_common = df_train[feature].mode()[0] if not df_train[feature].mode().empty else 'Unknown'
        test_categories = test_categories.apply(
            lambda x: x if x in encoder.classes_ else most_common
        )
        df_test_encoded[feature] = encoder.transform(test_categories)
        
        encoders[feature] = encoder
    
    return df_train_encoded, df_test_encoded, encoders

def scale_numerical_features(df_train, df_test, feature_list):
    """Applies StandardScaler and RobustScaler to numerical features."""
    scalers = {}
    numerical_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    for feature in numerical_features:
        if feature in config.ROBUST_SCALER_COLS:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        df_train_scaled[feature] = scaler.fit_transform(df_train[[feature]])
        df_test_scaled[feature] = scaler.transform(df_test[[feature]])
        
        scalers[feature] = scaler
    
    return df_train_scaled, df_test_scaled, scalers

def preprocess_for_training():
    """Main preprocessing pipeline for training the model."""
    df = load_data(config.DATA_FILE)
    if df is None:
        raise FileNotFoundError("Training data not found.")

    df_engineered = engineer_gbv_risk_features(df)

    model_features = config.TOP_FEATURES + [config.TARGET_VARIABLE]
    
    for col in model_features:
        if col not in df_engineered.columns:
            df_engineered[col] = 0
            print(f"Warning: Column '{col}' not found in engineered data, filled with 0.")

    df_final = df_engineered[model_features]

    X = df_final.drop(config.TARGET_VARIABLE, axis=1)
    y = df_final[config.TARGET_VARIABLE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Impute missing values
    # For numerical features, fill with the median
    for feature in numerical_features:
        median_val = X_train[feature].median()
        X_train[feature].fillna(median_val, inplace=True)
        X_test[feature].fillna(median_val, inplace=True)

    # For categorical features, fill with the most frequent value (mode)
    for feature in categorical_features:
        mode_val = X_train[feature].mode()[0]
        X_train[feature].fillna(mode_val, inplace=True)
        X_test[feature].fillna(mode_val, inplace=True)

    X_train, X_test, encoders = encode_categorical_features(X_train.copy(), X_test.copy())
    X_train, X_test, scalers = scale_numerical_features(X_train.copy(), X_test.copy(), config.TOP_FEATURES)

    return X_train, X_test, y_train, y_test, encoders, scalers