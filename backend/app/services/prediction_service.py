# backend/app/services/prediction_service.py: Feature engineering, preprocessing, prediction, and explanation services.  

import pandas as pd
import numpy as np
import json
from . import explanation_service
from ..config import config
from ..utils import utils

# --- Safe Artifact Loading with Fallback ---
class DummyModel:
    """Fallback model when real model artifacts are missing."""
    def predict_proba(self, X):
        # Always return 0.5 probability for both classes
        return np.array([[0.5, 0.5] for _ in range(len(X))])

try:
    model = utils.load_artifact(config.MODEL_PATH)
    if model is None:
        print("⚠️ Model not found, using DummyModel.")
        model = DummyModel()
except Exception as e:
    print(f"⚠️ Failed to load model: {e}, using DummyModel.")
    model = DummyModel()

try:
    encoders = utils.load_artifact(config.ENCODERS_PATH) or {}
except Exception as e:
    print(f"⚠️ Failed to load encoders: {e}, using empty dict.")
    encoders = {}

try:
    scalers = utils.load_artifact(config.SCALERS_PATH) or {}
except Exception as e:
    print(f"⚠️ Failed to load scalers: {e}, using empty dict.")
    scalers = {}

# ---------------------------------------------------------------------
def engineer_features_for_prediction(data: dict) -> pd.DataFrame:
    """Computes engineered features from a dictionary of input data for a single prediction."""
    df_features = pd.DataFrame([data])
    
    # Economic Dependency Score
    def calculate_economic_dependency(row):
        score = 0
        employment_main = str(row.get('employment_status_main', '')).strip()
        if employment_main == 'Unemployed':
            score += 3
        elif employment_main == 'Self employed':
            score += 1
        elif employment_main in ['Unknown', 'Not reported']:
            score += 1
        employment_victim = str(row.get('employment_status_victim_main', '')).strip()
        if employment_victim == 'Unemployed':
            score += 2
        elif employment_victim == 'Self employed':
            score += 1
        elif employment_victim in ['Unknown', 'Not reported']:
            score += 1
        age = row.get('survivor_age', 0)
        if age < 18 or age > 65:
            score += 2
        for factor in ['widow', 'out_of_school_child', 'minor', 'household_help', 'child_apprentice', 'orphans']:
            if row.get(factor, 0) == 1:
                score += 1
        return min(score, 10)
    
    df_features['economic_dependency_score'] = df_features.apply(calculate_economic_dependency, axis=1)

    # Financial Access Proxy
    def financial_access_proxy(row):
        score = 5
        employment_main = row.get('employment_status_main', '')
        if pd.notna(employment_main) and employment_main:
            emp_str = str(employment_main).lower()
            if 'formal' in emp_str:
                score += 3
            elif 'self-employed' in emp_str or 'self employed' in emp_str:
                score += 1
        educational_status = row.get('educational_status', '')
        if pd.notna(educational_status) and educational_status:
            edu = str(educational_status).lower()
            if any(k in edu for k in ['tertiary', 'university', 'undergraduate', 'graduate']):
                score += 3
            elif any(k in edu for k in ['secondary', 'diploma']):
                score += 2
            elif 'primary' in edu:
                score += 1
        for factor in ['PLWD', 'PLHIV', 'female_sex_worker', 'IDP', 'drug_user', 'minor']:
            if row.get(factor, 0) == 1:
                score -= 2
        return max(score, 0)
    
    df_features['financial_access_proxy'] = df_features.apply(financial_access_proxy, axis=1)

    # Income Stability Score
    def income_stability_score(row):
        score = 5
        marital_status = row.get('marital_status', '')
        if pd.notna(marital_status) and marital_status:
            ms = str(marital_status).lower()
            if 'married' in ms:
                score += 2
            elif 'divorced' in ms or 'separated' in ms:
                score -= 2
            elif 'single' in ms:
                score -= 1
        employment_main = row.get('employment_status_main', '')
        if pd.notna(employment_main) and employment_main:
            emp = str(employment_main).lower()
            if 'permanent' in emp or 'formal' in emp:
                score += 3
            elif 'casual' in emp or 'informal' in emp:
                score -= 1
            elif 'unemployed' in emp:
                score -= 3
        age = row.get('survivor_age', 0)
        if 25 <= age <= 50:
            score += 1
        elif age < 18:
            score -= 2
        return max(score, 0)
    
    df_features['income_stability_score'] = df_features.apply(income_stability_score, axis=1)

    # Housing Security Score
    def housing_security_score(row):
        score = 5
        living_arrangement = row.get('who_survivor_victim_stay_with', '')
        if pd.notna(living_arrangement) and living_arrangement:
            la = str(living_arrangement).lower()
            if any(k in la for k in ['spouse', 'family', 'partner']):
                score += 2
            elif 'alone' in la:
                score -= 1
            elif 'friends' in la:
                score += 1
        for factor in ['IDP', 'orphans', 'PLWD', 'PLHIV', 'female_sex_worker', 'drug_user', 'minor']:
            if row.get(factor, 0) == 1:
                score -= 3
        emp_main = row.get('employment_status_main', '')
        if pd.notna(emp_main) and 'unemployed' in str(emp_main).lower():
            score -= 2
        return max(score, 0)
    
    df_features['housing_security_score'] = df_features.apply(housing_security_score, axis=1)

    # Social Isolation Score
    def social_isolation_score(row):
        score = 0
        living_arrangement = row.get('who_survivor_victim_stay_with', '')
        if pd.notna(living_arrangement) and 'alone' in str(living_arrangement).lower():
            score += 3
        age = row.get('survivor_age', 0)
        if age > 65 or age < 18:
            score += 1
        for factor in ['widow', 'PLHIV', 'PLWD', 'drug_user', 'female_sex_worker', 'orphans', 'IDP']:
            if row.get(factor, 0) == 1:
                score += 2
        ms = row.get('marital_status', '')
        if pd.notna(ms):
            ms_l = str(ms).lower()
            if 'divorced' in ms_l or 'separated' in ms_l:
                score += 2
            elif 'widowed' in ms_l:
                score += 3
        return score
    
    df_features['social_isolation_score'] = df_features.apply(social_isolation_score, axis=1)

    # Community Connection Score
    def community_connection_score(row):
        score = 5
        ms = row.get('marital_status', '')
        if pd.notna(ms) and 'married' in str(ms).lower():
            score += 2
        emp = row.get('employment_status_main', '')
        if pd.notna(emp) and 'unemployed' not in str(emp).lower():
            score += 2
        edu = row.get('educational_status', '')
        if pd.notna(edu):
            edu_l = str(edu).lower()
            if any(k in edu_l for k in ['tertiary', 'university', 'undergraduate', 'graduate']):
                score += 3
            elif any(k in edu_l for k in ['secondary', 'diploma']):
                score += 2
        for factor in ['IDP', 'drug_user', 'out_of_school_child']:
            if row.get(factor, 0) == 1:
                score -= 2
        return max(score, 0)
    
    df_features['community_connection_score'] = df_features.apply(community_connection_score, axis=1)

    if 'who_survivor_victim_stay_with' in df_features.columns:
        df_features['who_survivor/victim_stay_with'] = df_features['who_survivor_victim_stay_with']
        
    return df_features

# ---------------------------------------------------------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the input DataFrame for prediction."""
    df_processed = df.copy()
    
    for col in config.TOP_FEATURES:
        if col not in df_processed.columns:
            if col == 'who_survivor/victim_stay_with' and 'who_survivor_victim_stay_with' in df_processed.columns:
                df_processed[col] = df_processed['who_survivor_victim_stay_with']
            else:
                df_processed[col] = 0
    for col in config.TOP_FEATURES:
        if df_processed[col].dtype in ['object', 'category']:
            df_processed[col] = df_processed[col].fillna('Unknown')
        else:
            df_processed[col] = df_processed[col].fillna(0)
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    for feature in categorical_features:
        if feature in encoders and feature in config.TOP_FEATURES:
            df_processed[feature] = df_processed[feature].astype(str)
            known_classes = set(encoders[feature].classes_)
            most_common = encoders[feature].classes_[0]
            df_processed[feature] = df_processed[feature].apply(lambda x: x if x in known_classes else most_common)
            df_processed[feature] = encoders[feature].transform(df_processed[feature])
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    for feature in numerical_features:
        if feature in scalers and feature in config.TOP_FEATURES:
            values = df_processed[feature].values.reshape(-1, 1)
            df_processed[feature] = scalers[feature].transform(values).flatten()
    return df_processed[config.TOP_FEATURES]

# ---------------------------------------------------------------------
def make_prediction(input_data: dict) -> dict:
    """Makes a prediction with detailed error handling and feature info."""
    try:
        print(f"Input data: {input_data}")
        
        engineered_df = engineer_features_for_prediction(input_data)
        processed_df = preprocess_input(engineered_df)
        
        processed_features = {}
        for feature_name in config.TOP_FEATURES:
            value = None
            if feature_name in engineered_df.columns:
                value = engineered_df[feature_name].iloc[0]
            elif feature_name in processed_df.columns:
                value = processed_df[feature_name].iloc[0]
            if value is not None:
                processed_features[feature_name] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
            else:
                processed_features[feature_name] = "N/A"

        risk_probability = model.predict_proba(processed_df)[0, 1]
        prediction = "High Risk" if risk_probability >= config.OPTIMAL_THRESHOLD else "Low Risk"
        confidence = abs(risk_probability - 0.5) * 2
        
        try:
            explanation = explanation_service.get_shap_explanation(processed_df)
            summary = explanation_service.get_generative_summary(
                explanation=explanation,
                input_data=input_data,
                prediction=prediction,
                risk_probability=risk_probability
            )
        except Exception as e:
            print(f"⚠️ Explanation service failed: {e}")
            explanation = {"risk_factors": [], "protective_factors": []}
            summary = "Explanation service unavailable. Showing baseline prediction."

        return {
            "prediction": prediction,
            "risk_probability": float(risk_probability),
            "confidence": float(confidence),
            "key_risk_factors": explanation.get("risk_factors", []),
            "key_protective_factors": explanation.get("protective_factors", []),
            "generative_summary": summary,
            "processed_features": processed_features
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in make_prediction: {str(e)}")
        print(f"Full traceback: {error_trace}")
        return {
            "prediction": "Error",
            "risk_probability": None,
            "confidence": None,
            "key_risk_factors": [],
            "key_protective_factors": [],
            "generative_summary": f"Prediction failed: {str(e)}",
            "processed_features": {}
        }