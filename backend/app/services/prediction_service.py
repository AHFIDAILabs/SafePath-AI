# backend/app/services/prediction_service.py: Feature engineering, preprocessing, prediction, and explanation services.  
import pandas as pd
import numpy as np
import json
from . import explanation_service
from ..config import config
from ..utils import utils

# Load artifacts on startup
model = utils.load_artifact(config.MODEL_PATH)
encoders = utils.load_artifact(config.ENCODERS_PATH)
scalers = utils.load_artifact(config.SCALERS_PATH)

def engineer_features_for_prediction(data: dict) -> pd.DataFrame:
    """Computes engineered features from a dictionary of input data for a single prediction."""
    df_features = pd.DataFrame([data])
    
    # Economic Dependency Score
    def calculate_economic_dependency(row):
        score = 0
        
        # Employment status scoring (parent/guardian)
        employment_main = str(row.get('employment_status_main', '')).strip()
        if employment_main == 'Unemployed':
            score += 3
        elif employment_main == 'Self employed':
            score += 1
        elif employment_main in ['Unknown', 'Not reported']:
            score += 1
        
        # Victim employment status
        employment_victim = str(row.get('employment_status_victim_main', '')).strip()
        if employment_victim == 'Unemployed':
            score += 2
        elif employment_victim == 'Self employed':
            score += 1
        elif employment_victim in ['Unknown', 'Not reported']:
            score += 1
        
        # Age-based dependency
        age = row.get('survivor_age', 0)
        if age < 18 or age > 65:
            score += 2
        
        # Vulnerability categories that increase economic dependency
        vulnerability_factors = ['widow', 'out_of_school_child', 'minor', 'household_help', 'child_apprentice', 'orphans']
        for factor in vulnerability_factors:
            if row.get(factor, 0) == 1:
                score += 1
        
        return min(score, 10)
    
    df_features['economic_dependency_score'] = df_features.apply(calculate_economic_dependency, axis=1)

    # Financial Access Proxy
    def financial_access_proxy(row):
        score = 5  # Baseline
        
        # Employment increases access
        employment_main = row.get('employment_status_main', '')
        if pd.notna(employment_main) and employment_main:
            employment_str = str(employment_main).lower()
            if 'formal' in employment_str:
                score += 3
            elif 'self-employed' in employment_str or 'self employed' in employment_str:
                score += 1
        
        # Education level
        educational_status = row.get('educational_status', '')
        if pd.notna(educational_status) and educational_status:
            education = str(educational_status).lower()
            if 'tertiary' in education or 'university' in education or 'undergraduate' in education or 'graduate' in education:
                score += 3
            elif 'secondary' in education or 'diploma' in education:
                score += 2
            elif 'primary' in education:
                score += 1
        
        # Vulnerability factors that reduce access
        reducing_factors = ['PLWD', 'PLHIV', 'female_sex_worker', 'IDP', 'drug_user', 'minor']
        for factor in reducing_factors:
            if row.get(factor, 0) == 1:
                score -= 2
        
        return max(score, 0)
    
    df_features['financial_access_proxy'] = df_features.apply(financial_access_proxy, axis=1)

    # Income Stability Score
    def income_stability_score(row):
        score = 5
        
        # Marital status affects stability
        marital_status = row.get('marital_status', '')
        if pd.notna(marital_status) and marital_status:
            marital = str(marital_status).lower()
            if 'married' in marital:
                score += 2
            elif 'divorced' in marital or 'separated' in marital:
                score -= 2
            elif 'single' in marital:
                score -= 1
        
        # Employment stability
        employment_main = row.get('employment_status_main', '')
        if pd.notna(employment_main) and employment_main:
            employment = str(employment_main).lower()
            if 'permanent' in employment or 'formal' in employment:
                score += 3
            elif 'casual' in employment or 'informal' in employment:
                score -= 1
            elif 'unemployed' in employment:
                score -= 3
        
        # Age stability (prime working age)
        age = row.get('survivor_age', 0)
        if 25 <= age <= 50:
            score += 1
        elif age < 18:
            score -= 2
        
        return max(score, 0)
    
    df_features['income_stability_score'] = df_features.apply(income_stability_score, axis=1)

    # Housing Security Score - FIXED VERSION
    def housing_security_score(row):
        score = 5
        
        # Who survivor lives with affects security
        living_arrangement = row.get('who_survivor_victim_stay_with', '')
        if pd.notna(living_arrangement) and living_arrangement:
            living_str = str(living_arrangement).lower()
            if 'spouse' in living_str or 'family' in living_str or 'partner' in living_str:
                score += 2
            elif 'alone' in living_str:
                score -= 1
            elif 'friends' in living_str:
                score += 1
        
        # Vulnerability factors affecting housing
        housing_risk_factors = ['IDP', 'orphans', 'PLWD', 'PLHIV', 'female_sex_worker', 'drug_user', 'minor']
        for factor in housing_risk_factors:
            if row.get(factor, 0) == 1:
                score -= 3
        
        # Employment affects housing security
        employment_main = row.get('employment_status_main', '')
        if pd.notna(employment_main) and employment_main:
            if 'unemployed' in str(employment_main).lower():
                score -= 2
        
        return max(score, 0)
    
    df_features['housing_security_score'] = df_features.apply(housing_security_score, axis=1)

    # Social Isolation Score
    def social_isolation_score(row):
        score = 0
        
        # Living alone increases isolation
        living_arrangement = row.get('who_survivor_victim_stay_with', '')
        if pd.notna(living_arrangement) and living_arrangement:
            if 'alone' in str(living_arrangement).lower():
                score += 3
        
        # Age-based isolation risk
        age = row.get('survivor_age', 0)
        if age > 65 or age < 18:
            score += 1
        
        # Certain vulnerabilities increase isolation
        isolation_factors = ['widow', 'PLHIV', 'PLWD', 'drug_user', 'female_sex_worker', 'orphans', 'IDP']
        for factor in isolation_factors:
            if row.get(factor, 0) == 1:
                score += 2
        
        # Marital status
        marital_status = row.get('marital_status', '')
        if pd.notna(marital_status) and marital_status:
            marital = str(marital_status).lower()
            if 'divorced' in marital or 'separated' in marital:
                score += 2
            elif 'widowed' in marital:
                score += 3
        
        return score
    
    df_features['social_isolation_score'] = df_features.apply(social_isolation_score, axis=1)

    # Community Connection Score
    def community_connection_score(row):
        score = 5
        
        # Marital status affects connections
        marital_status = row.get('marital_status', '')
        if pd.notna(marital_status) and marital_status:
            if 'married' in str(marital_status).lower():
                score += 2
        
        # Employment provides connections
        employment_main = row.get('employment_status_main', '')
        if pd.notna(employment_main) and employment_main:
            if 'unemployed' not in str(employment_main).lower():
                score += 2
        
        # Education provides connections
        educational_status = row.get('educational_status', '')
        if pd.notna(educational_status) and educational_status:
            education = str(educational_status).lower()
            if 'tertiary' in education or 'university' in education or 'undergraduate' in education or 'graduate' in education:
                score += 3
            elif 'secondary' in education or 'diploma' in education:
                score += 2
        
        # Some vulnerabilities reduce connections
        disconnection_factors = ['IDP', 'drug_user', 'out_of_school_child']
        for factor in disconnection_factors:
            if row.get(factor, 0) == 1:
                score -= 2
        
        return max(score, 0)
    
    df_features['community_connection_score'] = df_features.apply(community_connection_score, axis=1)

    # Handle column name mapping for the model
    if 'who_survivor_victim_stay_with' in df_features.columns:
        df_features['who_survivor/victim_stay_with'] = df_features['who_survivor_victim_stay_with']
        
    return df_features

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the input DataFrame for prediction - FIXED VERSION."""
    df_processed = df.copy()
    
    # Ensure we have all required columns
    for col in config.TOP_FEATURES:
        if col not in df_processed.columns:
            if col == 'who_survivor/victim_stay_with' and 'who_survivor_victim_stay_with' in df_processed.columns:
                df_processed[col] = df_processed['who_survivor_victim_stay_with']
            else:
                df_processed[col] = 0
    
    # Fill missing values
    for col in config.TOP_FEATURES:
        if col in df_processed.columns:
            if df_processed[col].dtype in ['object', 'category']:
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(0)
    
    # Apply encoders to categorical features - FIXED
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    for feature in categorical_features:
        if feature in encoders and feature in config.TOP_FEATURES:
            # Convert to string
            df_processed[feature] = df_processed[feature].astype(str)
            
            # Handle unseen categories
            known_classes = set(encoders[feature].classes_)
            current_values = df_processed[feature].unique()
            
            # Replace unseen values with the most common class
            most_common = encoders[feature].classes_[0]
            df_processed[feature] = df_processed[feature].apply(
                lambda x: x if x in known_classes else most_common
            )
            
            # Transform the feature
            df_processed[feature] = encoders[feature].transform(df_processed[feature])
    
    # Apply scalers to numerical features - FIXED
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    for feature in numerical_features:
        if feature in scalers and feature in config.TOP_FEATURES:
            # Use reshape(-1, 1) for single column transformation
            feature_values = df_processed[feature].values.reshape(-1, 1)
            df_processed[feature] = scalers[feature].transform(feature_values).flatten()
    
    # Return only the features the model expects
    return df_processed[config.TOP_FEATURES]

# UPDATED make_prediction function for prediction_service.py
def make_prediction(input_data: dict) -> dict:
    """Makes a prediction with detailed error handling and feature info."""
    try:
        print(f"Input data: {input_data}")
        
        if not all([model, encoders, scalers]):
            raise RuntimeError("Model artifacts are not loaded. Train the model first.")

        # Feature engineering
        print("Engineering features...")
        engineered_df = engineer_features_for_prediction(input_data)
        
        # Preprocessing
        print("Preprocessing...")
        processed_df = preprocess_input(engineered_df)
        
        # Complete processed_features handling in make_prediction function
        # Store the actual processed features for display in the frontend
        processed_features = {}

        # Include engineered features from the processed dataframe
        for feature_name in config.TOP_FEATURES:
            value = None
            
            # Priority 1: Get from engineered_df (for engineered features like scores)
            if feature_name in engineered_df.columns:
                value = engineered_df[feature_name].iloc[0]
                print(f"Got {feature_name} from engineered_df: {value}")
            
            # Priority 2: Get from processed_df (for original features that were scaled/encoded)
            elif feature_name in processed_df.columns:
                value = processed_df[feature_name].iloc[0]
                print(f"Got {feature_name} from processed_df: {value}")
            
            # Handle the value
            if value is not None:
                if isinstance(value, (int, float)):
                    processed_features[feature_name] = float(value)
                else:
                    processed_features[feature_name] = str(value)
            else:
                # This should rarely happen if feature engineering is working correctly
                processed_features[feature_name] = "N/A"
                print(f"Warning: {feature_name} not found in either dataframe")

        print(f"Final processed_features keys: {list(processed_features.keys())}")
        print(f"Sample values: {[(k, v) for k, v in list(processed_features.items())[:3]]}")
        
        # Prediction
        print("Making prediction...")
        risk_probability = model.predict_proba(processed_df)[0, 1]
        prediction = "High Risk" if risk_probability >= config.OPTIMAL_THRESHOLD else "Low Risk"
        confidence = abs(risk_probability - 0.5) * 2
        
        print(f"Risk probability: {risk_probability}, Prediction: {prediction}")
        
        # Get explanations
        explanation = explanation_service.get_shap_explanation(processed_df)
        
        # SIMPLIFIED: Get generative summary with full context for intelligent fallbacks
        summary = explanation_service.get_generative_summary(
            explanation=explanation,
            input_data=input_data,
            prediction=prediction,
            risk_probability=risk_probability
        )

        return {
            "prediction": prediction,
            "risk_probability": float(risk_probability),
            "confidence": float(confidence),
            "key_risk_factors": explanation['risk_factors'],
            "key_protective_factors": explanation['protective_factors'],
            "generative_summary": summary,
            "processed_features": processed_features
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in make_prediction: {str(e)}")
        print(f"Full traceback: {error_trace}")
        raise Exception(f"Prediction failed: {str(e)}") 