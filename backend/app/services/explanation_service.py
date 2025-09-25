# backend/app/services/explanation_service.py: Explanation Service - Generates SHAP explanations and AI summaries
# This module provides functionality to generate SHAP-based explanations for model predictions and
# AI-generated summaries and recommendations of the results.
import shap
import pandas as pd
import requests
import json
import os
import numpy as np
from dotenv import load_dotenv
from ..config import config
from ..utils import utils

# Load environment variables
load_dotenv()

# --- Initialize on Startup ---
model = utils.load_artifact(config.MODEL_PATH)
explainer = shap.TreeExplainer(model) if model else None

# --- Configure OpenRouter API ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_shap_explanation(processed_df: pd.DataFrame) -> dict:
    """Generates a SHAP-based explanation for a single prediction."""
    if not explainer:
        return {"risk_factors": [], "protective_factors": []}

    try:
        shap_values = explainer.shap_values(processed_df)
        
        # Handle both binary classification (returns list) and single array
        if isinstance(shap_values, list):
            # For binary classification, use positive class (index 1)
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
        
        # Ensure we have a 1D array for a single prediction
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]  # Take first row if 2D
        
        print(f"SHAP values shape: {shap_vals.shape}")
        print(f"SHAP values: {shap_vals}")
        print(f"TOP_FEATURES length: {len(config.TOP_FEATURES)}")
        print(f"TOP_FEATURES: {config.TOP_FEATURES}")
        
        # Ensure lengths match
        if len(shap_vals) != len(config.TOP_FEATURES):
            print(f"Warning: SHAP values length ({len(shap_vals)}) != TOP_FEATURES length ({len(config.TOP_FEATURES)})")
            min_len = min(len(shap_vals), len(config.TOP_FEATURES))
            shap_vals = shap_vals[:min_len]
            features = config.TOP_FEATURES[:min_len]
        else:
            features = config.TOP_FEATURES
        
        # Create contributions list - FIXED VERSION
        contributions = []
        for i, (feature, value) in enumerate(zip(features, shap_vals)):
            # Convert numpy types to Python floats to avoid array comparison issues
            contribution_value = float(value)
            contributions.append((feature, contribution_value))
        
        # Sort by absolute impact
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        # Separate risk and protective factors - FIXED VERSION
        risk_factors = []
        protective_factors = []
        
        for feature, impact in contributions:
            if impact > 0:  # Now impact is a regular float, not an array
                risk_factors.append({"feature": feature, "impact": impact})
            elif impact < 0:
                protective_factors.append({"feature": feature, "impact": impact})
        
        # Return top factors
        return {
            "risk_factors": risk_factors[:5],
            "protective_factors": protective_factors[:3]
        }
        
    except Exception as e:
        print(f"Error in get_shap_explanation: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {"risk_factors": [], "protective_factors": []}

def get_generative_summary(explanation: dict) -> str:
    """Uses OpenRouter API to create a comprehensive summary with separate recommendations."""
    
    # Create feature descriptions for better understanding
    feature_descriptions = {
        'economic_dependency_score': 'economic dependency',
        'survivor_sex': 'gender',
        'survivor_age': 'age',
        'who_survivor/victim_stay_with': 'living arrangement',
        'income_stability_score': 'income stability',
        'housing_security_score': 'housing security',
        'social_isolation_score': 'social isolation',
        'employment_status_victim_main': 'employment status',
        'educational_status': 'educational background',
        'community_connection_score': 'community connections',
        'marital_status': 'marital status',
        'financial_access_proxy': 'access to financial services'
    }
    
    # Format risk and protective factors with descriptions
    risk_descriptions = []
    for factor in explanation['risk_factors'][:4]:  # Top 4 risk factors for more detail
        feature_name = factor['feature']
        description = feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
        risk_descriptions.append(f"{description} (impact: {factor['impact']:.2f})")
    
    protective_descriptions = []
    for factor in explanation['protective_factors'][:3]:  # Top 3 protective factors
        feature_name = factor['feature']
        description = feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
        protective_descriptions.append(f"{description} (impact: {factor['impact']:.2f})")
    
    prompt = f"""You are an experienced social worker analyzing GBV risk factors. Based on the analysis below, provide a comprehensive but concise assessment with two distinct sections:

1. ASSESSMENT SUMMARY (4-5 sentences): Analyze the risk profile by explaining the key vulnerability factors and protective elements. Discuss how these factors interact and what they reveal about the individual's overall situation. Use professional language suitable for case documentation.

2. RECOMMENDATIONS (3-4 specific actionable items): Based on the risk factors identified, provide targeted intervention recommendations. Focus on evidence-based approaches that address the specific vulnerabilities while building on existing protective factors.

Analysis Data:
Key Risk Factors (higher scores = increased vulnerability):
{'; '.join(risk_descriptions) if risk_descriptions else 'No significant risk factors identified'}

Key Protective Factors (negative scores = reduced vulnerability):
{'; '.join(protective_descriptions) if protective_descriptions else 'Limited protective factors identified'}

Format your response exactly as:
ASSESSMENT SUMMARY: [Your detailed summary here]

RECOMMENDATIONS: [Your specific recommendations here]"""
    
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not found in environment variables")
        return _generate_fallback_summary(explanation, feature_descriptions)
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-gbv-app.com",  # Replace with your actual site
                "X-Title": "GBV Risk Assessment Tool",  # Replace with your app name
            },
            data=json.dumps({
                "model": "openai/gpt-oss-120b:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
        else:
            print(f"OpenRouter API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"OpenRouter API request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse OpenRouter API response: {e}")
    except Exception as e:
        print(f"Unexpected error with OpenRouter API: {e}")
    
    # Fallback to rule-based summary if API fails
    return _generate_fallback_summary(explanation, feature_descriptions)

def _generate_fallback_summary(explanation: dict, feature_descriptions: dict) -> str:
    """Generate a comprehensive rule-based summary with separate recommendations when API is unavailable."""
    
    if not explanation['risk_factors'] and not explanation['protective_factors']:
        return """ASSESSMENT SUMMARY: The risk assessment analysis has been completed. Current data indicates no significant risk factors or protective factors were identified in this evaluation. The individual's risk profile appears to be within normal parameters based on the provided information. Additional assessment may be needed to capture a complete picture of the situation.

RECOMMENDATIONS: 1) Conduct a more detailed psychosocial assessment to identify any factors not captured in the initial screening. 2) Establish regular check-ins to monitor any changes in circumstances. 3) Provide information about available support services and resources. 4) Document baseline assessment for future reference."""
    
    # Analyze risk factors
    risk_analysis = ""
    recommendations = []
    
    if explanation['risk_factors']:
        top_risks = explanation['risk_factors'][:3]
        risk_names = []
        for factor in top_risks:
            feature_name = factor['feature']
            description = feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
            risk_names.append(description)
        
        if len(risk_names) == 1:
            risk_analysis = f"The assessment identifies {risk_names[0]} as the primary vulnerability factor."
        elif len(risk_names) == 2:
            risk_analysis = f"The assessment identifies {risk_names[0]} and {risk_names[1]} as key vulnerability factors."
        else:
            risk_analysis = f"The assessment identifies multiple vulnerability factors, primarily {risk_names[0]}, {risk_names[1]}, and {risk_names[2]}."
        
        # Generate specific recommendations based on top risk factors
        top_risk_feature = explanation['risk_factors'][0]['feature']
        if 'economic_dependency' in top_risk_feature:
            recommendations.append("Provide referrals to economic empowerment programs, financial literacy training, or vocational skills development")
        elif 'social_isolation' in top_risk_feature:
            recommendations.append("Connect with community support groups, peer networks, or social integration programs")
        elif 'housing_security' in top_risk_feature:
            recommendations.append("Assess housing stability and provide referrals to housing assistance programs or emergency shelter services")
        elif 'employment' in top_risk_feature:
            recommendations.append("Connect with employment support services, job training programs, or career counseling resources")
        else:
            recommendations.append("Develop targeted interventions addressing the identified primary vulnerability factors")
    
    # Analyze protective factors
    protective_analysis = ""
    if explanation['protective_factors']:
        top_protective = explanation['protective_factors'][0]
        protective_name = feature_descriptions.get(
            top_protective['feature'], 
            top_protective['feature'].replace('_', ' ')
        )
        protective_analysis = f" However, {protective_name} serves as a significant protective factor that should be leveraged in intervention planning."
        recommendations.append(f"Build upon existing strengths in {protective_name} to enhance overall resilience and coping capacity")
    else:
        protective_analysis = " Limited protective factors were identified, indicating a need for comprehensive support system development."
        recommendations.append("Focus on building protective factors through social support networks, coping skills development, and resource connections")
    
    # Combine analysis
    full_analysis = risk_analysis + protective_analysis + " This risk profile suggests the need for targeted, evidence-based interventions to address identified vulnerabilities while strengthening protective elements."
    
    # Ensure we have at least 3 recommendations
    if len(recommendations) < 3:
        recommendations.append("Conduct regular follow-up assessments to monitor risk factors and intervention effectiveness")
        recommendations.append("Coordinate with multidisciplinary team members to ensure comprehensive service delivery")
    
    # Format the response
    recommendations_text = " ".join([f"{i+1}) {rec}." for i, rec in enumerate(recommendations[:4])])
    
    return f"ASSESSMENT SUMMARY: {full_analysis}\n\nRECOMMENDATIONS: {recommendations_text}"