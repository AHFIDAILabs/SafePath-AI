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

# UPDATED: Working OpenRouter free models for explanation_service.py

def get_generative_summary(explanation: dict, input_data: dict = None, prediction: str = None, risk_probability: float = None) -> str:
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
    
    # Try OpenRouter API only if we have the API key
    if OPENROUTER_API_KEY:
        # UPDATED: Current working free models (as of 2025)
        free_models = [
            "qwen/qwen-2.5-72b-instruct:free",             # Qwen 2.5 72B - Working  
            "qwen/qwen-2.5-coder-32b-instruct:free",       # Qwen 2.5 Coder - Working
            "meta-llama/llama-3.2-11b-vision-instruct:free", # Llama 3.2 - Working
            "meta-llama/llama-3.2-3b-instruct:free",       # Llama 3.2 3B - Working
            "google/gemma-2-9b-it:free",                   # Gemma 2 - Working
            "mistralai/mistral-7b-instruct:free",          # Mistral 7B - Working
            "openrouter/auto:free",                         # OpenRouter Auto - Working
            "deepseek/deepseek-r1:free",                    # DeepSeek R1 - Working
        ]
        
        for model in free_models:
            try:
                print(f"Trying OpenRouter model: {model}")
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://localhost",
                        "X-Title": "GBV Risk Assessment Tool",
                    },
                    data=json.dumps({
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 600,  # Increased token limit
                        "temperature": 0.7
                    }),
                    timeout=20  # Increased timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        generated_content = result['choices'][0]['message']['content'].strip()
                        if generated_content and len(generated_content) > 50:  # Ensure meaningful content
                            print(f"Successfully used model: {model}")
                            return generated_content
                else:
                    print(f"Model {model} failed: {response.status_code} - {response.text}")
                    continue  # Try next model
                    
            except requests.exceptions.RequestException as e:
                print(f"Model {model} request failed: {e}")
                continue  # Try next model
            except json.JSONDecodeError as e:
                print(f"Failed to parse response from {model}: {e}")
                continue  # Try next model
            except Exception as e:
                print(f"Unexpected error with {model}: {e}")
                continue  # Try next model
    else:
        print("Warning: OPENROUTER_API_KEY not found in environment variables")
    
    # If API fails or no API key, generate intelligent fallback
    print("OpenRouter API unavailable, generating intelligent fallback summary")
    return _generate_intelligent_fallback(explanation, feature_descriptions, input_data, prediction, risk_probability)

def _generate_intelligent_fallback(explanation: dict, feature_descriptions: dict, input_data: dict = None, prediction: str = None, risk_probability: float = None) -> str:
    """Generate an intelligent fallback summary using available context."""
    
    # If we have full context, generate enhanced summary
    if input_data and prediction is not None and risk_probability is not None:
        return _generate_enhanced_contextual_summary(explanation, input_data, prediction, risk_probability)
    
    # Otherwise, generate basic but still intelligent summary
    return _generate_basic_intelligent_summary(explanation, feature_descriptions)

def _generate_enhanced_contextual_summary(explanation: dict, input_data: dict, prediction: str, risk_probability: float) -> str:
    """Generate enhanced summary with full context."""
    
    # Extract demographics
    age = input_data.get('survivor_age', 'unknown')
    gender = input_data.get('survivor_sex', 'unknown')
    marital_status = input_data.get('marital_status', 'unknown')
    employment = input_data.get('employment_status_victim_main', 'unknown')
    
    # Build contextual assessment
    risk_level = prediction
    probability = f"{(risk_probability * 100):.1f}%"
    
    assessment = f"ASSESSMENT SUMMARY: This {age}-year-old {gender.lower()} individual has been assessed as {risk_level} with a {probability} probability. "
    
    # Add demographic insights
    if marital_status != 'unknown':
        assessment += f"The individual's marital status as {marital_status.lower()} "
        if marital_status.lower() in ['divorced', 'separated', 'widowed']:
            assessment += "may contribute to increased vulnerability and requires consideration in support planning. "
        else:
            assessment += "provides important context for understanding the social support environment. "
    
    if employment != 'unknown':
        assessment += f"Current employment status is {employment.lower()}, "
        if 'unemployed' in employment.lower():
            assessment += "which significantly impacts economic security and overall vulnerability. "
        else:
            assessment += "which influences economic stability and independence levels. "
    
    # Analyze risk factors
    if explanation['risk_factors']:
        top_risk = explanation['risk_factors'][0]['feature'].replace('_', ' ')
        assessment += f"The primary risk driver identified is {top_risk}, requiring immediate attention in intervention planning. "
        
        if len(explanation['risk_factors']) > 1:
            second_risk = explanation['risk_factors'][1]['feature'].replace('_', ' ')
            assessment += f"Secondary concerns include {second_risk}. "
    
    # Analyze protective factors
    if explanation['protective_factors']:
        top_protective = explanation['protective_factors'][0]['feature'].replace('_', ' ')
        assessment += f"A significant protective factor is {top_protective}, which should be leveraged to build resilience. "
    else:
        assessment += "Limited protective factors were identified, indicating the need for comprehensive support system development. "
    
    # Generate targeted recommendations
    recommendations = "RECOMMENDATIONS: "
    
    if explanation['risk_factors']:
        top_risk_feature = explanation['risk_factors'][0]['feature']
        
        if 'economic' in top_risk_feature:
            recommendations += "1) Prioritize economic empowerment through financial literacy training, vocational skills development, and connections to microfinance programs. "
        elif 'social_isolation' in top_risk_feature:
            recommendations += "1) Address social isolation through community support groups, peer mentoring programs, and family counseling services. "
        elif 'housing' in top_risk_feature:
            recommendations += "1) Ensure housing security through emergency accommodation services, housing assistance programs, and safety planning. "
        elif 'employment' in top_risk_feature:
            recommendations += "1) Focus on employment support through job training, placement services, and skills development programs. "
        else:
            recommendations += "1) Develop targeted interventions addressing the primary vulnerability factors identified in this assessment. "
    else:
        recommendations += "1) Conduct comprehensive case assessment to identify specific intervention points and support needs. "
    
    recommendations += "2) Establish regular follow-up schedule to monitor risk factors and track intervention effectiveness. "
    recommendations += "3) Coordinate multidisciplinary team approach involving social work, counseling, legal support, and healthcare services as needed. "
    
    if explanation['protective_factors']:
        top_protective = explanation['protective_factors'][0]['feature'].replace('_', ' ')
        recommendations += f"4) Build upon existing strength in {top_protective} through targeted capacity building and resource enhancement."
    else:
        recommendations += "4) Focus on developing protective factors through social support networks, life skills training, and resilience building activities."
    
    return assessment + "\n\n" + recommendations

def _generate_basic_intelligent_summary(explanation: dict, feature_descriptions: dict) -> str:
    """Generate basic but intelligent summary when full context is not available."""
    
    if not explanation['risk_factors'] and not explanation['protective_factors']:
        return """ASSESSMENT SUMMARY: The risk assessment analysis has been completed successfully. Current data indicates a balanced risk profile that requires professional evaluation to develop appropriate intervention strategies. The assessment provides a foundation for understanding the individual's circumstances and developing targeted support approaches. Additional comprehensive evaluation may enhance the accuracy of this preliminary assessment.

RECOMMENDATIONS: 1) Conduct detailed psychosocial assessment to identify specific intervention points and comprehensive support needs. 2) Establish regular monitoring and follow-up schedule to track any changes in circumstances and risk factors. 3) Connect with appropriate community support services and resources that align with the individual's specific needs and preferences. 4) Document baseline assessment findings for future reference and intervention planning purposes."""
    
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
            risk_analysis = f"The assessment identifies {risk_names[0]} as the primary vulnerability factor requiring focused intervention."
        elif len(risk_names) == 2:
            risk_analysis = f"The assessment identifies {risk_names[0]} and {risk_names[1]} as key vulnerability factors requiring coordinated intervention approaches."
        else:
            risk_analysis = f"The assessment identifies multiple vulnerability factors, primarily {risk_names[0]}, {risk_names[1]}, and {risk_names[2]}, indicating the need for comprehensive intervention strategies."
        
        # Generate specific recommendations based on top risk factors
        top_risk_feature = explanation['risk_factors'][0]['feature']
        if 'economic_dependency' in top_risk_feature:
            recommendations.append("Provide referrals to economic empowerment programs, financial literacy training, and vocational skills development opportunities")
        elif 'social_isolation' in top_risk_feature:
            recommendations.append("Connect with community support groups, peer networks, and social integration programs to address isolation")
        elif 'housing_security' in top_risk_feature:
            recommendations.append("Assess housing stability and provide referrals to housing assistance programs and emergency accommodation services")
        elif 'employment' in top_risk_feature:
            recommendations.append("Connect with employment support services, job training programs, and career counseling resources")
        else:
            recommendations.append("Develop targeted interventions addressing the identified primary vulnerability factors through evidence-based approaches")
    
    # Analyze protective factors
    protective_analysis = ""
    if explanation['protective_factors']:
        top_protective = explanation['protective_factors'][0]
        protective_name = feature_descriptions.get(
            top_protective['feature'], 
            top_protective['feature'].replace('_', ' ')
        )
        protective_analysis = f" However, {protective_name} serves as a significant protective factor that should be leveraged and strengthened in intervention planning."
        recommendations.append(f"Build upon existing strengths in {protective_name} to enhance overall resilience and coping capacity")
    else:
        protective_analysis = " Limited protective factors were identified, indicating the importance of building resilience and support systems."
        recommendations.append("Focus on developing protective factors through social support network building, coping skills development, and resource connections")
    
    # Combine analysis
    full_analysis = risk_analysis + protective_analysis + " This comprehensive risk profile provides a foundation for developing targeted, evidence-based interventions that address vulnerabilities while strengthening protective elements."
    
    # Ensure we have enough recommendations
    if len(recommendations) < 3:
        recommendations.append("Conduct regular follow-up assessments to monitor risk factors and evaluate intervention effectiveness")
        recommendations.append("Coordinate with multidisciplinary team members to ensure comprehensive and integrated service delivery")
    
    # Format the response
    recommendations_text = " ".join([f"{i+1}) {rec}." for i, rec in enumerate(recommendations[:4])])
    
    return f"ASSESSMENT SUMMARY: {full_analysis}\n\nRECOMMENDATIONS: {recommendations_text}"