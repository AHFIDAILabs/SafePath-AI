# backend/app/utils/utils.py: Utility functions for model handling, data validation, and other helpers.
import joblib
import os
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

def load_artifact(file_path: str) -> Optional[Any]:
    """Load a joblib artifact (model, scaler, encoder, etc.)"""
    if not os.path.exists(file_path):
        print(f"Warning: Artifact file not found at {file_path}")
        return None
    
    try:
        artifact = joblib.load(file_path)
        print(f"Successfully loaded artifact from {file_path}")
        return artifact
    except Exception as e:
        print(f"Error loading artifact from {file_path}: {e}")
        return None

def save_artifact(artifact: Any, file_path: str) -> bool:
    """Save a joblib artifact to file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(artifact, file_path)
        print(f"Successfully saved artifact to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving artifact to {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict]:
    """Load JSON data from file"""
    if not os.path.exists(file_path):
        print(f"Warning: JSON file not found at {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def save_json(data: Dict, file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved JSON to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def validate_input_data(data: Dict) -> tuple[bool, str]:
    """Validate input data structure and types"""
    required_fields = [
        'survivor_age', 'survivor_sex', 'marital_status', 
        'educational_status', 'employment_status_main',
        'employment_status_victim_main', 'who_survivor/victim_stay_with'
    ]
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate age
    try:
        age = int(data['survivor_age'])
        if age < 1 or age > 120:
            return False, "Age must be between 1 and 120"
    except (ValueError, TypeError):
        return False, "Age must be a valid integer"
    
    # Validate binary fields
    binary_fields = [
        'PLWD', 'PLHIV', 'IDP', 'drug_user', 'widow', 
        'out_of_school_child', 'minor', 'household_help', 
        'child_apprentice', 'orphans', 'female_sex_worker'
    ]
    
    for field in binary_fields:
        if field in data:
            if data[field] not in [0, 1]:
                return False, f"{field} must be 0 or 1"
    
    return True, "Valid"

def sanitize_feature_name(feature_name: str) -> str:
    """Convert feature names for display purposes"""
    # Mapping for common feature names
    name_mapping = {
        'survivor_age': 'Age',
        'survivor_sex': 'Gender',
        'marital_status': 'Marital Status',
        'educational_status': 'Education Level',
        'employment_status_main': 'Main Provider Employment',
        'employment_status_victim_main': 'Employment Status',
        'who_survivor/victim_stay_with': 'Living Arrangement',
        'economic_dependency_score': 'Economic Dependency',
        'financial_access_proxy': 'Financial Access',
        'income_stability_score': 'Income Stability',
        'housing_security_score': 'Housing Security',
        'social_isolation_score': 'Social Isolation',
        'community_connection_score': 'Community Connections',
        'PLWD': 'Person with Disability',
        'PLHIV': 'Person Living with HIV',
        'IDP': 'Internally Displaced Person',
        'female_sex_worker': 'Sex Worker'
    }
    
    if feature_name in name_mapping:
        return name_mapping[feature_name]
    
    # Default formatting: replace underscores with spaces and title case
    return feature_name.replace('_', ' ').title()

def check_artifacts_exist(artifact_paths: Dict[str, str]) -> tuple[bool, list]:
    """Check if all required artifacts exist"""
    missing_artifacts = []
    
    for name, path in artifact_paths.items():
        if not os.path.exists(path):
            missing_artifacts.append(f"{name}: {path}")
    
    return len(missing_artifacts) == 0, missing_artifacts

def log_prediction_request(input_data: Dict, prediction_result: Dict) -> None:
    """Log prediction request for monitoring (without storing sensitive data)"""
    log_entry = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'input_summary': {
            'age': input_data.get('survivor_age'),
            'sex': input_data.get('survivor_sex'),
            'vulnerability_count': sum([
                input_data.get('PLWD', 0),
                input_data.get('PLHIV', 0),
                input_data.get('IDP', 0),
                input_data.get('drug_user', 0),
                input_data.get('widow', 0),
                input_data.get('out_of_school_child', 0),
                input_data.get('minor', 0),
                input_data.get('household_help', 0),
                input_data.get('child_apprentice', 0),
                input_data.get('orphans', 0),
                input_data.get('female_sex_worker', 0)
            ])
        },
        'prediction': prediction_result.get('prediction'),
        'risk_probability': prediction_result.get('risk_probability'),
        'confidence': prediction_result.get('confidence')
    }
    
    print(f"Prediction logged: {log_entry['prediction']} "
          f"(probability: {log_entry['risk_probability']:.3f}, "
          f"confidence: {log_entry['confidence']:.3f})")

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    return {
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'working_directory': os.getcwd(),
        'environment_variables': {
            key: value for key, value in os.environ.items() 
            if key.startswith(('OPENROUTER', 'FASTAPI', 'PYTHON'))
        }
    }