# backend/app/config/config.py: Configuration file for paths and parameters used in the GBV vulnerability prediction model.

import os
import json

# --- Project Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'backend', 'artifacts')
ANALYSIS_DIR = os.path.join(ARTIFACTS_DIR, 'analysis')

# --- Artifact File Paths ---
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'gradient_boosting_gbv_model.pkl')
SCALERS_PATH = os.path.join(ARTIFACTS_DIR, 'scalers.pkl')
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl')
TOP_FEATURES_PATH = os.path.join(ANALYSIS_DIR, 'top_features.json')
OPTIMAL_THRESHOLD_PATH = os.path.join(ANALYSIS_DIR, 'optimal_threshold.json')


# --- Load Analysis Artifacts ---
try: 
    with open(TOP_FEATURES_PATH, 'r') as f:
        TOP_FEATURES = json.load(f)
    with open(OPTIMAL_THRESHOLD_PATH, 'r') as f:
        OPTIMAL_THRESHOLD = json.load(f)['optimal_threshold']
except FileNotFoundError:
    print("Warning: Analysis files not found. Using default values.")
    TOP_FEATURES = []
    OPTIMAL_THRESHOLD = 0.5
    
# Features to be scaled with RobustScaler (from training)
ROBUST_SCALER_COLS = ['financial_access_proxy', 'community_connection_score']

# backend/app/config/config.py: Configuration file for paths and parameters used in the GBV vulnerability prediction model.

import os
import json

# --- Project Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'backend', 'artifacts')
ANALYSIS_DIR = os.path.join(ARTIFACTS_DIR, 'analysis')

# --- Artifact File Paths ---
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'gradient_boosting_gbv_model.pkl')
SCALERS_PATH = os.path.join(ARTIFACTS_DIR, 'scalers.pkl')
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl')
TOP_FEATURES_PATH = os.path.join(ANALYSIS_DIR, 'top_features.json')
OPTIMAL_THRESHOLD_PATH = os.path.join(ANALYSIS_DIR, 'optimal_threshold.json')

# --- Load Analysis Artifacts ---
try:
    with open(TOP_FEATURES_PATH, 'r') as f:
        TOP_FEATURES = json.load(f)
    with open(OPTIMAL_THRESHOLD_PATH, 'r') as f:
        OPTIMAL_THRESHOLD = json.load(f)['optimal_threshold']
    print(f"Loaded {len(TOP_FEATURES)} features and threshold {OPTIMAL_THRESHOLD}")
except FileNotFoundError as e:
    print(f"Warning: Analysis files not found: {e}. Using default values.")
    # Default features based on your training code
    TOP_FEATURES = [
        'economic_dependency_score',
        'survivor_sex',
        'survivor_age',
        'who_survivor/victim_stay_with', 
        'income_stability_score',
        'housing_security_score',
        'social_isolation_score',
        'employment_status_victim_main',
        'educational_status',
        'community_connection_score',
        'marital_status',
        'financial_access_proxy'
    ]
    OPTIMAL_THRESHOLD = 0.5
    
# Features to be scaled with RobustScaler (from training)
ROBUST_SCALER_COLS = ['financial_access_proxy', 'community_connection_score']