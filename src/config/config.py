# Configuration file for GBV Predictive Tool training
import os

# --- Project Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'backend', 'artifacts')
ANALYSIS_DIR = os.path.join(ARTIFACTS_DIR, 'analysis')

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Data Files ---
DATA_FILE = os.path.join(DATA_DIR, 'ngbv_data.csv')

# --- Artifact File Paths ---
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'gradient_boosting_gbv_model.pkl')
SCALERS_PATH = os.path.join(ARTIFACTS_DIR, 'scalers.pkl')
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl')
TOP_FEATURES_PATH = os.path.join(ANALYSIS_DIR, 'top_features.json')
OPTIMAL_THRESHOLD_PATH = os.path.join(ANALYSIS_DIR, 'optimal_threshold.json')

# --- Feature Engineering & Selection ---
# Top 12 features from the notebook's importance analysis
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

# All original input features required by the detailed engineering function
ORIGINAL_INPUT_FEATURES = [
    'employment_status_main',
    'employment_status_victim_main',
    'survivor_age',
    'survivor_sex',
    'widow',
    'out_of_school_child',
    'minor',
    'household_help',
    'child_apprentice',
    'orphans',
    'educational_status',
    'IDP',
    'marital_status',
    'who_survivor/victim_stay_with' # Note: API will map this to 'who_survivor_victim_stay_with'
    'PLHIV',
    'PLWD',
    'drug_user',
    'female_sex_worker'
]

# Vulnerability columns used for counting
VULNERABILITY_COLUMNS = [
    'PLWD', 'PLHIV', 'female_sex_worker', 'IDP', 'drug_user', 'widow',
    'out_of_school_child', 'minor', 'household_help', 'child_apprentice', 'orphans'
]

# --- Model & Training Parameters ---
TARGET_VARIABLE = 'vulnerability_target'

BEST_HYPERPARAMS = {
    'subsample': 0.8,
    'n_estimators': 300,
    'min_samples_split': 20,
    'min_samples_leaf': 8,
    'max_features': None,
    'max_depth': 5,
    'learning_rate': 0.2,
    'random_state': 42
}

# --- Preprocessing Parameters ---
ROBUST_SCALER_COLS = ['financial_access_proxy', 'community_connection_score']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load optimal threshold
try:
    import json
    with open(OPTIMAL_THRESHOLD_PATH, 'r') as f:
        threshold_data = json.load(f)
        OPTIMAL_THRESHOLD = threshold_data.get('optimal_threshold', 0.5)
except FileNotFoundError:
    OPTIMAL_THRESHOLD = 0.5  # Default threshold