# tests/test_preprocessing.py: Unit tests for data preprocessing functions
import os
import logging
import pytest
import pandas as pd
from src.data_processing import preprocessor
from src.config import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")


@pytest.fixture
def sample_dataframe():
    """
    Returns a larger, class-balanced sample dataset for stable stratified splits.
    The test dataset must contain enough rows per class to avoid
    ValueError during train_test_split with stratify=y.
    """
    data = {
        "survivor_age": [23, 35, 45, 17, 29, 41, 33, 26],
        "survivor_sex": ["Female", "Male", "Female", "Female", "Female", "Male", "Female", "Male"],
        "marital_status": ["Single", "Married", "Divorced", "Single", "Widowed", "Married", "Single", "Divorced"],
        "employment_status_victim_main": ["Unemployed", "Casual", "Formal", "Unemployed", "Formal", "Casual", "Formal", "Unemployed"],
        "educational_status": ["Some primary", "Secondary", "Tertiary", "Primary", "Tertiary", "Secondary", "Primary", "Tertiary"],
        "employment_status_main": ["Self employed", "Formal", "Casual", "Unemployed", "Formal", "Casual", "Unemployed", "Formal"],
        "location_state": ["State1", "State2", "State3", "State1", "State2", "State3", "State1", "State2"],
        "location_lga": ["LGA1", "LGA2", "LGA3", "LGA1", "LGA2", "LGA3", "LGA1", "LGA2"],
        "location_ward": ["Ward1", "Ward2", "Ward3", "Ward1", "Ward2", "Ward3", "Ward1", "Ward2"],
        "who_survivor/victim_stay_with": ["Guardian", "Spouse", "Alone", "Family", "Alone", "Guardian", "Spouse", "Family"],
        "PLWD": [0, 0, 1, 0, 1, 0, 0, 1],
        "PLHIV": [0, 1, 0, 0, 0, 1, 0, 0],
        "female_sex_worker": [0, 0, 0, 1, 0, 0, 1, 0],
        "IDP": [0, 0, 0, 0, 1, 1, 0, 0],
        "drug_user": [0, 0, 1, 0, 1, 0, 0, 0],
        "widow": [0, 0, 1, 0, 0, 0, 1, 0],
        "out_of_school_child": [0, 0, 0, 1, 0, 1, 0, 0],
        "minor": [0, 0, 0, 1, 0, 0, 0, 0],
        "household_help": [1, 0, 0, 1, 0, 0, 1, 0],
        "child_apprentice": [0, 1, 0, 0, 1, 0, 0, 0],
        "orphans": [1, 0, 0, 0, 0, 1, 0, 0],
        "vulnerability_target": [1, 0, 1, 0, 1, 0, 1, 0],  # Balanced target variable
    }
    return pd.DataFrame(data)


def test_engineer_features(sample_dataframe):
    """Ensure feature engineering generates all expected columns."""
    engineered_df = preprocessor.engineer_gbv_risk_features(sample_dataframe)

    if os.getenv("CI"):
        logger.debug("Engineered DataFrame head:\n%s", engineered_df.head())
        logger.debug("Engineered DataFrame columns: %s", list(engineered_df.columns))

    for feature in [
        "economic_dependency_score",
        "financial_access_proxy",
        "income_stability_score",
        "housing_security_score",
        "social_isolation_score",
        "community_connection_score"
    ]:
        assert feature in engineered_df.columns, f"{feature} missing in engineered features"


def test_preprocess_for_training_pipeline(sample_dataframe, monkeypatch):
    """
    Test full preprocessing pipeline with a balanced and adequate dataset
    to avoid stratified splitting errors in CI.
    """
    # Optional safeguard: ensure class balance before running preprocessing
    class_counts = sample_dataframe["vulnerability_target"].value_counts()
    assert len(class_counts) == 2, "Target variable must have both classes (0 and 1)"
    assert all(class_counts >= 2), "Each class in target variable must have at least 2 samples"

    monkeypatch.setattr(preprocessor, "load_data", lambda _: sample_dataframe)

    X_train, X_test, y_train, y_test, encoders, scalers = preprocessor.preprocess_for_training()

    if os.getenv("CI"):
        logger.debug("X_train shape: %s", X_train.shape)
        logger.debug("X_test shape: %s", X_test.shape)
        logger.debug("y_train distribution:\n%s", y_train.value_counts())
        logger.debug("Features used: %s", list(X_train.columns))

    assert not X_train.empty
    assert not X_test.empty
    assert config.TARGET_VARIABLE not in X_train.columns





# import os
# import logging
# import pytest
# import pandas as pd
# from src.data_processing import preprocessor
# from src.config import config

# # Configure logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

# @pytest.fixture
# def sample_dataframe():
#     data = {
#         "survivor_age": [23, 35, 45, 17],
#         "survivor_sex": ["Female", "Male", "Female", "Female"],
#         "marital_status": ["Single", "Married", "Divorced", "Single"],
#         "employment_status_victim_main": ["Unemployed", "Casual", "Formal", "Unemployed"],
#         "educational_status": ["Some primary", "Secondary", "Tertiary", "Primary"],
#         "employment_status_main": ["Self employed", "Formal", "Casual", "Unemployed"],
#         "location_state": ["State1", "State2", "State3", "State1"],
#         "location_lga": ["LGA1", "LGA2", "LGA3", "LGA1"],
#         "location_ward": ["Ward1", "Ward2", "Ward3", "Ward1"],
#         "who_survivor/victim_stay_with": ["Guardian", "Spouse", "Alone", "Family"],
#         "PLWD": [0, 0, 1, 0],
#         "PLHIV": [0, 1, 0, 0],
#         "female_sex_worker": [0, 0, 0, 1],
#         "IDP": [0, 0, 0, 0],
#         "drug_user": [0, 0, 1, 0],
#         "widow": [0, 0, 1, 0],
#         "out_of_school_child": [0, 0, 0, 1],
#         "minor": [0, 0, 0, 1],
#         "household_help": [1, 0, 0, 1],
#         "child_apprentice": [0, 1, 0, 0],
#         "orphans": [1, 0, 0, 0],
#         "vulnerability_target": [1, 0, 1, 0],
#     }
#     return pd.DataFrame(data)

# def test_engineer_features(sample_dataframe):
#     engineered_df = preprocessor.engineer_gbv_risk_features(sample_dataframe)

#     if os.getenv("CI"):  # Show detailed debug only in CI
#         logger.debug("Engineered DataFrame head:\n%s", engineered_df.head())
#         logger.debug("Engineered DataFrame columns: %s", list(engineered_df.columns))

#     for feature in [
#         "economic_dependency_score",
#         "financial_access_proxy",
#         "income_stability_score",
#         "housing_security_score",
#         "social_isolation_score",
#         "community_connection_score"
#     ]:
#         assert feature in engineered_df.columns, f"{feature} missing in engineered features"

# def test_preprocess_for_training_pipeline(sample_dataframe, monkeypatch):
#     monkeypatch.setattr(preprocessor, "load_data", lambda _: sample_dataframe)

#     X_train, X_test, y_train, y_test, encoders, scalers = preprocessor.preprocess_for_training()

#     if os.getenv("CI"):  # Show debug info in CI
#         logger.debug("X_train shape: %s", X_train.shape)
#         logger.debug("X_test shape: %s", X_test.shape)
#         logger.debug("y_train distribution:\n%s", y_train.value_counts())
#         logger.debug("Features used: %s", list(X_train.columns))

#     assert not X_train.empty
#     assert not X_test.empty
#     assert config.TARGET_VARIABLE not in X_train.columns