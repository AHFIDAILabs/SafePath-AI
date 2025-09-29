import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

# Initialize the test client for the FastAPI application
client = TestClient(app)

@pytest.fixture
def valid_payload():
    """
    Provides a valid, complete payload for the prediction endpoint.
    This fixture ensures all required fields are present and correctly formatted.
    """
    return {
        "survivor_age": 28,
        "survivor_sex": "Female",
        "marital_status": "Single",
        "educational_status": "Completed secondary",
        "employment_status_victim_main": "Unemployed",
        "employment_status_main": "Unemployed",
        "who_survivor_victim_stay_with": "Alone",
        "PLWD": 1,
        "PLHIV": 0,
        "IDP": 1,
        "drug_user": 0,
        "widow": 0,
        "out_of_school_child": 0,
        "minor": 0,
        "household_help": 0,
        "child_apprentice": 0,
        "orphans": 0,
        "female_sex_worker": 0
    }

def test_read_root():
    """Tests the root endpoint to ensure the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the GBV Predictive Tool API"}

def test_health_check():
    """Tests the health check endpoint for monitoring purposes."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_success(valid_payload):
    """
    Tests a successful prediction request with a valid payload.
    It verifies the response structure, status code, and data types.
    """
    response = client.post("/api/v1/predict", json=valid_payload)
    
    # Assert a successful response
    assert response.status_code == 200
    data = response.json()
    
    # Assert that all expected keys are present in the response
    expected_keys = [
        "prediction", "risk_probability", "confidence",
        "key_risk_factors", "key_protective_factors",
        "generative_summary", "processed_features"
    ]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from response"

    # Assert the data types of the response fields
    assert isinstance(data["prediction"], str)
    assert isinstance(data["risk_probability"], float)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["key_risk_factors"], list)
    assert isinstance(data["key_protective_factors"], list)
    assert isinstance(data["generative_summary"], str)
    assert isinstance(data["processed_features"], dict)

    # Assert specific value constraints
    assert data["prediction"] in ["High Risk", "Low Risk"]
    assert 0.0 <= data["risk_probability"] <= 1.0

def test_prediction_missing_field(valid_payload):
    """
    Tests the API's response to a payload with a missing required field.
    It expects a 422 Unprocessable Entity error, as handled by FastAPI.
    """
    invalid_payload = valid_payload.copy()
    del invalid_payload["survivor_age"]  # Remove a required field
    
    response = client.post("/api/v1/predict", json=invalid_payload)
    
    # Assert that the API correctly identifies the validation error
    assert response.status_code == 422
    assert "detail" in response.json()

def test_prediction_invalid_type(valid_payload):
    """
    Tests the API's response to a payload with an incorrect data type.
    It expects a 422 Unprocessable Entity error.
    """
    invalid_payload = valid_payload.copy()
    invalid_payload["survivor_age"] = "thirty"  # Invalid data type for an integer field
    
    response = client.post("/api/v1/predict", json=invalid_payload)
    
    # Assert that the API correctly identifies the type mismatch
    assert response.status_code == 422
    assert "detail" in response.json()