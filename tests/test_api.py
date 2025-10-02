import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
import json

# Initialize the test client
client = TestClient(app)

@pytest.fixture
def valid_payload():
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
        "female_sex_worker": 0,
    }

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

    if os.getenv("CI") == "true":
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"message": "Welcome to the GBV Predictive Tool API"}
    else:
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text or "<html" in response.text

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_success(valid_payload):
    response = client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    expected_keys = [
        "prediction", "risk_probability", "confidence",
        "key_risk_factors", "key_protective_factors",
        "generative_summary", "processed_features"
    ]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from response"

    assert isinstance(data["prediction"], str)
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert isinstance(data["confidence"], float)
    assert isinstance(data["key_risk_factors"], list)
    assert isinstance(data["key_protective_factors"], list)
    assert isinstance(data["generative_summary"], str)
    assert isinstance(data["processed_features"], dict)
    assert data["prediction"] in ["High Risk", "Low Risk"]

def test_prediction_missing_field(valid_payload):
    invalid_payload = valid_payload.copy()
    del invalid_payload["survivor_age"]
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_prediction_invalid_type(valid_payload):
    invalid_payload = valid_payload.copy()
    invalid_payload["survivor_age"] = "thirty"
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()