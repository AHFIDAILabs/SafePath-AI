from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the GBV Predictive Tool API"}

def test_prediction_endpoint():
    payload = {
        "survivor_age": 30,
        "survivor_sex": "Female",
        "marital_status": "Single",
        "educational_status": "Primary",
        "employment_status_victim_main": "Unemployed",
        "employment_status_main": "Unemployed", # Add this field to match the Pydantic model
        "who_survivor_victim_stay_with": "Alone",
        "PLWD": 0, "PLHIV": 0, "IDP": 0, "drug_user": 0,
        "widow": 0, "out_of_school_child": 0, "minor": 0, 'female_sex_worker'
        "household_help": 0, "child_apprentice": 0, "orphans": 0, "homeless": 0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "risk_probability" in data
    assert "generative_summary" in data