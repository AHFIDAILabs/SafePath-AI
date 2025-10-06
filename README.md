# 🧭 SafePath-AI: GBV Vulnerability Predictive Tool

SafePath-AI is a full-stack, data-driven machine learning application designed to *predict the vulnerability of individuals to Gender-Based Violence (GBV)* using socio-demographic and contextual indicators. It features a robust *Python-based ML pipeline, a FastAPI-powered prediction API, explainable AI (SHAP), and AI-generated summaries* to provide interpretable, human-readable insights that support case management and social intervention decisions. It is fully containerized for reliable deployment.
---

## Table of contents
* Project Structure
* Overview
* Installation and Setup
* Prediction API
* Core Features
* Testing
* Deployment
* Health Check
* Technology Stack
* License


## 🏗️ Project Structure
The project is organized into two main components: backend (the ML pipeline and API) and frontend (the user interface).

```text
SafePath-AI/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│       # Continuous Integration and Deployment configuration
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── predict.py
│   │   │       # FastAPI endpoint for prediction requests
│   │   ├── config/
│   │   │   └── config.py
│   │   │       # Model paths and artifact configuration
│   │   ├── models/
│   │   │   └── pydantic_models.py
│   │   │       # Input and output validation schemas
│   │   ├── services/
│   │   │   ├── prediction_service.py
│   │   │   │   # Feature engineering, preprocessing, and prediction
│   │   │   └── explanation_service.py
│   │   │       # SHAP explanations and AI-generated summaries
│   │   ├── utils/
│   │   │   └── utils.py
│   │   │       # Utility functions (artifact loading, logging, validation)
│   │   └── main.py
│   │       # FastAPI app setup, routes, and middleware
│   ├── artifacts/
│   │   ├── gradient_boosting_gbv_model.pkl
│   │   ├── scalers.pkl
│   │   ├── label_encoders.pkl
│   │   └── analysis/
│   │       ├── top_features.json
│   │       └── optimal_threshold.json
│   └── requirements.txt
├── src/
│   ├── config/
│   │   └── config.py
│   │       # Configuration for training pipeline
│   ├── data_processing/
│   │   └── preprocessor.py
│   │       # Data cleaning, encoding, and feature engineering for training
│   ├── training/
│   │   ├── train.py
│   │   └── trainer.py
│   │       # Model training orchestration and artifact saving
│   └── evaluation/
│       └── evaluator.py
│           # Model performance metrics and threshold optimization
├── data/
│   ├── raw/
│   └── preprocessed/
│       # Training and testing datasets
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   ├── js/
│   │   │   └── dashboard.js
│   │   └── images/
│   │       ├── ahfid_logo.png
│   │       └── home.png
│   └── templates/
│       ├── index.html
│       ├── dashboard.html
│       └── faq.html
├── notebooks/
│   # Jupyter notebooks for experimentation and analysis
├── Dockerfile
│   # Docker configuration for containerized deployment
├── render.yaml
│   # Render.com deployment configuration
├── pytest.ini
│   # Pytest configuration
├── .env
│   # Environment variables (API keys, etc.)
├── .gitignore
│
└── README.md
```

## 📖 Overview 
SafePath AI provides:

* **Risk Assessment** – A model that ingests demographic, behavioural and contextual data to compute a GBV risk score.
* **Explanation Engine** – Extracts the most influential risk and protective factors from the model output.
* **Recommendation Generator** – Produces actionable recommendations based on the top risk & protective factors.
    
The backend exposes a simple REST API (`/predict`) that accepts JSON payloads and returns a structured response

## ⚙️ Installation and Setup
1. Prerequisites
    * Python 3.11 or newer
    * Docker (optional, for containerized environments)
    * Model artifacts must exist in backend/artifacts
    * OpenRouter API key (for AI-generated summaries)

2. Clone the Repository
```
    git clone https://github.com/<your-org>/SafePath-AI.git
    cd SafePath-AI
```

3. Backend Installation
```
    cd backend
    pip install -r requirements.txt
```
4. Environment Configuration
    Create a .env file in the project root:
```
    OPENROUTER_API_KEY=your_api_key_here
```
If this key is not provided, the API will automatically use an intelligent fallback summary generator.

5. Run the API Server
```
    uvicorn app.main:app --reload --port 8000
```
Access the API documentation at: http://localhost:8000/docs

## 🧠 Prediction API
Endpoint: POST /api/v1/predict

Sample Request:
```
    {
    "survivor_age": 25,
    "survivor_sex": "Female",
    "marital_status": "Married",
    "educational_status": "Secondary",
    "employment_status_main": "Self employed",
    "employment_status_victim_main": "Unemployed",
    "who_survivor_victim_stay_with": "Partner"
    }
```
Sample Response:
```
    {
    "prediction": "Low Risk",
    "risk_probability": 0.15,
    "confidence": 0.85,
    "key_risk_factors": [
        { "feature": "economic_dependency_score", "impact": 0.12 }
    ],
    "key_protective_factors": [
        { "feature": "community_connection_score", "impact": -0.25 }
    ],
    "generative_summary": "ASSESSMENT SUMMARY: The individual demonstrates...",
    "processed_features": { "economic_dependency_score": 5.0, "survivor_sex": "Female" }
}
```

## 🧩 Core Features
* **Feature Engineering:** Generates risk and protection indicators from input data.
* **RESTful Prediction API:** Fast and reliable endpoint for risk prediction.
* **Web Interface:** Serves static content via Jinja2 templates (index, dashboard, FAQ).
* **Explainability:** Uses SHAP to reveal key contributing factors for each prediction.
* **AI Summaries:** Generates human-readable assessments and recommendations via OpenRouter.
* **Resilience:** Includes fallback logic for missing models or unavailable APIs.
* **Automation (Robust CI/CD):** CI/CD pipeline for automated testing, linting, and Docker builds via GitHub Actions.
* **Scalability (Containerized Deployment):** Dockerized for fast and consistent deployment across environments.
* **Health Checks:** Dedicated endpoint for monitoring service status.

## 🧪 Testing
Run automated tests:
```bash
    pytest --maxfail=1 --disable-warnings -q
```

## 🚀 Deployment
Option 1: Using Docker
```
    docker build -t gbv-predictive-tool .
    docker run -p 8000:8000 gbv-predictive-tool
```

Option 2: Render Deployment
Render automatically deploys using:
    * Dockerfile
    * .github/workflows/ci-cd.yml
    * render.yaml

## 🩺 Health Check
API health endpoint:
```
    GET /health
```
Response: 
```
    { "status": "healthy" }
```
## 🧰 Technology Stack

| Component            | Technology         |
| -------------------- | ------------------ |
| Framework            | FastAPI            |
| ML Model             | Gradient Boosting  |
| Explainability       | SHAP               |
| Generative Summaries | OpenRouter API     |
| CI/CD                | GitHub Actions     |
| Deployment           | Docker, Render.com |
| Language             | Python 3.11        |

## 📜 License
This project is distributed under the MIT License.

## 🤝 Contributors

* Lead Developer: ’Wale Ogundeji
* Contributors: AHFID AI Team