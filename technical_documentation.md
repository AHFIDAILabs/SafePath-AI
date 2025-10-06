# 📘 Technical Documentation

## Overview

The backend predicts GBV vulnerability levels using a pre-trained Gradient Boosting Classifier. Input data passes through feature engineering, encoding, and scaling before prediction.

Predictions are explained via SHAP and summarized using generative AI.

1. Workflow

    1. *Input Validation* – Uses Pydantic (PredictionInput)
    2. *Feature Engineering* – Generates engineered variables such as:
        * economic_dependency_score
        * housing_security_score
        * social_isolation_score
        * community_connection_score
        * financial_access_proxy
    3. *Preprocessing* – Encodes categorical variables, scales numerical ones.
    4. *Prediction* – Computes probability using the trained model.
    5. *Explainability* – SHAP identifies top positive and negative contributors.
    6. *Summarization* – OpenRouter (or fallback generator) produces assessment text.

    High level flow:

    1. API receives validated JSON input via Pydantic models
    2. The prediction service computes derived features from raw fields
    3. Data is preprocessed using saved encoders and scalers
    4. The model predicts probability for the positive class
    5. SHAP values generate explanation items for risk and protection
    6. An AI summary is requested from OpenRouter, if available; if not available a local summary is generated
    7. Final response includes prediction label, probability, confidence score, top risk items, top protective items, a narrative summary, and the processed feature set

2. Key Modules and responsibilities 

| Module | Function |
|---------|-----------|
| `backend/app/api/predict.py` | Defines API endpoints, handles request validation, and maps errors to HTTP responses. |
| `backend/app/services/prediction_service.py` | Executes feature engineering, preprocessing, model inference, and assembles the final response payload. |
| `backend/app/services/explanation_service.py` | Initializes SHAP explainers, extracts top risk and protective contributors, and generates narrative summaries via OpenRouter with a local fallback option. |
| `backend/app/models/pydantic_models.py` | Defines request and response schemas using Pydantic, including examples for automatic API documentation. |
| `backend/app/config/config.py` | Specifies artifact paths, model thresholds, and global runtime constants such as feature lists and version tags. |
| `backend/app/utils/utils.py` | Manages artifact loading, JSON and file I/O utilities, validation helpers, and lightweight application logging. |
| `src/data_processing/preprocessor.py` | Cleans, encodes, and transforms raw data into feature vectors used for model training and evaluation. |
| `src/evaluation/evaluator.py` | Computes model performance metrics and determines the optimal operating threshold using ROC-based analysis. |
| `src/training/train.py` | Coordinates the end-to-end model training workflow, including feature preparation, fitting, and artifact serialization. |

3. Feature engineering summary

Derived scores are created from raw fields to capture risk drivers and protective elements. Key engineered variables include

* economic_dependency_score, calculated from employment fields age and vulnerability flags
* financial_access_proxy, combining employment, education and vulnerability adjustments
* income_stability_score, using marital status, employment and age banding
* housing_security_score, driven by living arrangement and vulnerability flags
* social_isolation_score, based on living arrangement age and vulnerability flags
* community_connection_score, combining education, employment and marital status

The final model uses a subset defined in top_features.json to ensure consistent input ordering for SHAP and model inference.

4. Model artifacts (Artifacts Directory)

Artifacts required for prediction are stored in:
```
    backend/artifacts/
    ├── gradient_boosting_gbv_model.pkl
    ├── scalers.pkl
    ├── label_encoders.pkl
    └── analysis/
        ├── top_features.json
        └── optimal_threshold.json
```
The app loads artifacts at startup and falls back to safe defaults or a dummy model when artifacts are missing.

5. Explainability and narratives

    * SHAP is used to compute per case contributions
    * Positive contribution values increase predicted risk, negative values reduce it
    * Top contributors are formatted into two lists, one for risk items and one for protective items
    * OpenRouter is used to convert those lists into a structured assessment summary and action oriented recommendations, if the API key is configured
    * If OpenRouter is not available the service creates a structured fallback summary using rule based templates

6. Error handling and resilience

    * Missing artifacts result in a DummyModel that returns neutral probabilities, the service logs warnings and returns informative messages
    * Exceptions in prediction path are caught and returned as structured error payloads, the API uses standard HTTP error codes
    * Missing analysis files → defaults to pre-defined top features
    * Explanation or summary failures do not block a prediction, they trigger fallback text and allow normal operation, generates a local intelligent summary

7. Logging and observability

    * Prediction requests are light logged with a timestamp, anonymized input summary and prediction outcome
    * `utils.get_system_info` provides runtime environment details for troubleshooting
    * Integrate a log forwarder or monitoring agent in production to capture request rates errors and latency

8. Testing

    * Unit and integration tests belong under tests at repository root
    * Use pytest configuration present in pytest.ini
    * Run pytest --maxfail=1 --disable-warnings -q from repository root

9. CI and automation

    * GitHub Actions workflow at `.github/workflows/ci-cd.yml` runs dependency install linting and tests on pull requests and push events
    * The workflow builds a Docker image on main branch pushes and pushes to a container registry when credentials are provided

10. Deployment

    * Dockerfile packages the backend including artifacts and the frontend templates
    * The Dockerfile performs sanity checks to ensure required artifacts are present at build time, this prevents broken images
    * `render.yaml` contains sample configuration for Render.com deployments
    * For production deploy behind a secure load balancer and enable HTTPS

11. Security and data handling

    * Keep `.env` and any secrets out of version control
    * Store OpenRouter credentials in secure secret stores provided by your cloud vendor
    * Avoid logging personal identifiers, log only aggregate or anonymized information
    * Validate all input fields to prevent malformed or malicious payloads

12. Extension points for developers

    *  Add new features to the engineering module, then include them in top_features.json and retrain
    * Swap the model file with any scikit learn compatible classifier that implements predict_proba
    * Replace the OpenRouter call with another LLM provider by implementing a small adapter in explanation_service.py
    * Add batch prediction routes by reusing the prediction pipeline and validating payload sizes

## Testing and CI

* Run tests locally with pytest
* CI runs linting and tests automatically on pull requests
* Ensure tests cover feature engineering edge cases missing columns and encoding fallbacks

## Deployment notes

* Build and run with Docker as the simplest production ready option
* Ensure artifacts are included in the build context or accessible from storage at runtime
* Configure environment variables in your host or deployment platform for the OpenRouter key and any other secrets

## Operational checklist for production

* Confirm artifact integrity at startup
* Verify OpenRouter connectivity if you rely on external summaries
* Enable centralized logging
* Configure health checks on /health endpoint
* Set appropriate resource limits for model memory use and request throughput

## 📜 License
This project is distributed under the MIT License.

## 🤝 Contributors

* Lead Developer: ’Wale Ogundeji
* Contributors: AHFID AI Team