# AI-Powered Gender-Based Violence (GBV) Predictive Tool

A proactive, data-driven tool designed to predict individuals at risk of Gender-Based Violence (GBV) using historical case data and demographic patterns. Unlike reactive models that rely on confirmed incidents, this system identifies vulnerability early to support timely, targeted interventions and prevention strategies.

This repository contains the source code for a full-stack web application designed to predict the risk of Gender-Based Violence. It uses a Gradient Boosting machine learning model trained on socio-economic and demographic data.

## Features

- **High-Accuracy Predictions**: Utilizes a model with 99.6% accuracy and 99.5% sensitivity.
- **Explainable AI (XAI)**: Integrates SHAP to explain the factors driving each prediction.
- **Generative AI Summaries**: Provides concise, human-readable summaries of risk profiles for non-technical users.
- **Modular Architecture**: Built with a FastAPI backend and a clean HTML/CSS/JS frontend.
- **Containerized**: Dockerized for easy and consistent deployment.
- **CI/CD Ready**: Includes a GitHub Actions workflow for automated testing and deployment.

## Project Structure

- `backend/`: Contains the FastAPI application, services, and Dockerfile.
- `src/`: Contains the Python scripts for data preprocessing and model training.
- `frontend/`: Contains the HTML, CSS, and JavaScript for the user interface.
- `data/`: Contains the training data.
- `tests/`: Contains unit and integration tests.
- `.github/`: Contains the CI/CD workflow definition.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- An environment variable manager (e.g., `python-dotenv`)

### 1. Training the Model (Run once)

First, you need to run the training pipeline to generate the model artifacts.

```bash
# 1. Navigate to the src directory
cd src

# 2. Install training dependencies
pip install -r ../backend/requirements.txt

# 3. Run the training pipeline
python training/train.py