# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY ./backend/requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy backend application code and artifacts
COPY ./backend/app /app/app
COPY ./backend/artifacts /app/artifacts

# Copy frontend templates and static files
COPY ./frontend/templates /app/templates
COPY ./frontend/static /app/static

# --- Validate artifacts exist (fail fast if missing) ---
RUN test -f /app/artifacts/gradient_boosting_gbv_model.pkl \
 && test -f /app/artifacts/scalers.pkl \
 && test -f /app/artifacts/label_encoders.pkl \
 && test -f /app/artifacts/analysis/top_features.json \
 && test -f /app/artifacts/analysis/optimal_threshold.json \
 || (echo "❌ Required model artifacts are missing! Please train the model first." && exit 1)

# Make port 8000 available
EXPOSE 8000

# Define environment variable for Python path
ENV PYTHONPATH=/app

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]