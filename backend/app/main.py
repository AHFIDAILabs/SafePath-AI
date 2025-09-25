# backend/app/main.py:  FastAPI application setup with CORS middleware and route inclusion.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import predict

app = FastAPI(
    title="GBV Predictive Tool API",
    description="An API to predict the risk of Gender-Based Violence.",
    version="1.0.0"
)

# CORS Middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the GBV Predictive Tool API"}

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}