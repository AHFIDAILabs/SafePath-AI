# backend/app/api/predict.py: API endpoint for making predictions
from fastapi import APIRouter, HTTPException
from ..models.pydantic_models import PredictionInput, PredictionResponse
from ..services import prediction_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_risk(input_data: PredictionInput):
    """
    Predicts the risk of Gender-Based Violence based on input features.
    """
    try:
        input_dict = input_data.dict()
        result = prediction_service.make_prediction(input_dict)
        return PredictionResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")