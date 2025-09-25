# backend/app/models/pydantic_models.py: Pydantic models for request and response validation
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionInput(BaseModel):
    # Direct features
    survivor_age: int = Field(..., example=25, description="Age")
    survivor_sex: str = Field(..., example="Female", description="Sex")
    marital_status: str = Field(..., example="Married", description="Marital status")
    educational_status: str = Field(..., example="No formal education", description="Educational level")
    employment_status_main: str = Field(..., example="Unemployed", description="Main Provider Employment Status")
    employment_status_victim_main: str = Field(..., example="Unemployed", description="Employment status")
    who_survivor_victim_stay_with: str = Field(..., example="Spouse/Partner", description="Who do you live with?")

    # Binary vulnerability flags (0 or 1) - Only those used in TOP_FEATURES engineering
    PLWD: int = Field(0, example=0, description=" Do you have a disability? (1=yes, 0=no)")
    PLHIV: int = Field(0, example=0, description="Do you live with HIV? (1=yes, 0=no)")
    IDP: int = Field(0, example=0, description="Are you internally displaced? (1=yes, 0=no)")
    drug_user: int = Field(0, example=0, description="Do you use drugs? (1=yes, 0=no)")
    widow: int = Field(0, example=0, description="Are you a widow? (1=yes, 0=no)")
    out_of_school_child: int = Field(0, example=0, description="Are you an out of school child? (1=yes, 0=no)")
    minor: int = Field(0, example=0, description="Are you a minor (under 18)? (1=yes, 0=no)")
    household_help: int = Field(0, example=0, description="Are you a household help/domestic worker? (1=yes, 0=no)")
    child_apprentice: int = Field(0, example=0, description="Are you a child apprentice? (1=yes, 0=no)")
    orphans: int = Field(0, example=0, description="Are you an orphan? (1=yes, 0=no)")
    female_sex_worker: int = Field(0, example=0, description="Do you engage in sex work? (1=yes, 0=no)")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "survivor_age": 25,
                "survivor_sex": "Female",
                "marital_status": "Married",
                "educational_status": "Secondary",
                "employment_status_main": "Self employed",
                "employment_status_victim_main": "Unemployed",
                "who_survivor_victim_stay_with": "Spouse/Partner",
                "PLWD": 0,
                "PLHIV": 0,
                "IDP": 0,
                "drug_user": 0,
                "widow": 0,
                "out_of_school_child": 0,
                "minor": 0,
                "household_help": 0,
                "child_apprentice": 0,
                "orphans": 0,
                "female_sex_worker": 0
            }
        }

class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Name of the feature")
    impact: float = Field(..., description="Impact score of the feature on the prediction")
    description: Optional[str] = Field(None, description="Human-readable description of the feature")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Risk level prediction (High Risk or Low Risk)")
    risk_probability: float = Field(..., description="Probability of high risk (0-1)", ge=0, le=1)
    confidence: float = Field(..., description="Confidence level of the prediction (0-1)", ge=0, le=1)
    key_risk_factors: List[FeatureContribution] = Field(..., description="Top factors contributing to increased risk")
    key_protective_factors: List[FeatureContribution] = Field(..., description="Top factors providing protection/reducing risk")
    generative_summary: str = Field(..., description="Comprehensive AI-generated summary with assessment and recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "Low Risk",
                "risk_probability": 0.15,
                "confidence": 0.85,
                "key_risk_factors": [
                    {
                        "feature": "economic_dependency_score",
                        "impact": 0.12,
                        "description": "Economic dependency increases vulnerability"
                    },
                    {
                        "feature": "social_isolation_score",
                        "impact": 0.08,
                        "description": "Social isolation increases risk factors"
                    }
                ],
                "key_protective_factors": [
                    {
                        "feature": "community_connection_score",
                        "impact": -0.25,
                        "description": "Strong community connections provide protection"
                    },
                    {
                        "feature": "housing_security_score",
                        "impact": -0.18,
                        "description": "Stable housing provides security and reduces risk"
                    }
                ],
                "generative_summary": "ASSESSMENT SUMMARY: The risk assessment indicates a low-risk profile with several protective factors present. The individual demonstrates strong community connections and stable housing arrangements, which serve as significant buffers against potential vulnerabilities. While some economic dependency factors are present, they are offset by supportive living arrangements and community integration. The overall risk profile suggests resilience and adequate support systems are in place.\n\nRECOMMENDATIONS: 1) Continue to strengthen existing community connections through ongoing participation in social activities and support networks. 2) Monitor economic stability and provide resources for financial literacy or income generation if circumstances change. 3) Maintain regular check-ins to ensure continued housing stability and address any emerging concerns. 4) Document current protective factors to inform future assessments and maintain continuity of care."
            }
        }

class BatchPredictionInput(BaseModel):
    cases: List[PredictionInput] = Field(..., description="List of cases to predict")
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    summary: dict = Field(..., description="Summary statistics of the batch predictions")