from typing import Literal

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=120)
    sex: Literal[0, 1]
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: Literal[0, 1]
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str

