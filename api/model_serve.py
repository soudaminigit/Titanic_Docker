from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
import pickle
import numpy as np
from datetime import datetime

app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
    description="REST API for predicting Titanic passenger survival"
)

# -----------------------
# Load artifacts on startup
# -----------------------
@app.on_event("startup")
def load_model():
    global model, feature_columns
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("features.pkl", "rb") as f:
            feature_columns = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}")

# -----------------------
# Request schema
# -----------------------
class PassengerRequest(BaseModel):
    Pclass: int = Field(..., example=3)
    Sex: int = Field(..., example=1, description="0 = female, 1 = male")
    Age: float = Field(..., example=22.0, ge=0)
    SibSp: int = Field(..., example=1, ge=0)
    Parch: int = Field(..., example=0, ge=0)
    Fare: float = Field(..., example=7.25, ge=0)

# -----------------------
# Response schema
# -----------------------
class PredictionResponse(BaseModel):
    survived: int
    model_version: str
    timestamp: str

# -----------------------
# Health check
# -----------------------
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok"}

# -----------------------
# Prediction endpoint
# -----------------------
@app.post(
    "/v1/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def predict_survival(passenger: PassengerRequest):

    try:
        input_data = np.array([[
            passenger.Pclass,
            passenger.Sex,
            passenger.Age,
            passenger.SibSp,
            passenger.Parch,
            passenger.Fare
        ]])

        prediction = int(model.predict(input_data)[0])

        return {
            "survived": prediction,
            "model_version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
