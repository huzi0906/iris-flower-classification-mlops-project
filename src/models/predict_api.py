"""
FastAPI service for Iris classification model predictions.
"""

import os
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="API for Iris flower classification",
    version="1.0.0",
)

# Load the model
MODEL_PATH = os.environ.get("MODEL_PATH", "models/iris_model.pkl")

# Global variable for model
model = None


def load_model():
    """Load the trained model."""
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail=f"Model file not found at {MODEL_PATH}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error loading model: {str(e)}"
        )


# Define request and response models
class IrisFeatures(BaseModel):
    """Input features for Iris prediction."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    predicted_class: int
    predicted_species: str
    probabilities: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Iris Classification API. Use /predict for predictions."}


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded"
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Make a prediction with the Iris classification model.
    
    Args:
        features: Iris flower measurements
    
    Returns:
        PredictionResponse: Prediction result
    """
    if model is None:
        load_model()
    
    # Convert input to array for prediction
    input_data = np.array([
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
    ])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Map class to species name
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = species_map.get(prediction, "unknown")
    
    # Create response
    response = {
        "predicted_class": int(prediction),
        "predicted_species": predicted_species,
        "probabilities": {
            species_map[i]: float(prob) for i, prob in enumerate(probabilities)
        }
    }
    
    return response


@app.get("/predict", response_model=PredictionResponse)
async def predict_get(
    sepal_length: float = Query(..., description="Sepal length in cm"),
    sepal_width: float = Query(..., description="Sepal width in cm"),
    petal_length: float = Query(..., description="Petal length in cm"),
    petal_width: float = Query(..., description="Petal width in cm")
):
    """
    Make a prediction using GET request.
    
    Args:
        sepal_length: Sepal length in cm
        sepal_width: Sepal width in cm
        petal_length: Petal length in cm
        petal_width: Petal width in cm
    
    Returns:
        PredictionResponse: Prediction result
    """
    features = IrisFeatures(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width
    )
    return await predict(features)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)