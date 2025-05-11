"""
Prediction API for Iris Classification
This module provides a FastAPI service to serve predictions from the trained model.
"""

import os
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load the model
MODEL_PATH = os.environ.get("MODEL_PATH", "models/iris_classifier.joblib")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="API for Iris flower classification using machine learning",
    version="1.0.0",
)


# Input data model with validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }


# Output data model
class PredictionResult(BaseModel):
    species: str
    species_id: int
    probability: float


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Iris Classification API",
        "endpoints": {
            "POST /predict": "Make predictions based on iris features",
            "GET /health": "Check API health and model status",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify the API and model status."""
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded correctly. Please check server logs.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResult)
def predict(features: IrisFeatures):
    """Make a prediction using the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not available. Please try again later."
        )

    try:
        # Create feature array for prediction
        feature_array = np.array(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ]
        )

        # Get prediction and probabilities
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        max_prob = max(probabilities)

        # Map prediction to species
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

        species = species_map.get(prediction, "unknown")

        # Log the prediction
        logger.info(
            f"Prediction: {species} (ID: {prediction}) with probability {max_prob:.4f}"
        )

        return PredictionResult(
            species=species, species_id=int(prediction), probability=float(max_prob)
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}"
        )


if __name__ == "__main__":
    # For development only - use uvicorn in production
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
