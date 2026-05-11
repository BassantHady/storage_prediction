"""
api.py
======
FastAPI-based REST API for storage type prediction.
Provides endpoints for single and batch predictions.

Author: NLP Engineering Team
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import StoragePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Storage Type Prediction API",
    description="NLP API for predicting whether items should be stored in freezer, fridge, or at room temperature.",
    version="1.0.0"
)

# Global predictor instances (loaded once at startup)
predictors = {}


@app.on_event("startup")
async def load_models():
    """Load all models on startup."""
    logger.info("Loading models...")
    model_names = ["logistic", "svm", "rf", "distilbert"]
    
    for model_name in model_names:
        try:
            predictors[model_name] = StoragePredictor(model_name=model_name)
            logger.info(f"Loaded {model_name} model")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
    
    logger.info(f"Loaded {len(predictors)} models")


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., description="Input sentence asking about storage", example="Should I store milk in the fridge?")
    model: str = Field(default="distilbert", description="Model to use", example="distilbert")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., description="List of input sentences", example=["Should I store milk in the fridge?", "Where to keep frozen pizza?"])
    model: str = Field(default="distilbert", description="Model to use")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    text: str
    predicted_storage: str
    confidence: Optional[float] = None
    all_probs: Optional[Dict[str, float]] = None
    model_used: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    total_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_available: List[str]


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Storage Type Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_available=list(predictors.keys())
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict storage type for a single sentence.
    
    Returns whether the item should be stored in freezer, fridge, or at room temperature.
    """
    if request.model not in predictors:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not available. Choose from: {list(predictors.keys())}")
    
    try:
        predictor = predictors[request.model]
        result = predictor.predict(request.text)
        
        return PredictionResponse(
            text=result["text"],
            predicted_storage=result["predicted_storage"],
            confidence=result.get("confidence"),
            all_probs=result.get("all_probs"),
            model_used=request.model
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict storage types for multiple sentences in batch.
    """
    if request.model not in predictors:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not available. Choose from: {list(predictors.keys())}")
    
    try:
        predictor = predictors[request.model]
        results = predictor.predict_batch(request.texts)
        
        predictions = [
            PredictionResponse(
                text=r["text"],
                predicted_storage=r["predicted_storage"],
                confidence=r.get("confidence"),
                all_probs=r.get("all_probs"),
                model_used=request.model
            )
            for r in results
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[str])
async def list_models():
    """List all available models."""
    return list(predictors.keys())


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_api()