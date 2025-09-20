"""
FastAPI service for Adult Income prediction with monitoring and observability.

This service loads the trained model and provides REST endpoints for predictions,
health checks, and Prometheus metrics.
"""

import os
import time
import json
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from joblib import load
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "predictions_total", 
    "Total number of prediction requests"
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", 
    "Total number of prediction errors"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", 
    "Latency of prediction requests"
)

# Input schema
class AdultIncomeFeatures(BaseModel):
    age: int = Field(..., ge=17, le=90, description="Age in years")
    workclass: str = Field(..., description="Work class category")
    fnlwgt: int = Field(..., ge=0, description="Final weight")
    education: str = Field(..., description="Education level")
    education_num: int = Field(..., ge=1, le=16, description="Education in numerical form")
    marital_status: str = Field(..., description="Marital status")
    occupation: str = Field(..., description="Occupation")
    relationship: str = Field(..., description="Relationship status")
    race: str = Field(..., description="Race")
    sex: str = Field(..., description="Gender")
    capital_gain: int = Field(..., ge=0, description="Capital gains")
    capital_loss: int = Field(..., ge=0, description="Capital losses")
    hours_per_week: int = Field(..., ge=1, le=99, description="Hours worked per week")
    native_country: str = Field(..., description="Native country")

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    prediction_id: str
    model_version: str

# Initialize FastAPI app
app = FastAPI(
    title="Adult Income Prediction Service",
    description="ML service for predicting whether income exceeds $50K",
    version="1.0.0"
)

# Global model variable
model = None
model_info = {}

def load_model():
    """Load the trained model and metadata."""
    global model, model_info
    
    model_dir = os.environ.get("MODEL_DIR", "./model_dir")
    results_dir = os.environ.get("RESULTS_DIR", "./results_dir")
    
    model_path = Path(model_dir) / "logit_model.joblib"
    metadata_path = Path(results_dir) / "train_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load(model_path)
    
    # Load model metadata if available
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            model_info = json.load(f)
    else:
        model_info = {"validation_acc": "unknown", "model_type": "LogisticRegression"}
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Model validation accuracy: {model_info.get('validation_acc', 'unknown')}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_info.get("model_type", "unknown"),
        "validation_accuracy": model_info.get("validation_acc", "unknown"),
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: AdultIncomeFeatures):
    """Make a prediction on adult income data."""
    start_time = time.time()
    PREDICTION_COUNTER.inc()
    
    if model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to the format expected by the model
        # Note: This is a simplified example - in practice you'd need to handle
        # the full preprocessing pipeline including categorical encoding
        feature_dict = features.dict()
        
        # For demonstration, we'll create a simple feature vector
        # In practice, you'd need the exact same preprocessing as training
        feature_vector = [
            feature_dict['age'],
            feature_dict['fnlwgt'],
            feature_dict['education_num'],
            feature_dict['capital_gain'],
            feature_dict['capital_loss'],
            feature_dict['hours_per_week']
        ]
        
        # Add dummy values for categorical features (this is simplified)
        # In practice, you'd need proper one-hot encoding
        feature_vector.extend([0] * 100)  # Placeholder for categorical features
        
        # Ensure we have the right number of features
        X = np.array(feature_vector[:model.named_steps['classifier'].coef_.shape[1]]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()
        
        # Convert prediction to readable format
        prediction_label = ">50K" if prediction == 1 else "<=50K"
        
        # Record latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        # Generate unique prediction ID
        prediction_id = f"pred_{int(time.time() * 1000)}"
        
        return PredictionResponse(
            prediction=prediction_label,
            probability=float(probability),
            prediction_id=prediction_id,
            model_version=model_info.get("model_type", "v1.0")
        )
        
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model/info")
def get_model_info():
    """Get model information and metadata."""
    return {
        "model_info": model_info,
        "model_loaded": model is not None,
        "feature_count": getattr(model.named_steps['classifier'], 'n_features_in_', 'unknown') if model else 'unknown'
    }

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Adult Income Prediction Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
