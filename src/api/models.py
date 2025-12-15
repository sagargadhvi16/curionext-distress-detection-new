"""Pydantic models for API."""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class PredictionResponse(BaseModel):
    """Response model for distress prediction."""

    distress_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of distress (0-1)"
    )
    prediction: str = Field(
        ...,
        description="Binary prediction: 'distress' or 'no_distress'"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence (0-1)"
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    explanation: Optional[Dict] = Field(
        None,
        description="SHAP explanation (if requested)"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
