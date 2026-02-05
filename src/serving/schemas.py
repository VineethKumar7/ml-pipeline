"""
Pydantic schemas for API request/response validation.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    features: List[Any] = Field(
        ..., description="Input features for the model", example=[5.1, 3.5, 1.4, 0.2]
    )
    model_version: Optional[str] = Field(
        default="latest", description="Model version to use (default: latest)"
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: Any = Field(..., description="Model prediction")
    probability: Optional[float] = Field(
        default=None, description="Prediction probability/confidence"
    )
    model_version: str = Field(..., description="Model version used for prediction")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class ModelInfo(BaseModel):
    """Response schema for model info endpoint."""

    name: str
    version: str
    framework: str
    status: str


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""

    status: str = "healthy"
    model_loaded: bool
    version: str
