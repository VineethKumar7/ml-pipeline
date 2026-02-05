"""
API Request/Response Schemas
============================

This module defines Pydantic models for validating and documenting
the FastAPI endpoints. These schemas provide:

- Request validation with automatic error messages
- Response serialization with type checking
- OpenAPI documentation generation
- IDE autocompletion support

Each schema corresponds to an API endpoint:
- PredictRequest/PredictResponse: /predict endpoint
- ModelInfo: /model/info endpoint
- HealthResponse: /health endpoint

Pydantic V2 Notes:
    This module uses Pydantic V2 syntax. Key differences from V1:
    - Use json_schema_extra instead of schema_extra
    - Use model_config instead of Config class
    - Field examples use json_schema_extra={"example": ...}

See Also:
    - src/serving/main.py: FastAPI application using these schemas
    - https://docs.pydantic.dev/latest/: Pydantic documentation
    - https://fastapi.tiangolo.com/: FastAPI documentation
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Request schema for the /predict endpoint.

    This schema validates incoming prediction requests and provides
    documentation for the API. Features are passed as a list that
    will be processed by the model's predict() method.

    Attributes:
        features: List of input features for the model. The exact
                 format depends on your model (e.g., [5.1, 3.5, 1.4, 0.2]
                 for Iris classifier with 4 numeric features).

        model_version: Optional version specifier. Currently supports
                      "latest" (default) to use the most recent model.
                      Future versions may support specific version tags.

    Example Request:
        POST /predict
        {
            "features": [5.1, 3.5, 1.4, 0.2],
            "model_version": "latest"
        }
    """

    features: List[Any] = Field(
        ...,  # Required field (no default)
        description="Input features for the model as a list of values",
        json_schema_extra={"example": [5.1, 3.5, 1.4, 0.2]},
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use for prediction (default: 'latest')",
    )


class PredictResponse(BaseModel):
    """
    Response schema for the /predict endpoint.

    This schema defines the structure of prediction responses.
    The 'prediction' field contains the model output, while
    additional fields provide context and debugging information.

    Attributes:
        prediction: The model's prediction. Type varies by model:
                   - Integer class label for classification
                   - Float value for regression
                   - String for text generation

        probability: Optional confidence score (0-1) for the prediction.
                    Only provided if the model supports probability output.

        model_version: Version identifier of the model used. Useful for
                      tracking which model produced which predictions.

        latency_ms: Time taken for inference in milliseconds. Useful for
                   performance monitoring and SLA tracking.

    Example Response:
        {
            "prediction": 0,
            "probability": 0.97,
            "model_version": "v1.2.0",
            "latency_ms": 5.2
        }
    """

    prediction: Any = Field(
        ...,
        description="Model prediction (type depends on model)",
    )
    probability: Optional[float] = Field(
        default=None,
        description="Prediction probability/confidence score (0-1)",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for this prediction",
    )
    latency_ms: float = Field(
        ...,
        description="Inference latency in milliseconds",
    )


class ModelInfo(BaseModel):
    """
    Response schema for the /model/info endpoint.

    Provides metadata about the currently loaded model.
    Useful for debugging, monitoring, and documentation.

    Attributes:
        name: The model's registered name from config.yaml
        version: Current version (e.g., "Production", "v1.2.0", "local")
        framework: ML framework used (sklearn, pytorch, tensorflow)
        status: Model status ("loaded", "not_loaded", "error")

    Example Response:
        {
            "name": "iris-classifier",
            "version": "Production",
            "framework": "sklearn",
            "status": "loaded"
        }
    """

    name: str = Field(description="Model name from configuration")
    version: str = Field(description="Currently loaded model version")
    framework: str = Field(description="ML framework (sklearn/pytorch/tensorflow)")
    status: str = Field(description="Model loading status")


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.

    Provides basic health check information for load balancers,
    Kubernetes probes, and monitoring systems.

    Attributes:
        status: Overall health status. "healthy" indicates the
               service is running and can accept requests.

        model_loaded: Boolean indicating if a model is loaded
                     and ready for predictions.

        version: Current model version or "none" if not loaded.

    Example Response:
        {
            "status": "healthy",
            "model_loaded": true,
            "version": "Production"
        }

    Kubernetes Integration:
        Configure liveness and readiness probes to use this endpoint:

        livenessProbe:
          httpGet:
            path: /health
            port: 8000
    """

    status: str = Field(
        default="healthy",
        description="Service health status",
    )
    model_loaded: bool = Field(
        description="Whether a model is loaded and ready",
    )
    version: str = Field(
        description="Loaded model version or 'none'",
    )
