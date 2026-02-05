"""
FastAPI Model Serving Application
=================================

This module provides a REST API for serving ML model predictions.
It handles model loading, request processing, and Prometheus metrics.

Features:
    - Automatic model loading from MLflow or local storage
    - RESTful prediction endpoint with validation
    - Health checks for Kubernetes probes
    - Prometheus metrics for monitoring
    - OpenAPI documentation (Swagger UI)

Endpoints:
    GET  /health      - Health check for load balancers/K8s
    GET  /model/info  - Information about loaded model
    POST /predict     - Make predictions
    GET  /metrics     - Prometheus metrics

Running the Server:
    # Development mode (with auto-reload)
    uvicorn src.serving.main:app --reload --port 8000

    # Production mode (via Makefile)
    make serve

    # Docker
    docker run -p 8000:8000 ml-pipeline:latest

Configuration:
    The server reads settings from config.yaml:
    - serving.port: Server port (default: 8000)
    - serving.workers: Number of uvicorn workers
    - model.name: Model name for MLflow registry
    - model.module: Python module containing ModelWrapper

Model Loading Priority:
    1. MLflow Model Registry (Production stage)
    2. MLflow Model Registry (latest version)
    3. Local models/ directory
    4. No model (predictions will return 503)

See Also:
    - src/serving/schemas.py: Request/response schemas
    - src/model.py: ModelWrapper interface
    - src/config.py: Configuration loading
"""

import importlib
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.serving.schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================
# These metrics are exposed at /metrics for Prometheus scraping

PREDICTIONS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
    ["status"],  # Labels: 'success' or 'error'
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to process prediction requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],  # Latency buckets in seconds
)


# =============================================================================
# GLOBAL STATE
# =============================================================================
# These variables hold the loaded model and configuration.
# They are initialized during application startup (lifespan).

model = None  # The loaded ModelWrapper instance
model_version = "unknown"  # Current model version string
config = None  # Configuration instance


# =============================================================================
# MODEL LOADING UTILITIES
# =============================================================================


def load_model_class(module_path: str, class_name: str):
    """
    Dynamically import a model class from a module path.

    This function enables loading different model implementations
    without hardcoding imports, based on config.yaml settings.

    Args:
        module_path: Dot-separated Python module path
                    (e.g., 'examples.iris.model')
        class_name: Name of the class to import
                   (e.g., 'ModelWrapper')

    Returns:
        The imported class object

    Raises:
        ImportError: If module cannot be found or class doesn't exist

    Example:
        ModelClass = load_model_class('examples.iris.model', 'ModelWrapper')
        model = ModelClass(config)
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load {class_name} from {module_path}: {e}")


def load_model():
    """
    Load the model from MLflow registry or local storage.

    This function implements the model loading priority:
    1. Try MLflow Model Registry (Production stage)
    2. Try MLflow Model Registry (latest version)
    3. Fall back to local models/ directory
    4. Set model_version to "none" if nothing found

    The function updates global variables:
    - model: The loaded ModelWrapper instance
    - model_version: String describing the loaded version
    - config: Configuration instance

    Side Effects:
        Modifies global model, model_version, and config variables.
        Prints loading status to stdout.

    Note:
        This function is called automatically during app startup.
        If loading fails, the API will return 503 for predictions.
    """
    global model, model_version, config

    # Load configuration
    config = get_config()

    # Get model settings from config
    model_module = config.get("model", "module")
    model_class = config.get("model", "class")
    model_name = config.get("model", "name", default="model")

    # Load the model class
    ModelClass = load_model_class(model_module, model_class)
    model = ModelClass(config._config)

    # Check if MLflow is enabled
    mlflow_enabled = config.get("training", "mlflow_enabled", default=True)

    if mlflow_enabled:
        # Try loading from MLflow Model Registry
        try:
            import mlflow

            mlflow.set_tracking_uri(config.mlflow_tracking_uri)

            # Attempt 1: Load Production stage model
            model_uri = f"models:/{model_name}/Production"
            try:
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                model.model = loaded_model
                model_version = "Production"
                print(f"✅ Loaded model '{model_name}' from MLflow (Production)")
                return
            except Exception:
                # Production model not found, try latest
                pass

            # Attempt 2: Load latest version
            model_uri = f"models:/{model_name}/latest"
            try:
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                model.model = loaded_model
                model_version = "latest"
                print(f"✅ Loaded model '{model_name}' from MLflow (latest)")
                return
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ Could not load from MLflow: {e}")

    # Attempt 3: Load from local models/ directory
    local_path = f"models/{model_name}"
    if Path(local_path).exists():
        model.load(local_path)
        model_version = "local"
        print(f"✅ Loaded model from {local_path}")
    else:
        # No model found anywhere
        print("⚠️ Warning: No trained model found. Train a model first.")
        model_version = "none"


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    This async context manager is called by FastAPI during:
    - Startup: Code before 'yield' runs (load model)
    - Shutdown: Code after 'yield' runs (cleanup)

    The lifespan pattern replaces the deprecated @app.on_event decorators
    and provides cleaner resource management.

    Args:
        app: The FastAPI application instance

    Yields:
        Control to the application (runs while app is active)
    """
    # Startup: Load the model
    load_model()

    yield  # Application runs here

    # Shutdown: Cleanup (if needed)
    pass


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="ML Pipeline API",
    description="REST API for serving ML model predictions",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint for monitoring and orchestration.

    This endpoint is designed for:
    - Kubernetes liveness and readiness probes
    - Load balancer health checks
    - Monitoring system availability checks

    Returns:
        HealthResponse with current service status

    Status Codes:
        200: Service is healthy and accepting requests
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and model_version != "none",
        version=model_version,
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """
    Get information about the currently loaded model.

    Returns metadata about the model including:
    - Name (from configuration)
    - Version (from MLflow or local)
    - Framework (sklearn, pytorch, etc.)
    - Loading status

    Returns:
        ModelInfo with model metadata

    Raises:
        HTTPException 503: If configuration is not loaded
    """
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")

    return ModelInfo(
        name=config.get("model", "name", default="unknown"),
        version=model_version,
        framework=config.get("model", "framework", default="unknown"),
        status="loaded" if model_version != "none" else "not_loaded",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make a prediction using the loaded model.

    This endpoint accepts input features and returns the model's
    prediction along with confidence scores and timing information.

    The request is validated against PredictRequest schema.
    Features are passed to the model's predict() method.

    Args:
        request: PredictRequest containing features and optional version

    Returns:
        PredictResponse with prediction, probability, and metadata

    Raises:
        HTTPException 503: If no model is loaded
        HTTPException 500: If prediction fails

    Example:
        POST /predict
        {"features": [5.1, 3.5, 1.4, 0.2]}

        Response:
        {
            "prediction": 0,
            "probability": 0.97,
            "model_version": "Production",
            "latency_ms": 5.2
        }
    """
    # Check if model is loaded
    if model is None or model_version == "none":
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train a model first."
        )

    # Record start time for latency measurement
    start_time = time.time()

    try:
        # Call model's predict method
        result = model.predict(request.features)

        # Calculate latency in milliseconds
        latency = (time.time() - start_time) * 1000

        # Update Prometheus metrics
        PREDICTIONS.labels(status="success").inc()
        LATENCY.observe(latency / 1000)  # Histogram expects seconds

        # Return response
        return PredictResponse(
            prediction=result.get("prediction"),
            probability=result.get("probability"),
            model_version=model_version,
            latency_ms=round(latency, 2),
        )

    except Exception as e:
        # Record error metric
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics for monitoring.

    This endpoint returns metrics in Prometheus text format,
    ready to be scraped by a Prometheus server.

    Metrics exposed:
    - prediction_requests_total: Counter of predictions (success/error)
    - prediction_latency_seconds: Histogram of inference times

    Returns:
        Plain text response with Prometheus metrics

    Prometheus Configuration:
        Add to prometheus.yml:

        scrape_configs:
          - job_name: 'ml-pipeline'
            static_configs:
              - targets: ['localhost:8000']
            metrics_path: /metrics
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run the server directly for development.

    Usage:
        python -m src.serving.main

    For production, use:
        uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import uvicorn

    config = get_config()
    port = config.get("serving", "port", default=8000)
    workers = config.get("serving", "workers", default=4)

    uvicorn.run(
        "src.serving.main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=True,  # Enable auto-reload for development
    )
