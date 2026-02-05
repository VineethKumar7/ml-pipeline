"""
FastAPI Model Serving Application

Loads model from config and serves predictions via REST API.

Usage:
    uvicorn src.serving.main:app --reload --port 8000
"""

import importlib
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.serving.schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

# Metrics
PREDICTIONS = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['status']
)
LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Global model instance
model = None
model_version = "unknown"
config = None


def load_model_class(module_path: str, class_name: str):
    """Dynamically load model class from module path."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load {class_name} from {module_path}: {e}"
        )


def load_model():
    """Load model from MLflow or local path."""
    global model, model_version, config
    
    config = get_config()
    
    model_module = config.get('model', 'module')
    model_class = config.get('model', 'class')
    model_name = config.get('model', 'name', default='model')
    
    ModelClass = load_model_class(model_module, model_class)
    model = ModelClass(config._config)
    
    # Try to load from MLflow first
    mlflow_enabled = config.get('training', 'mlflow_enabled', default=True)
    
    if mlflow_enabled:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            
            # Load latest production model
            model_uri = f"models:/{model_name}/Production"
            
            try:
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                model.model = loaded_model
                model_version = "Production"
                print(f"Loaded model '{model_name}' from MLflow (Production)")
                return
            except Exception:
                # Try latest version
                model_uri = f"models:/{model_name}/latest"
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                model.model = loaded_model
                model_version = "latest"
                print(f"Loaded model '{model_name}' from MLflow (latest)")
                return
        except Exception as e:
            print(f"Could not load from MLflow: {e}")
    
    # Fallback to local model
    local_path = f"models/{model_name}"
    if Path(local_path).exists():
        model.load(local_path)
        model_version = "local"
        print(f"Loaded model from {local_path}")
    else:
        print(f"Warning: No trained model found. Train a model first.")
        model_version = "none"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="ML Pipeline API",
    description="Model serving API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and model_version != "none",
        version=model_version,
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    if config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")
    
    return ModelInfo(
        name=config.get('model', 'name', default='unknown'),
        version=model_version,
        framework=config.get('model', 'framework', default='unknown'),
        status="loaded" if model_version != "none" else "not_loaded",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make a prediction."""
    if model is None or model_version == "none":
        PREDICTIONS.labels(status='error').inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first."
        )
    
    start_time = time.time()
    
    try:
        result = model.predict(request.features)
        latency = (time.time() - start_time) * 1000
        
        PREDICTIONS.labels(status='success').inc()
        LATENCY.observe(latency / 1000)
        
        return PredictResponse(
            prediction=result.get('prediction'),
            probability=result.get('probability'),
            model_version=model_version,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        PREDICTIONS.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


# For running with: python -m src.serving.main
if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    port = config.get('serving', 'port', default=8000)
    workers = config.get('serving', 'workers', default=4)
    
    uvicorn.run(
        "src.serving.main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=True,
    )
