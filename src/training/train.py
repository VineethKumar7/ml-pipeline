#!/usr/bin/env python3
"""
Generic Training Script
=======================

This module provides a framework-agnostic training pipeline that:
1. Loads configuration from config.yaml
2. Dynamically imports the specified model class
3. Loads training data using the model's data loader
4. Trains the model with MLflow experiment tracking
5. Saves and registers the trained model

The script is designed to work with any model that implements the
ModelWrapper interface, making it reusable across different projects.

Usage:
    # Basic training (uses config.yaml settings)
    python -m src.training.train

    # Override training parameters
    python -m src.training.train --params '{"n_estimators": 200}'

    # From Makefile
    make train

MLflow Integration:
    When mlflow_enabled is True in config.yaml:
    - Creates/uses experiment specified in config
    - Logs all training parameters
    - Logs returned metrics
    - Saves model artifacts to S3 (MinIO)
    - Registers model in MLflow Model Registry

    When mlflow_enabled is False:
    - Trains model locally
    - Saves artifacts to models/ directory
    - No experiment tracking

Data Loading:
    The script expects a load_data() function in the model's parent module.
    This function should return: (X_train, y_train, X_test, y_test)

    Example:
        # In examples/iris/data.py
        def load_data():
            X, y = load_iris(return_X_y=True)
            return train_test_split(X, y, test_size=0.2)

See Also:
    - src/model.py: ModelWrapper interface
    - examples/iris/model.py: Example implementation
    - examples/iris/data.py: Example data loader
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config

# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model_class(module_path: str, class_name: str):
    """
    Dynamically import and return a model class.

    Uses Python's importlib to load a class at runtime based on
    configuration. This allows the training script to work with
    any model without hardcoded imports.

    Args:
        module_path: Dot-separated module path (e.g., 'examples.iris.model')
        class_name: Name of the class to import (e.g., 'ModelWrapper')

    Returns:
        The imported class object, ready to be instantiated

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class doesn't exist in the module

    Example:
        ModelClass = load_model_class('examples.iris.model', 'ModelWrapper')
        model = ModelClass(config)
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load {class_name} from {module_path}: {e}")


# =============================================================================
# MLFLOW SETUP
# =============================================================================


def setup_mlflow(config):
    """
    Configure MLflow tracking for the training run.

    Sets up the MLflow tracking URI and experiment name from config.
    The tracking URI points to the MLflow server (local or remote).

    Args:
        config: Config instance with MLflow settings

    Returns:
        The configured mlflow module, ready for logging

    Side Effects:
        - Sets MLflow tracking URI
        - Creates experiment if it doesn't exist
        - Sets the active experiment

    Note:
        Requires MLflow server to be running at the configured URI.
        Start with: docker-compose up -d mlflow
    """
    import mlflow

    # Set tracking server URI
    tracking_uri = config.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Set or create experiment
    experiment_name = config.get("training", "experiment", default="default")
    mlflow.set_experiment(experiment_name)

    return mlflow


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train(data_loader=None, params_override: dict = None):
    """
    Execute the complete training pipeline.

    This function orchestrates the entire training process:
    1. Load configuration
    2. Import model class dynamically
    3. Load training data
    4. Train model with specified parameters
    5. Evaluate on test set (if evaluate method exists)
    6. Log metrics and artifacts to MLflow
    7. Register model in MLflow Model Registry

    Args:
        data_loader: Optional callable that returns training data.
                    Signature: () -> (X_train, y_train, X_test, y_test)
                    If None, attempts to import load_data from model module.

        params_override: Optional dictionary of parameters to override
                        config.yaml settings. Useful for hyperparameter
                        tuning or command-line overrides.

    Returns:
        Tuple of (metrics_dict, run_id):
        - metrics_dict: Dictionary of all logged metrics
        - run_id: MLflow run ID (or None if MLflow disabled)

    Raises:
        ValueError: If no data loader is available
        ImportError: If model class cannot be loaded
        Exception: Various MLflow errors if tracking fails

    Example:
        # Basic training
        metrics, run_id = train()
        print(f"Accuracy: {metrics['accuracy']}")

        # With custom data loader
        def my_loader():
            return X_train, y_train, X_test, y_test

        metrics, run_id = train(data_loader=my_loader)

        # With parameter overrides
        metrics, run_id = train(params_override={'epochs': 50})
    """
    # Load configuration
    config = get_config()

    # -------------------------------------------------------------------------
    # Step 1: Load Model Class
    # -------------------------------------------------------------------------
    model_module = config.get("model", "module")
    model_class = config.get("model", "class")

    print(f"Loading model class: {model_class} from {model_module}")
    ModelClass = load_model_class(model_module, model_class)

    # -------------------------------------------------------------------------
    # Step 2: Get Training Parameters
    # -------------------------------------------------------------------------
    # Start with config params
    params = config.get("training", "params", default={})

    # Apply overrides if provided
    if params_override:
        params.update(params_override)

    print(f"Training parameters: {params}")

    # -------------------------------------------------------------------------
    # Step 3: Initialize Model
    # -------------------------------------------------------------------------
    model = ModelClass(config._config)

    # -------------------------------------------------------------------------
    # Step 4: Load Training Data
    # -------------------------------------------------------------------------
    if data_loader is None:
        # Try to import load_data from model's parent module
        try:
            # Get parent module (e.g., 'examples.iris' from 'examples.iris.model')
            parent_module = model_module.rsplit(".", 1)[0]
            data_module = importlib.import_module(f"{parent_module}.data")
            data_loader = getattr(data_module, "load_data", None)
        except (ImportError, AttributeError):
            pass

    if data_loader is None:
        raise ValueError(
            "No data loader provided. Either pass data_loader argument "
            "or implement load_data() in your model's data module."
        )

    print("Loading training data...")
    X_train, y_train, X_test, y_test = data_loader()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # Step 5: Train with or without MLflow
    # -------------------------------------------------------------------------
    mlflow_enabled = config.get("training", "mlflow_enabled", default=True)

    if mlflow_enabled:
        # Training with MLflow tracking
        return _train_with_mlflow(
            config, model, X_train, y_train, X_test, y_test, params, model_class
        )
    else:
        # Training without MLflow (local only)
        return _train_local(config, model, X_train, y_train, params)


def _train_with_mlflow(
    config, model, X_train, y_train, X_test, y_test, params, model_class
):
    """
    Execute training with full MLflow integration.

    This internal function handles the MLflow-specific training workflow:
    - Start a new MLflow run
    - Log parameters and metrics
    - Save and register the model
    - Handle artifact storage

    Args:
        config: Configuration instance
        model: Initialized ModelWrapper instance
        X_train, y_train: Training data
        X_test, y_test: Test data for evaluation
        params: Training parameters
        model_class: Name of the model class (for logging)

    Returns:
        Tuple of (metrics_dict, run_id)
    """
    mlflow = setup_mlflow(config)
    model_name = config.get("model", "name", default="model")
    model_module = config.get("model", "module")

    with mlflow.start_run():
        # Log training parameters
        mlflow.log_params(params)
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("model_module", model_module)

        # =====================================================================
        # Train the model
        # =====================================================================
        print(f"Training {model_class}...")
        metrics = model.train(X_train, y_train, params)

        # =====================================================================
        # Evaluate on test set (if model supports it)
        # =====================================================================
        if hasattr(model, "evaluate"):
            print("Evaluating on test set...")
            test_metrics = model.evaluate(X_test, y_test)
            # Prefix test metrics to distinguish from training metrics
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        # =====================================================================
        # Log metrics to MLflow
        # =====================================================================
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        # =====================================================================
        # Save model artifacts
        # =====================================================================
        artifact_path = f"models/{model_name}"
        Path(artifact_path).mkdir(parents=True, exist_ok=True)
        model.save(artifact_path)

        # Log artifacts to MLflow (uploaded to MinIO)
        mlflow.log_artifacts(artifact_path, artifact_path="model")

        # =====================================================================
        # Register model in MLflow Model Registry
        # =====================================================================
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        try:
            mlflow.register_model(model_uri, model_name)
            print(f"✅ Model registered as '{model_name}'")
        except Exception as e:
            print(f"⚠️ Could not register model: {e}")

        print(f"✅ Training complete. Run ID: {run_id}")
        return metrics, run_id


def _train_local(config, model, X_train, y_train, params):
    """
    Execute training without MLflow (local mode).

    This internal function handles training when MLflow is disabled:
    - Train the model
    - Save artifacts locally
    - No experiment tracking

    Args:
        config: Configuration instance
        model: Initialized ModelWrapper instance
        X_train, y_train: Training data
        params: Training parameters

    Returns:
        Tuple of (metrics_dict, None)
    """
    model_class = config.get("model", "class")
    model_name = config.get("model", "name", default="model")

    # Train
    print(f"Training {model_class} (MLflow disabled)...")
    metrics = model.train(X_train, y_train, params)
    print(f"Metrics: {metrics}")

    # Save model locally
    artifact_path = f"models/{model_name}"
    Path(artifact_path).mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)
    print(f"✅ Model saved to {artifact_path}")

    return metrics, None


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """
    Command-line interface for the training script.

    Parses command-line arguments and runs training.

    Usage:
        python -m src.training.train
        python -m src.training.train --params '{"n_estimators": 200}'
    """
    parser = argparse.ArgumentParser(
        description="Train ML model using configuration from config.yaml"
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of parameters to override config (e.g., '{\"epochs\": 50}')",
    )
    args = parser.parse_args()

    # Parse parameter overrides if provided
    params_override = None
    if args.params:
        params_override = json.loads(args.params)

    # Run training
    train(params_override=params_override)


if __name__ == "__main__":
    main()
