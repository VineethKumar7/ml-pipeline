#!/usr/bin/env python3
"""
Generic Training Script

Reads config.yaml, loads the specified model class, trains it,
and logs everything to MLflow.

Usage:
    python -m src.training.train
    python -m src.training.train --params '{"n_estimators": 200}'
"""

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config


def load_model_class(module_path: str, class_name: str):
    """Dynamically load model class from module path."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load {class_name} from {module_path}: {e}"
        )


def setup_mlflow(config):
    """Configure MLflow tracking."""
    import mlflow
    
    tracking_uri = config.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config.get('training', 'experiment', default='default')
    mlflow.set_experiment(experiment_name)
    
    return mlflow


def train(data_loader=None, params_override: dict = None):
    """
    Main training function.
    
    Args:
        data_loader: Optional function that returns (X_train, y_train, X_test, y_test)
        params_override: Optional parameters to override config
    """
    config = get_config()
    
    # Load model class
    model_module = config.get('model', 'module')
    model_class = config.get('model', 'class')
    ModelClass = load_model_class(model_module, model_class)
    
    # Get training params
    params = config.get('training', 'params', default={})
    if params_override:
        params.update(params_override)
    
    # Initialize model
    model = ModelClass(config._config)
    
    # Load data
    if data_loader is None:
        # Default: try to import from model module
        try:
            data_module = importlib.import_module(model_module.rsplit('.', 1)[0])
            data_loader = getattr(data_module, 'load_data', None)
        except (ImportError, AttributeError):
            pass
    
    if data_loader is None:
        raise ValueError(
            "No data loader provided. Either pass data_loader argument "
            "or implement load_data() in your model module."
        )
    
    X_train, y_train, X_test, y_test = data_loader()
    
    # Setup MLflow if enabled
    mlflow_enabled = config.get('training', 'mlflow_enabled', default=True)
    
    if mlflow_enabled:
        mlflow = setup_mlflow(config)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param('model_class', model_class)
            mlflow.log_param('model_module', model_module)
            
            # Train
            print(f"Training {model_class}...")
            metrics = model.train(X_train, y_train, params)
            
            # Evaluate on test set if model has evaluate method
            if hasattr(model, 'evaluate'):
                test_metrics = model.evaluate(X_test, y_test)
                metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
            
            # Log metrics
            mlflow.log_metrics(metrics)
            print(f"Metrics: {metrics}")
            
            # Log model
            model_name = config.get('model', 'name', default='model')
            
            # Save model artifacts
            artifact_path = f"models/{model_name}"
            model.save(artifact_path)
            
            # Log to MLflow
            mlflow.log_artifacts(artifact_path, artifact_path="model")
            
            # Register model
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            
            try:
                mlflow.register_model(model_uri, model_name)
                print(f"Model registered as '{model_name}'")
            except Exception as e:
                print(f"Could not register model: {e}")
            
            print(f"Training complete. Run ID: {run_id}")
            return metrics, run_id
    else:
        # Train without MLflow
        print(f"Training {model_class} (MLflow disabled)...")
        metrics = model.train(X_train, y_train, params)
        print(f"Metrics: {metrics}")
        
        # Save model locally
        model_name = config.get('model', 'name', default='model')
        artifact_path = f"models/{model_name}"
        Path(artifact_path).mkdir(parents=True, exist_ok=True)
        model.save(artifact_path)
        print(f"Model saved to {artifact_path}")
        
        return metrics, None


def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument(
        '--params',
        type=str,
        help='JSON string of parameters to override config'
    )
    args = parser.parse_args()
    
    params_override = None
    if args.params:
        params_override = json.loads(args.params)
    
    train(params_override=params_override)


if __name__ == '__main__':
    main()
