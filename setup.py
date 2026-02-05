#!/usr/bin/env python3
"""
ML Pipeline Bootstrap - Interactive Setup Wizard

Run this script to configure your ML pipeline project.
Press Enter to accept defaults or type your own values.
"""

import os
import yaml
from pathlib import Path

DEFAULTS = {
    'project_name': 'ml-pipeline',
    'version': '1.0.0',
    'description': 'ML model deployment pipeline',
    'model_name': 'my-model',
    'framework': 'sklearn',
    'model_module': 'src.model',
    'model_class': 'ModelWrapper',
    'experiment': 'default',
    'api_port': 8000,
    'workers': 4,
    'mlflow_enabled': True,
    'mlflow_port': 5000,
    'minio_enabled': True,
    'minio_port': 9000,
    'minio_console_port': 9001,
    'k8s_enabled': False,
    'k8s_namespace': 'ml-pipeline',
    'k8s_min_replicas': 2,
    'k8s_max_replicas': 10,
}


def prompt(message: str, default, type_cast=str):
    """Prompt user with default value."""
    if isinstance(default, bool):
        default_str = 'y' if default else 'n'
        result = input(f"{message} (y/n) [{default_str}]: ").strip().lower()
        if not result:
            return default
        return result in ('y', 'yes', 'true', '1')
    else:
        result = input(f"{message} [{default}]: ").strip()
        if not result:
            return default
        return type_cast(result)


def generate_config(answers: dict) -> dict:
    """Generate config.yaml content from answers."""
    return {
        'project': {
            'name': answers['project_name'],
            'version': answers['version'],
            'description': answers['description'],
        },
        'model': {
            'name': answers['model_name'],
            'framework': answers['framework'],
            'module': answers['model_module'],
            'class': answers['model_class'],
        },
        'training': {
            'experiment': answers['experiment'],
            'mlflow_enabled': answers['mlflow_enabled'],
            'params': {},
        },
        'serving': {
            'host': '0.0.0.0',
            'port': answers['api_port'],
            'workers': answers['workers'],
        },
        'infrastructure': {
            'mlflow': {
                'enabled': answers['mlflow_enabled'],
                'port': answers['mlflow_port'],
                'backend_store': 'postgresql://mlflow:mlflow@postgres:5432/mlflow',
                'artifact_store': 's3://mlflow-artifacts',
            },
            'minio': {
                'enabled': answers['minio_enabled'],
                'port': answers['minio_port'],
                'console_port': answers['minio_console_port'],
                'access_key': 'minioadmin',
                'secret_key': 'minioadmin',
            },
        },
        'kubernetes': {
            'enabled': answers['k8s_enabled'],
            'namespace': answers['k8s_namespace'],
            'replicas': {
                'min': answers['k8s_min_replicas'],
                'max': answers['k8s_max_replicas'],
            },
            'resources': {
                'requests': {
                    'cpu': '250m',
                    'memory': '256Mi',
                },
                'limits': {
                    'cpu': '1000m',
                    'memory': '1Gi',
                },
            },
        },
    }


def create_project_structure():
    """Create necessary directories if they don't exist."""
    dirs = [
        'src/training',
        'src/serving',
        'examples/iris',
        'docker',
        'k8s/base',
        'k8s/overlays/staging',
        'k8s/overlays/production',
        '.github/workflows',
        'tests',
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    print()
    print("🚀 ML Pipeline Setup Wizard")
    print("━" * 40)
    print()
    print("Press Enter to accept defaults, or type your values.")
    print()
    
    answers = {}
    
    # Project Configuration
    print("Project Configuration")
    print("─" * 25)
    answers['project_name'] = prompt("Project name", DEFAULTS['project_name'])
    answers['version'] = prompt("Version", DEFAULTS['version'])
    answers['description'] = prompt("Description", DEFAULTS['description'])
    print()
    
    # Model Configuration
    print("Model Configuration")
    print("─" * 25)
    answers['model_name'] = prompt("Model name", DEFAULTS['model_name'])
    answers['framework'] = prompt("ML framework (sklearn/pytorch/tensorflow)", DEFAULTS['framework'])
    answers['model_module'] = prompt("Model module path", DEFAULTS['model_module'])
    answers['model_class'] = prompt("Model class name", DEFAULTS['model_class'])
    print()
    
    # Training Configuration
    print("Training Configuration")
    print("─" * 25)
    answers['experiment'] = prompt("Experiment name", DEFAULTS['experiment'])
    answers['mlflow_enabled'] = prompt("Track with MLflow?", DEFAULTS['mlflow_enabled'])
    print()
    
    # Serving Configuration
    print("Serving Configuration")
    print("─" * 25)
    answers['api_port'] = prompt("API port", DEFAULTS['api_port'], int)
    answers['workers'] = prompt("Number of workers", DEFAULTS['workers'], int)
    print()
    
    # Infrastructure
    print("Infrastructure")
    print("─" * 25)
    answers['minio_enabled'] = prompt("Enable MinIO storage?", DEFAULTS['minio_enabled'])
    answers['mlflow_port'] = prompt("MLflow port", DEFAULTS['mlflow_port'], int)
    answers['minio_port'] = prompt("MinIO port", DEFAULTS['minio_port'], int)
    answers['minio_console_port'] = prompt("MinIO console port", DEFAULTS['minio_console_port'], int)
    print()
    
    # Kubernetes (optional)
    print("Kubernetes (optional)")
    print("─" * 25)
    answers['k8s_enabled'] = prompt("Configure Kubernetes?", DEFAULTS['k8s_enabled'])
    if answers['k8s_enabled']:
        answers['k8s_namespace'] = prompt("Namespace", DEFAULTS['k8s_namespace'])
        answers['k8s_min_replicas'] = prompt("Min replicas", DEFAULTS['k8s_min_replicas'], int)
        answers['k8s_max_replicas'] = prompt("Max replicas", DEFAULTS['k8s_max_replicas'], int)
    else:
        answers['k8s_namespace'] = DEFAULTS['k8s_namespace']
        answers['k8s_min_replicas'] = DEFAULTS['k8s_min_replicas']
        answers['k8s_max_replicas'] = DEFAULTS['k8s_max_replicas']
    
    print()
    print("━" * 40)
    print()
    
    # Generate config
    config = generate_config(answers)
    
    # Save config.yaml
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Create project structure
    create_project_structure()
    
    print("✅ Configuration saved to config.yaml")
    print("✅ Project structure created")
    print()
    print("Next steps:")
    print("  1. Implement your model in src/model.py")
    print("  2. Run 'make up' to start infrastructure")
    print("  3. Run 'make train' to train your model")
    print("  4. Run 'make serve' to start the API")
    print()
    print("Happy deploying! 🎉")
    print()


if __name__ == '__main__':
    main()
