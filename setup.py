#!/usr/bin/env python3
"""
ML Pipeline Bootstrap - Interactive Setup Wizard
=================================================

This script provides an interactive command-line interface for configuring
a new ML pipeline project. It guides users through setting up:

- Project metadata (name, version, description)
- Model configuration (name, framework, module path)
- Training settings (experiment name, MLflow integration)
- Serving configuration (API port, workers)
- Infrastructure options (MLflow, MinIO)
- Kubernetes deployment settings (optional)

Usage:
    python setup.py

The wizard will prompt for each configuration option, showing default
values in brackets. Press Enter to accept defaults, or type a custom value.

After completion, the script:
1. Generates a config.yaml file with all settings
2. Creates the project directory structure
3. Displays next steps for the user

Example:
    $ python setup.py

    🚀 ML Pipeline Setup Wizard
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Project name [ml-pipeline]: my-classifier
    ...
"""

import yaml
from pathlib import Path

# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================

DEFAULTS = {
    # Project settings
    "project_name": "ml-pipeline",
    "version": "1.0.0",
    "description": "ML model deployment pipeline",
    # Model settings
    "model_name": "my-model",
    "framework": "sklearn",
    "model_module": "src.model",
    "model_class": "ModelWrapper",
    # Training settings
    "experiment": "default",
    # Serving settings
    "api_port": 8000,
    "workers": 4,
    # Infrastructure settings
    "mlflow_enabled": True,
    "mlflow_port": 5000,
    "minio_enabled": True,
    "minio_port": 9000,
    "minio_console_port": 9001,
    # Kubernetes settings
    "k8s_enabled": False,
    "k8s_namespace": "ml-pipeline",
    "k8s_min_replicas": 2,
    "k8s_max_replicas": 10,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def prompt(message: str, default, type_cast=str):
    """
    Display a prompt to the user and return their input or the default value.

    This function handles different input types:
    - Boolean: Shows (y/n) prompt, returns True/False
    - String/Number: Shows [default] prompt, returns typed value

    Args:
        message: The prompt message to display to the user
        default: The default value to use if user presses Enter
        type_cast: Function to convert string input to desired type
                   (default: str, can be int, float, etc.)

    Returns:
        The user's input converted to the appropriate type,
        or the default value if no input was provided

    Examples:
        >>> prompt("API port", 8000, int)
        API port [8000]: 3000
        3000

        >>> prompt("Enable MLflow?", True)
        Enable MLflow? (y/n) [y]:
        True
    """
    if isinstance(default, bool):
        # Boolean prompt: show (y/n) format
        default_str = "y" if default else "n"
        result = input(f"{message} (y/n) [{default_str}]: ").strip().lower()
        if not result:
            return default
        return result in ("y", "yes", "true", "1")
    else:
        # Standard prompt: show [default] format
        result = input(f"{message} [{default}]: ").strip()
        if not result:
            return default
        return type_cast(result)


def generate_config(answers: dict) -> dict:
    """
    Generate a complete configuration dictionary from user answers.

    Takes the collected user responses and structures them into the
    hierarchical format expected by config.yaml. This includes:
    - Project metadata
    - Model configuration
    - Training parameters
    - Serving settings
    - Infrastructure configuration
    - Kubernetes deployment options

    Args:
        answers: Dictionary containing all user responses from the wizard.
                Keys should match the DEFAULTS dictionary structure.

    Returns:
        A nested dictionary ready to be serialized to YAML format.
        The structure matches what src/config.py expects to load.

    Example output structure:
        {
            'project': {'name': '...', 'version': '...'},
            'model': {'name': '...', 'framework': '...'},
            'training': {...},
            'serving': {...},
            'infrastructure': {...},
            'kubernetes': {...}
        }
    """
    return {
        "project": {
            "name": answers["project_name"],
            "version": answers["version"],
            "description": answers["description"],
        },
        "model": {
            "name": answers["model_name"],
            "framework": answers["framework"],
            "module": answers["model_module"],
            "class": answers["model_class"],
        },
        "training": {
            "experiment": answers["experiment"],
            "mlflow_enabled": answers["mlflow_enabled"],
            "params": {},  # User adds training params here after setup
        },
        "serving": {
            "host": "0.0.0.0",  # Bind to all interfaces
            "port": answers["api_port"],
            "workers": answers["workers"],
        },
        "infrastructure": {
            "mlflow": {
                "enabled": answers["mlflow_enabled"],
                "port": answers["mlflow_port"],
                # PostgreSQL connection for MLflow backend store
                "backend_store": "postgresql://mlflow:mlflow@postgres:5432/mlflow",
                # S3-compatible storage for artifacts
                "artifact_store": "s3://mlflow-artifacts",
            },
            "minio": {
                "enabled": answers["minio_enabled"],
                "port": answers["minio_port"],
                "console_port": answers["minio_console_port"],
                # Default MinIO credentials (change in production!)
                "access_key": "minioadmin",
                "secret_key": "minioadmin",
            },
        },
        "kubernetes": {
            "enabled": answers["k8s_enabled"],
            "namespace": answers["k8s_namespace"],
            "replicas": {
                "min": answers["k8s_min_replicas"],
                "max": answers["k8s_max_replicas"],
            },
            "resources": {
                "requests": {
                    "cpu": "250m",
                    "memory": "256Mi",
                },
                "limits": {
                    "cpu": "1000m",
                    "memory": "1Gi",
                },
            },
        },
    }


def create_project_structure():
    """
    Create the standard project directory structure.

    Creates all necessary directories for the ML pipeline project
    if they don't already exist. This includes:

    - src/training/     : Training scripts and utilities
    - src/serving/      : FastAPI application and schemas
    - examples/iris/    : Example Iris classifier implementation
    - docker/           : Dockerfiles for build and serve
    - k8s/base/         : Base Kubernetes manifests
    - k8s/overlays/     : Environment-specific K8s overlays
    - .github/workflows/: CI/CD pipeline definitions
    - tests/            : Unit and integration tests

    This function is idempotent - safe to run multiple times.
    """
    dirs = [
        "src/training",
        "src/serving",
        "examples/iris",
        "docker",
        "k8s/base",
        "k8s/overlays/staging",
        "k8s/overlays/production",
        ".github/workflows",
        "tests",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN WIZARD FUNCTION
# =============================================================================


def main():
    """
    Run the interactive setup wizard.

    This is the main entry point for the setup script. It:
    1. Displays a welcome banner
    2. Collects user input for each configuration section
    3. Generates and saves config.yaml
    4. Creates the project directory structure
    5. Displays next steps for the user

    The wizard is organized into logical sections:
    - Project Configuration: Basic project metadata
    - Model Configuration: ML model settings
    - Training Configuration: Experiment tracking options
    - Serving Configuration: API server settings
    - Infrastructure: MLflow and MinIO options
    - Kubernetes: Optional K8s deployment settings

    Each prompt shows the default value in brackets. Users can:
    - Press Enter to accept the default
    - Type a custom value to override

    Raises:
        KeyboardInterrupt: If user presses Ctrl+C to cancel
        IOError: If config.yaml cannot be written
    """
    # Display welcome banner
    print()
    print("🚀 ML Pipeline Setup Wizard")
    print("━" * 40)
    print()
    print("Press Enter to accept defaults, or type your values.")
    print()

    # Collect all user answers
    answers = {}

    # -------------------------------------------------------------------------
    # Section 1: Project Configuration
    # -------------------------------------------------------------------------
    print("Project Configuration")
    print("─" * 25)
    answers["project_name"] = prompt("Project name", DEFAULTS["project_name"])
    answers["version"] = prompt("Version", DEFAULTS["version"])
    answers["description"] = prompt("Description", DEFAULTS["description"])
    print()

    # -------------------------------------------------------------------------
    # Section 2: Model Configuration
    # -------------------------------------------------------------------------
    print("Model Configuration")
    print("─" * 25)
    answers["model_name"] = prompt("Model name", DEFAULTS["model_name"])
    answers["framework"] = prompt(
        "ML framework (sklearn/pytorch/tensorflow)", DEFAULTS["framework"]
    )
    answers["model_module"] = prompt("Model module path", DEFAULTS["model_module"])
    answers["model_class"] = prompt("Model class name", DEFAULTS["model_class"])
    print()

    # -------------------------------------------------------------------------
    # Section 3: Training Configuration
    # -------------------------------------------------------------------------
    print("Training Configuration")
    print("─" * 25)
    answers["experiment"] = prompt("Experiment name", DEFAULTS["experiment"])
    answers["mlflow_enabled"] = prompt("Track with MLflow?", DEFAULTS["mlflow_enabled"])
    print()

    # -------------------------------------------------------------------------
    # Section 4: Serving Configuration
    # -------------------------------------------------------------------------
    print("Serving Configuration")
    print("─" * 25)
    answers["api_port"] = prompt("API port", DEFAULTS["api_port"], int)
    answers["workers"] = prompt("Number of workers", DEFAULTS["workers"], int)
    print()

    # -------------------------------------------------------------------------
    # Section 5: Infrastructure
    # -------------------------------------------------------------------------
    print("Infrastructure")
    print("─" * 25)
    answers["minio_enabled"] = prompt(
        "Enable MinIO storage?", DEFAULTS["minio_enabled"]
    )
    answers["mlflow_port"] = prompt("MLflow port", DEFAULTS["mlflow_port"], int)
    answers["minio_port"] = prompt("MinIO port", DEFAULTS["minio_port"], int)
    answers["minio_console_port"] = prompt(
        "MinIO console port", DEFAULTS["minio_console_port"], int
    )
    print()

    # -------------------------------------------------------------------------
    # Section 6: Kubernetes (Optional)
    # -------------------------------------------------------------------------
    print("Kubernetes (optional)")
    print("─" * 25)
    answers["k8s_enabled"] = prompt("Configure Kubernetes?", DEFAULTS["k8s_enabled"])

    if answers["k8s_enabled"]:
        # Only ask K8s questions if user wants to configure it
        answers["k8s_namespace"] = prompt("Namespace", DEFAULTS["k8s_namespace"])
        answers["k8s_min_replicas"] = prompt(
            "Min replicas", DEFAULTS["k8s_min_replicas"], int
        )
        answers["k8s_max_replicas"] = prompt(
            "Max replicas", DEFAULTS["k8s_max_replicas"], int
        )
    else:
        # Use defaults if K8s not enabled
        answers["k8s_namespace"] = DEFAULTS["k8s_namespace"]
        answers["k8s_min_replicas"] = DEFAULTS["k8s_min_replicas"]
        answers["k8s_max_replicas"] = DEFAULTS["k8s_max_replicas"]

    # -------------------------------------------------------------------------
    # Generate and Save Configuration
    # -------------------------------------------------------------------------
    print()
    print("━" * 40)
    print()

    # Generate config dictionary from answers
    config = generate_config(answers)

    # Write config.yaml with human-readable formatting
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create project directories
    create_project_structure()

    # -------------------------------------------------------------------------
    # Display Success Message and Next Steps
    # -------------------------------------------------------------------------
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


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
