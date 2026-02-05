"""
Configuration Loader Module
===========================

This module provides a centralized configuration management system for the
ML Pipeline. It implements a singleton pattern to ensure consistent access
to configuration values across all modules.

Features:
- Loads configuration from config.yaml
- Supports environment variable overrides
- Provides typed access to nested configuration values
- Thread-safe singleton implementation

Usage:
    from src.config import get_config

    config = get_config()

    # Access nested values
    model_name = config.get('model', 'name')

    # Access sections as dictionaries
    serving_config = config.serving

    # Get MLflow URI (with env override support)
    mlflow_uri = config.mlflow_tracking_uri

Environment Variables:
    The following environment variables can override config values:

    - MLFLOW_TRACKING_URI: Override MLflow server URL
    - MODEL_NAME: Override the model name
    - API_PORT: Override the serving port
    - API_WORKERS: Override the number of workers

Example config.yaml:
    project:
      name: my-model
      version: 1.0.0

    model:
      name: classifier
      module: src.model
      class: ModelWrapper

    serving:
      port: 8000
      workers: 4
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration manager implementing the singleton pattern.

    This class loads and manages all configuration settings for the ML Pipeline.
    It ensures that configuration is loaded only once and provides convenient
    access methods for retrieving values.

    The singleton pattern guarantees that all parts of the application
    share the same configuration instance, preventing inconsistencies.

    Attributes:
        _instance: Class-level singleton instance
        _config: Dictionary containing all configuration values

    Example:
        # Get the singleton instance
        config = Config()

        # Access values
        port = config.get('serving', 'port', default=8000)

        # Access sections
        model_config = config.model
    """

    # Class-level singleton instance
    _instance: Optional["Config"] = None

    # Configuration data storage
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """
        Create or return the singleton Config instance.

        This method implements the singleton pattern by checking if an
        instance already exists. If not, it creates one and loads the
        configuration from config.yaml.

        Returns:
            The singleton Config instance

        Raises:
            FileNotFoundError: If config.yaml doesn't exist
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        Load configuration from config.yaml file.

        Reads the YAML configuration file from the current working directory
        and parses it into a dictionary. After loading, applies any
        environment variable overrides.

        Raises:
            FileNotFoundError: If config.yaml is not found in the current
                             directory. Users should run setup.py first.
            yaml.YAMLError: If the config file contains invalid YAML
        """
        config_path = Path("config.yaml")

        if not config_path.exists():
            raise FileNotFoundError(
                "config.yaml not found. Run 'python setup.py' first."
            )

        # Load YAML configuration
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """
        Apply environment variable overrides to configuration.

        Checks for specific environment variables and overrides the
        corresponding configuration values. This allows for runtime
        configuration without modifying config.yaml, useful for:

        - Container deployments with environment injection
        - CI/CD pipelines with dynamic configuration
        - Local development with temporary overrides

        Supported environment variables:
            MLFLOW_TRACKING_URI -> infrastructure.mlflow.tracking_uri
            MODEL_NAME -> model.name
            API_PORT -> serving.port
            API_WORKERS -> serving.workers
        """
        # Mapping of environment variables to config paths
        env_mappings = {
            "MLFLOW_TRACKING_URI": ("infrastructure", "mlflow", "tracking_uri"),
            "MODEL_NAME": ("model", "name"),
            "API_PORT": ("serving", "port"),
            "API_WORKERS": ("serving", "workers"),
        }

        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(path, value)

    def _set_nested(self, path: tuple, value: Any):
        """
        Set a nested configuration value using a path tuple.

        Navigates through the configuration dictionary using the provided
        path and sets the value at the final key. Automatically creates
        intermediate dictionaries if they don't exist.

        Args:
            path: Tuple of keys representing the path to the value
                  e.g., ('serving', 'port') for config['serving']['port']
            value: The value to set. Will be type-cast to match the
                   original value's type if possible.

        Example:
            # Set config['model']['name'] = 'new-model'
            config._set_nested(('model', 'name'), 'new-model')
        """
        d = self._config

        # Navigate to the parent of the target key
        for key in path[:-1]:
            d = d.setdefault(key, {})

        # Type-cast value to match original type
        original = d.get(path[-1])
        if isinstance(original, int):
            value = int(value)
        elif isinstance(original, bool):
            value = value.lower() in ("true", "1", "yes")

        # Set the value
        d[path[-1]] = value

    def get(self, *keys, default=None) -> Any:
        """
        Get a configuration value by key path.

        Safely retrieves nested configuration values, returning a default
        if any key in the path doesn't exist.

        Args:
            *keys: Variable number of keys forming the path to the value
            default: Value to return if the path doesn't exist (default: None)

        Returns:
            The configuration value at the specified path, or default

        Examples:
            # Get serving port with default
            port = config.get('serving', 'port', default=8000)

            # Get nested value
            mlflow_port = config.get('infrastructure', 'mlflow', 'port')

            # Returns None if path doesn't exist
            missing = config.get('nonexistent', 'key')
        """
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default

        return value

    # =========================================================================
    # CONVENIENCE PROPERTIES
    # =========================================================================
    # These properties provide direct access to top-level configuration sections

    @property
    def project(self) -> Dict[str, Any]:
        """
        Get the project configuration section.

        Returns:
            Dictionary containing project settings:
            - name: Project name
            - version: Semantic version string
            - description: Project description
        """
        return self._config.get("project", {})

    @property
    def model(self) -> Dict[str, Any]:
        """
        Get the model configuration section.

        Returns:
            Dictionary containing model settings:
            - name: Model name for MLflow registry
            - framework: ML framework (sklearn/pytorch/tensorflow)
            - module: Python module path (e.g., 'src.model')
            - class: Model wrapper class name
        """
        return self._config.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        """
        Get the training configuration section.

        Returns:
            Dictionary containing training settings:
            - experiment: MLflow experiment name
            - mlflow_enabled: Whether to log to MLflow
            - params: Training hyperparameters
        """
        return self._config.get("training", {})

    @property
    def serving(self) -> Dict[str, Any]:
        """
        Get the serving configuration section.

        Returns:
            Dictionary containing API server settings:
            - host: Bind address (default: 0.0.0.0)
            - port: Server port (default: 8000)
            - workers: Number of uvicorn workers
        """
        return self._config.get("serving", {})

    @property
    def infrastructure(self) -> Dict[str, Any]:
        """
        Get the infrastructure configuration section.

        Returns:
            Dictionary containing infrastructure settings:
            - mlflow: MLflow server configuration
            - minio: MinIO storage configuration
        """
        return self._config.get("infrastructure", {})

    @property
    def kubernetes(self) -> Dict[str, Any]:
        """
        Get the Kubernetes configuration section.

        Returns:
            Dictionary containing K8s deployment settings:
            - enabled: Whether K8s deployment is configured
            - namespace: Target namespace
            - replicas: Min/max replica counts
            - resources: CPU/memory requests and limits
        """
        return self._config.get("kubernetes", {})

    @property
    def mlflow_tracking_uri(self) -> str:
        """
        Get the MLflow tracking server URI.

        Checks for MLFLOW_TRACKING_URI environment variable first,
        then falls back to constructing URL from config.

        Returns:
            Full URL to the MLflow tracking server
            e.g., 'http://localhost:5000'
        """
        port = self.get("infrastructure", "mlflow", "port", default=5000)
        return os.environ.get("MLFLOW_TRACKING_URI", f"http://localhost:{port}")

    def __repr__(self):
        """Return string representation of config for debugging."""
        return f"Config({self._config})"


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# =============================================================================


def get_config() -> Config:
    """
    Get the singleton Config instance.

    This is the recommended way to access configuration throughout
    the application. It ensures all modules share the same config.

    Returns:
        The singleton Config instance

    Example:
        from src.config import get_config

        config = get_config()
        model_name = config.get('model', 'name')
    """
    return Config()
