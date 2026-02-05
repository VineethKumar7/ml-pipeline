"""
Configuration loader utility.

Loads config.yaml and provides access to all settings.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for ML Pipeline."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from config.yaml."""
        config_path = Path('config.yaml')
        
        if not config_path.exists():
            raise FileNotFoundError(
                "config.yaml not found. Run 'python setup.py' first."
            )
        
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'MLFLOW_TRACKING_URI': ('infrastructure', 'mlflow', 'tracking_uri'),
            'MODEL_NAME': ('model', 'name'),
            'API_PORT': ('serving', 'port'),
            'API_WORKERS': ('serving', 'workers'),
        }
        
        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(path, value)
    
    def _set_nested(self, path: tuple, value: Any):
        """Set a nested config value."""
        d = self._config
        for key in path[:-1]:
            d = d.setdefault(key, {})
        
        # Try to cast to original type
        original = d.get(path[-1])
        if isinstance(original, int):
            value = int(value)
        elif isinstance(original, bool):
            value = value.lower() in ('true', '1', 'yes')
        
        d[path[-1]] = value
    
    def get(self, *keys, default=None) -> Any:
        """Get a config value by key path."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
    
    @property
    def project(self) -> Dict[str, Any]:
        return self._config.get('project', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get('training', {})
    
    @property
    def serving(self) -> Dict[str, Any]:
        return self._config.get('serving', {})
    
    @property
    def infrastructure(self) -> Dict[str, Any]:
        return self._config.get('infrastructure', {})
    
    @property
    def kubernetes(self) -> Dict[str, Any]:
        return self._config.get('kubernetes', {})
    
    @property
    def mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        port = self.get('infrastructure', 'mlflow', 'port', default=5000)
        return os.environ.get('MLFLOW_TRACKING_URI', f'http://localhost:{port}')
    
    def __repr__(self):
        return f"Config({self._config})"


def get_config() -> Config:
    """Get the singleton config instance."""
    return Config()
