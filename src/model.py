"""
Model Interface - Base class for all models.

Implement this interface to integrate your model with the pipeline.
See examples/iris/model.py for a complete example.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ModelWrapper(ABC):
    """
    Base class that all models must implement.
    
    This interface ensures your model works with:
    - Training pipeline (train.py)
    - Serving API (FastAPI)
    - MLflow tracking and registry
    
    Example:
        class MyModel(ModelWrapper):
            def __init__(self, config: dict):
                self.model = None
            
            def train(self, X, y, params: dict) -> dict:
                self.model = SomeModel(**params)
                self.model.fit(X, y)
                return {'accuracy': self.model.score(X, y)}
            
            def predict(self, features: list) -> dict:
                prediction = self.model.predict([features])[0]
                return {'prediction': prediction}
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model wrapper.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        pass
    
    @abstractmethod
    def train(self, X, y, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Training features (numpy array or similar)
            y: Training labels (numpy array or similar)
            params: Training parameters from config
        
        Returns:
            Dictionary of metrics (e.g., {'accuracy': 0.95, 'f1': 0.93})
        """
        pass
    
    @abstractmethod
    def predict(self, features: List[Any]) -> Dict[str, Any]:
        """
        Make a prediction.
        
        Args:
            features: Input features as a list
        
        Returns:
            Dictionary with at least 'prediction' key
            Example: {'prediction': 1, 'probability': 0.95}
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model artifacts to path.
        
        Args:
            path: Directory path to save artifacts
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model artifacts from path.
        
        Args:
            path: Directory path to load artifacts from
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Override this to provide custom model info.
        
        Returns:
            Dictionary with model information
        """
        return {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
        }
