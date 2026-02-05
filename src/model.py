"""
Model Interface Module
======================

This module defines the abstract base class that all models must implement
to integrate with the ML Pipeline. The ModelWrapper interface ensures
consistent behavior across different ML frameworks and model types.

Design Philosophy:
    The interface is intentionally minimal, requiring only the essential
    methods needed for training, inference, and persistence. This allows
    maximum flexibility while ensuring compatibility with:

    - Training pipeline (src/training/train.py)
    - Serving API (src/serving/main.py)
    - MLflow experiment tracking and model registry

Implementing a Custom Model:
    1. Create a new class that inherits from ModelWrapper
    2. Implement all abstract methods
    3. Update config.yaml to point to your model module

Example Implementation:
    class MyModel(ModelWrapper):
        def __init__(self, config: dict):
            self.model = None
            self.config = config

        def train(self, X, y, params: dict) -> dict:
            self.model = SomeAlgorithm(**params)
            self.model.fit(X, y)
            return {'accuracy': self.model.score(X, y)}

        def predict(self, features: list) -> dict:
            pred = self.model.predict([features])[0]
            return {'prediction': pred}

        def save(self, path: str):
            joblib.dump(self.model, f"{path}/model.pkl")

        def load(self, path: str):
            self.model = joblib.load(f"{path}/model.pkl")

See Also:
    - examples/iris/model.py: Complete working example
    - src/training/train.py: How models are trained
    - src/serving/main.py: How models are served
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ModelWrapper(ABC):
    """
    Abstract base class defining the model interface.

    All models in the ML Pipeline must inherit from this class and implement
    its abstract methods. This ensures consistent behavior across different
    ML frameworks (scikit-learn, PyTorch, TensorFlow, etc.).

    The interface separates concerns:
    - __init__: Setup and configuration
    - train: Learning from data
    - predict: Making predictions
    - save/load: Persistence

    Attributes:
        This base class defines no attributes. Subclasses should define
        their own attributes as needed (e.g., self.model, self.scaler).

    Note:
        The config dictionary passed to __init__ contains the full
        configuration from config.yaml. Access settings via:
        config.get('model', {}).get('param_name')
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model wrapper.

        This method should:
        1. Store the configuration for later use
        2. Initialize any preprocessing components (scalers, encoders)
        3. Optionally set up model architecture (for neural networks)

        The model itself (self.model) may be None until train() or load()
        is called.

        Args:
            config: Complete configuration dictionary from config.yaml.
                   Access model-specific settings via config['model']
                   or config.get('training', {}).get('params', {})

        Example:
            def __init__(self, config):
                self.config = config
                self.model = None
                self.scaler = StandardScaler()
                self.label_encoder = LabelEncoder()
        """
        pass

    @abstractmethod
    def train(self, X, y, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the model on provided data.

        This method should:
        1. Apply any necessary preprocessing to X
        2. Create/initialize the model with given params
        3. Fit the model to the data
        4. Calculate and return training metrics

        The returned metrics dictionary will be:
        - Logged to MLflow (if enabled)
        - Displayed to the user
        - Used for model comparison

        Args:
            X: Training features. Type depends on your model:
               - NumPy array (n_samples, n_features) for sklearn
               - PyTorch tensor for PyTorch models
               - TensorFlow tensor for TF models

            y: Training labels/targets. Shape depends on task:
               - (n_samples,) for classification
               - (n_samples, n_outputs) for multi-output

            params: Training hyperparameters from config.yaml.
                   Common params: learning_rate, epochs, batch_size,
                   n_estimators, max_depth, etc.

        Returns:
            Dictionary of metric names to values. Common metrics:
            - 'accuracy': Classification accuracy (0-1)
            - 'loss': Training loss value
            - 'f1': F1 score
            - 'mse': Mean squared error (regression)

            All values should be JSON-serializable (float, int, str).

        Example:
            def train(self, X, y, params):
                X_scaled = self.scaler.fit_transform(X)

                self.model = RandomForest(
                    n_estimators=params.get('n_estimators', 100)
                )
                self.model.fit(X_scaled, y)

                return {
                    'accuracy': self.model.score(X_scaled, y),
                    'n_estimators': self.model.n_estimators
                }
        """
        pass

    @abstractmethod
    def predict(self, features: List[Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single input.

        This method is called by the serving API for each prediction
        request. It should:
        1. Apply the same preprocessing used during training
        2. Run inference using the trained model
        3. Return prediction with any relevant metadata

        The method receives raw features and should handle all
        necessary transformations internally.

        Args:
            features: Input features as a flat list.
                     Example: [5.1, 3.5, 1.4, 0.2] for Iris

                     The features should match the order and format
                     expected by your model.

        Returns:
            Dictionary containing at minimum:
            - 'prediction': The model's prediction (class label, value, etc.)

            Optional additional fields:
            - 'probability': Confidence score (0-1)
            - 'class_name': Human-readable class label
            - 'probabilities': Dict of class -> probability
            - Any other relevant metadata

        Raises:
            ValueError: If model hasn't been trained/loaded
            ValueError: If features have incorrect format

        Example:
            def predict(self, features):
                if self.model is None:
                    raise ValueError("Model not loaded")

                X = np.array(features).reshape(1, -1)
                X_scaled = self.scaler.transform(X)

                pred = self.model.predict(X_scaled)[0]
                prob = self.model.predict_proba(X_scaled).max()

                return {
                    'prediction': int(pred),
                    'class_name': self.classes[pred],
                    'probability': float(prob)
                }
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model artifacts to the specified path.

        This method should persist everything needed to restore the
        model for inference:
        - Trained model weights/parameters
        - Preprocessing components (scalers, encoders)
        - Any configuration needed for prediction

        The path is a directory - create files within it as needed.
        Use standard formats for portability:
        - pickle (.pkl) for sklearn models
        - .pt/.pth for PyTorch
        - SavedModel format for TensorFlow
        - ONNX for cross-framework compatibility

        Args:
            path: Directory path where artifacts should be saved.
                 The directory will be created if it doesn't exist.

        Example:
            def save(self, path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)

                # Save model
                with open(path / 'model.pkl', 'wb') as f:
                    pickle.dump(self.model, f)

                # Save scaler
                with open(path / 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)

                # Save metadata
                with open(path / 'metadata.json', 'w') as f:
                    json.dump({'classes': self.classes}, f)
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model artifacts from the specified path.

        This method should restore the model to a state ready for
        inference. It must load all components saved by save():
        - Model weights/parameters
        - Preprocessing components
        - Any metadata needed for prediction

        After calling load(), the model should be ready to accept
        predict() calls.

        Args:
            path: Directory path containing saved artifacts.
                 Should match the path used in save().

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If artifacts are incompatible/corrupted

        Example:
            def load(self, path):
                path = Path(path)

                # Load model
                with open(path / 'model.pkl', 'rb') as f:
                    self.model = pickle.load(f)

                # Load scaler
                with open(path / 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information.

        Override this method to provide custom model information.
        The default implementation returns basic class information.

        This information is:
        - Returned by the /model/info API endpoint
        - Logged to MLflow as model metadata
        - Useful for debugging and model tracking

        Returns:
            Dictionary containing model information:
            - 'class': Model class name
            - 'module': Module path
            - Additional custom fields as needed

        Example override:
            def get_model_info(self):
                info = super().get_model_info()
                info.update({
                    'algorithm': 'RandomForest',
                    'n_features': self.model.n_features_in_,
                    'classes': self.classes
                })
                return info
        """
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
