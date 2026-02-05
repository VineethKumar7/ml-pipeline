"""
Iris Classifier - Example ModelWrapper Implementation
=====================================================

This module provides a complete, working example of how to implement
the ModelWrapper interface for the ML Pipeline. It demonstrates:

- Proper implementation of all required methods
- Preprocessing with scikit-learn (StandardScaler)
- Training a Random Forest classifier
- Returning predictions with probabilities
- Saving and loading model artifacts

The Iris classifier serves as:
1. A reference implementation for creating custom models
2. A working example for testing the ML pipeline
3. Documentation of best practices

Model Architecture:
    Input (4 features) -> StandardScaler -> RandomForestClassifier -> Output (3 classes)

    Features: sepal_length, sepal_width, petal_length, petal_width
    Classes: setosa (0), versicolor (1), virginica (2)

Usage:
    # Using with the training pipeline
    make train  # Trains using config.yaml settings

    # Using directly
    from examples.iris.model import ModelWrapper
    from examples.iris.data import load_data

    X_train, y_train, X_test, y_test = load_data()
    model = ModelWrapper({})
    metrics = model.train(X_train, y_train, {'n_estimators': 100})
    result = model.predict([5.1, 3.5, 1.4, 0.2])

See Also:
    - src/model.py: Abstract base class definition
    - examples/iris/data.py: Data loading module
    - examples/iris/README.md: Detailed usage instructions
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ModelWrapper:
    """
    Iris flower classifier using Random Forest.

    This class implements the ModelWrapper interface for classifying
    iris flowers into three species based on measurements of their
    sepals and petals.

    The model uses:
    - StandardScaler for feature normalization
    - RandomForestClassifier for classification

    Attributes:
        config (dict): Configuration dictionary from config.yaml
        model (RandomForestClassifier): The trained classifier (None until trained)
        scaler (StandardScaler): Feature normalizer
        CLASSES (list): Class names for prediction output

    Example:
        model = ModelWrapper({})

        # Train
        metrics = model.train(X_train, y_train, {'n_estimators': 100})
        print(f"Accuracy: {metrics['accuracy']}")

        # Predict
        result = model.predict([5.1, 3.5, 1.4, 0.2])
        print(f"Class: {result['class_name']}, Prob: {result['probability']}")
    """

    # Human-readable class names for output
    CLASSES = ["setosa", "versicolor", "virginica"]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Iris classifier.

        Sets up the model wrapper with configuration and initializes
        the preprocessing components. The actual classifier is not
        created until train() is called.

        Args:
            config: Configuration dictionary from config.yaml.
                   Currently not used, but available for customization.

        Example:
            model = ModelWrapper({'model': {'name': 'iris-classifier'}})
        """
        self.config = config
        self.model = None  # Created during training
        self.scaler = StandardScaler()  # For feature normalization

    def train(self, X, y, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the Random Forest classifier on Iris data.

        Performs the following steps:
        1. Fit the StandardScaler and transform features
        2. Create RandomForestClassifier with given params
        3. Fit the classifier to the normalized data
        4. Calculate and return training metrics

        Args:
            X: Training features, shape (n_samples, 4)
               Features: [sepal_length, sepal_width, petal_length, petal_width]

            y: Training labels, shape (n_samples,)
               Integer class labels: 0, 1, or 2

            params: Training hyperparameters:
               - n_estimators (int): Number of trees (default: 100)
               - max_depth (int): Maximum tree depth (default: 5)
               - random_state (int): Random seed (default: 42)

        Returns:
            Dictionary containing training metrics:
            - accuracy: Training accuracy (0-1)
            - n_estimators: Number of trees used
            - max_depth: Max depth used
            - feature_importance_*: Importance of each feature

        Example:
            metrics = model.train(X_train, y_train, {
                'n_estimators': 200,
                'max_depth': 10
            })
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        """
        # Step 1: Fit scaler and normalize features
        # fit_transform learns the mean/std and applies normalization
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Create the Random Forest classifier with parameters
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
            n_jobs=-1,  # Use all CPU cores for parallel training
        )

        # Step 3: Train the model
        self.model.fit(X_scaled, y)

        # Step 4: Calculate metrics
        train_accuracy = self.model.score(X_scaled, y)
        importances = self.model.feature_importances_

        # Return comprehensive metrics dictionary
        return {
            "accuracy": float(train_accuracy),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth or 0,  # None -> 0 for logging
            # Feature importances (useful for model interpretability)
            "feature_importance_sepal_length": float(importances[0]),
            "feature_importance_sepal_width": float(importances[1]),
            "feature_importance_petal_length": float(importances[2]),
            "feature_importance_petal_width": float(importances[3]),
        }

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.

        Calculates performance metrics on held-out data.
        The features are normalized using the scaler fitted during training.

        Args:
            X: Test features, shape (n_samples, 4)
            y: True labels, shape (n_samples,)

        Returns:
            Dictionary with evaluation metrics:
            - accuracy: Test accuracy (0-1)

        Example:
            test_metrics = model.evaluate(X_test, y_test)
            print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        """
        # Use transform (not fit_transform) to apply same normalization
        X_scaled = self.scaler.transform(X)
        accuracy = self.model.score(X_scaled, y)
        return {"accuracy": float(accuracy)}

    def predict(self, features: List[Any]) -> Dict[str, Any]:
        """
        Predict the class of an iris flower.

        Takes raw feature values, normalizes them using the fitted
        scaler, and returns the predicted class with probabilities.

        Args:
            features: List of 4 measurements:
                     [sepal_length, sepal_width, petal_length, petal_width]
                     Values should be in centimeters.

        Returns:
            Dictionary containing:
            - prediction: Integer class label (0, 1, or 2)
            - class_name: Human-readable class name
            - probability: Confidence for predicted class (0-1)
            - probabilities: Dict mapping class names to probabilities

        Raises:
            ValueError: If model hasn't been trained or loaded

        Example:
            # Typical setosa measurements
            result = model.predict([5.1, 3.5, 1.4, 0.2])

            print(result)
            # {
            #     'prediction': 0,
            #     'class_name': 'setosa',
            #     'probability': 0.97,
            #     'probabilities': {
            #         'setosa': 0.97,
            #         'versicolor': 0.02,
            #         'virginica': 0.01
            #     }
            # }
        """
        # Validate model is ready
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        # Reshape features from list to 2D array (required by sklearn)
        X = np.array(features).reshape(1, -1)

        # Normalize using the fitted scaler
        X_scaled = self.scaler.transform(X)

        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Build response dictionary
        return {
            "prediction": int(prediction),
            "class_name": self.CLASSES[prediction],
            "probability": float(probabilities[prediction]),
            "probabilities": {
                self.CLASSES[i]: float(p) for i, p in enumerate(probabilities)
            },
        }

    def save(self, path: str) -> None:
        """
        Save model artifacts to disk.

        Persists both the trained classifier and the fitted scaler
        so they can be restored for inference.

        Artifacts saved:
        - model.pkl: The trained RandomForestClassifier
        - scaler.pkl: The fitted StandardScaler

        Args:
            path: Directory path where artifacts will be saved.
                 Created if it doesn't exist.

        Example:
            model.save('models/iris-classifier')
            # Creates:
            #   models/iris-classifier/model.pkl
            #   models/iris-classifier/scaler.pkl
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the trained model
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save the fitted scaler (important for consistent preprocessing)
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"✅ Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model artifacts from disk.

        Restores the classifier and scaler from a previous save.
        After loading, the model is ready for predictions.

        Args:
            path: Directory containing saved artifacts
                 (model.pkl and scaler.pkl)

        Raises:
            FileNotFoundError: If artifacts don't exist

        Example:
            model = ModelWrapper({})
            model.load('models/iris-classifier')
            result = model.predict([5.1, 3.5, 1.4, 0.2])
        """
        path = Path(path)

        # Load the trained model
        with open(path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load the fitted scaler
        with open(path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        print(f"✅ Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get metadata about the model.

        Returns information useful for debugging, logging,
        and API responses.

        Returns:
            Dictionary containing:
            - class: Model class name
            - module: Python module path
            - framework: ML framework used
            - algorithm: Algorithm name
            - classes: List of class names
            - n_estimators: Number of trees (if trained)
            - max_depth: Max tree depth (if trained)
            - n_features: Number of input features (if trained)
        """
        info = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "framework": "sklearn",
            "algorithm": "RandomForestClassifier",
            "classes": self.CLASSES,
        }

        # Add model-specific info if trained
        if self.model is not None:
            info.update(
                {
                    "n_estimators": self.model.n_estimators,
                    "max_depth": self.model.max_depth,
                    "n_features": self.model.n_features_in_,
                }
            )

        return info
