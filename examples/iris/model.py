"""
Iris Classifier - Example ModelWrapper Implementation

Demonstrates a complete model integration with:
- Training with scikit-learn
- MLflow logging
- Prediction with probabilities
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class ModelWrapper:
    """
    Iris classifier using Random Forest.
    
    This example shows how to implement the ModelWrapper interface
    for a scikit-learn model with preprocessing.
    """
    
    CLASSES = ['setosa', 'versicolor', 'virginica']
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model wrapper."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Training features (n_samples, 4)
            y: Training labels (n_samples,)
            params: Training parameters
                - n_estimators: Number of trees (default: 100)
                - max_depth: Max tree depth (default: 5)
                - random_state: Random seed (default: 42)
        
        Returns:
            Dictionary of training metrics
        """
        # Preprocess
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model with params
        self.model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            random_state=params.get('random_state', 42),
            n_jobs=-1,
        )
        
        # Train
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        train_accuracy = self.model.score(X_scaled, y)
        
        # Feature importances
        importances = self.model.feature_importances_
        
        return {
            'accuracy': float(train_accuracy),
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth or 0,
            'feature_importance_sepal_length': float(importances[0]),
            'feature_importance_sepal_width': float(importances[1]),
            'feature_importance_petal_length': float(importances[2]),
            'feature_importance_petal_width': float(importances[3]),
        }
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate model on test set."""
        X_scaled = self.scaler.transform(X)
        accuracy = self.model.score(X_scaled, y)
        return {'accuracy': float(accuracy)}
    
    def predict(self, features: List[Any]) -> Dict[str, Any]:
        """
        Make a prediction.
        
        Args:
            features: List of 4 features [sepal_length, sepal_width, 
                      petal_length, petal_width]
        
        Returns:
            Dictionary with prediction, class name, and probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Reshape and scale
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'class_name': self.CLASSES[prediction],
            'probability': float(probabilities[prediction]),
            'probabilities': {
                self.CLASSES[i]: float(p) 
                for i, p in enumerate(probabilities)
            },
        }
    
    def save(self, path: str) -> None:
        """Save model and scaler to path."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(path / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model and scaler from path."""
        path = Path(path)
        
        # Load model
        with open(path / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        info = {
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'framework': 'sklearn',
            'algorithm': 'RandomForestClassifier',
            'classes': self.CLASSES,
        }
        
        if self.model is not None:
            info.update({
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'n_features': self.model.n_features_in_,
            })
        
        return info
