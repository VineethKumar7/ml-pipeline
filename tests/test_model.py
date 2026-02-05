"""
Model Tests
===========

Unit tests for the Iris classifier example model.

This module tests all methods of the ModelWrapper class to ensure:
- Model initializes correctly
- Training produces valid metrics
- Predictions return expected format
- Save/load preserves model state
- Edge cases are handled properly

Test Organization:
    TestModelWrapper: Tests for the ModelWrapper class methods
    TestDataLoader: Tests for the data loading function

Running Tests:
    pytest tests/test_model.py -v
    pytest tests/test_model.py::TestModelWrapper::test_train -v

Fixtures:
    trained_model: Pre-trained model instance for prediction tests

See Also:
    - examples/iris/model.py: Model implementation
    - examples/iris/data.py: Data loader
"""

import pytest

from examples.iris.data import load_data
from examples.iris.model import ModelWrapper

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def trained_model():
    """
    Create and train a model for use in tests.

    This fixture provides a pre-trained model instance along with
    test data, reducing code duplication across tests.

    Returns:
        Tuple of (model, X_test, y_test):
        - model: Trained ModelWrapper instance
        - X_test: Test features for evaluation
        - y_test: Test labels for evaluation

    Note:
        Uses small n_estimators and max_depth for faster tests.
    """
    # Load the Iris dataset
    X_train, y_train, X_test, y_test = load_data()

    # Create and train model with minimal parameters for speed
    model = ModelWrapper({})
    model.train(X_train, y_train, {"n_estimators": 10, "max_depth": 3})

    return model, X_test, y_test


# =============================================================================
# MODEL WRAPPER TESTS
# =============================================================================


class TestModelWrapper:
    """
    Tests for the ModelWrapper class.

    These tests verify that the ModelWrapper implementation
    correctly follows the interface contract defined in src/model.py.
    """

    def test_init(self):
        """
        Test that ModelWrapper initializes correctly.

        Verifies:
        - Model can be created with empty config
        - Model starts with model=None (not trained)
        - Scaler is initialized
        """
        model = ModelWrapper({})

        # Model should not be trained yet
        assert model.model is None

        # Scaler should be initialized
        assert model.scaler is not None

    def test_train(self):
        """
        Test model training produces valid metrics.

        Verifies:
        - train() returns a metrics dictionary
        - Accuracy is reasonable (> 80% for this easy dataset)
        - Model is not None after training
        """
        # Load data
        X_train, y_train, _, _ = load_data()

        # Create and train model
        model = ModelWrapper({})
        metrics = model.train(
            X_train,
            y_train,
            {
                "n_estimators": 10,
                "max_depth": 3,
            },
        )

        # Verify metrics structure
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.8  # Should be high for Iris

        # Verify model is trained
        assert model.model is not None

    def test_predict(self, trained_model):
        """
        Test model prediction returns correct format.

        Verifies:
        - predict() returns required fields
        - class_name is one of the valid classes
        - probability is in valid range [0, 1]

        Args:
            trained_model: Fixture providing trained model
        """
        model, _, _ = trained_model

        # Make prediction with typical setosa measurements
        result = model.predict([5.1, 3.5, 1.4, 0.2])

        # Verify response structure
        assert "prediction" in result
        assert "class_name" in result
        assert "probability" in result

        # Verify values
        assert result["class_name"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= result["probability"] <= 1

    def test_predict_batch(self, trained_model):
        """
        Test predictions on multiple samples.

        Verifies model achieves reasonable accuracy on test set
        by checking predictions match true labels most of the time.

        Args:
            trained_model: Fixture providing trained model and test data
        """
        model, X_test, y_test = trained_model

        # Test first 10 samples
        correct = 0
        for features, label in zip(X_test[:10], y_test[:10]):
            result = model.predict(features.tolist())
            if result["prediction"] == label:
                correct += 1

        # Should get at least 70% correct
        accuracy = correct / 10
        assert accuracy >= 0.7

    def test_evaluate(self, trained_model):
        """
        Test model evaluation on held-out data.

        Verifies:
        - evaluate() returns metrics dictionary
        - Test accuracy is reasonable

        Args:
            trained_model: Fixture providing trained model and test data
        """
        model, X_test, y_test = trained_model

        # Evaluate on test set
        metrics = model.evaluate(X_test, y_test)

        # Verify structure and values
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.7

    def test_save_load(self, trained_model, tmp_path):
        """
        Test model serialization preserves state.

        Verifies:
        - Model can be saved to disk
        - Model can be loaded from disk
        - Loaded model produces same predictions

        Args:
            trained_model: Fixture providing trained model
            tmp_path: Pytest fixture for temporary directory
        """
        model, _, _ = trained_model

        # Save model
        save_path = tmp_path / "model"
        model.save(str(save_path))

        # Create new model instance and load
        new_model = ModelWrapper({})
        new_model.load(str(save_path))

        # Compare predictions
        test_features = [5.1, 3.5, 1.4, 0.2]
        original_result = model.predict(test_features)
        loaded_result = new_model.predict(test_features)

        # Predictions should match exactly
        assert original_result["prediction"] == loaded_result["prediction"]

    def test_get_model_info(self, trained_model):
        """
        Test model info retrieval.

        Verifies:
        - get_model_info() returns required fields
        - Framework is correctly identified

        Args:
            trained_model: Fixture providing trained model
        """
        model, _, _ = trained_model

        # Get model info
        info = model.get_model_info()

        # Verify structure
        assert "class" in info
        assert "framework" in info
        assert info["framework"] == "sklearn"


# =============================================================================
# DATA LOADER TESTS
# =============================================================================


class TestDataLoader:
    """
    Tests for the data loading function.

    These tests verify that load_data() returns correctly
    formatted data for training.
    """

    def test_load_data(self):
        """
        Test basic data loading.

        Verifies:
        - Returns four arrays (X_train, y_train, X_test, y_test)
        - Features have correct shape (4 features)
        - Labels match feature counts
        - Total samples equals Iris dataset size (150)
        """
        X_train, y_train, X_test, y_test = load_data()

        # Check feature dimensions (4 features in Iris)
        assert X_train.shape[1] == 4

        # Check label/feature alignment
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check total samples (Iris has 150 samples)
        total = len(X_train) + len(X_test)
        assert total == 150

    def test_load_data_split_ratio(self):
        """
        Test custom train/test split ratio.

        Verifies that the test_size parameter correctly
        controls the split proportions.
        """
        # Load with 30% test size
        X_train, y_train, X_test, y_test = load_data(test_size=0.3)

        # Calculate actual test ratio
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total

        # Should be approximately 30% (within 5% tolerance)
        assert abs(test_ratio - 0.3) < 0.05
