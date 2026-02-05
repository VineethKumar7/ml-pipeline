"""
Tests for the Iris model example.
"""

import pytest
from examples.iris.model import ModelWrapper
from examples.iris.data import load_data


@pytest.fixture
def trained_model():
    """Create and train a model for testing."""
    X_train, y_train, X_test, y_test = load_data()
    model = ModelWrapper({})
    model.train(X_train, y_train, {"n_estimators": 10, "max_depth": 3})
    return model, X_test, y_test


class TestModelWrapper:
    """Tests for ModelWrapper class."""

    def test_init(self):
        """Test model initialization."""
        model = ModelWrapper({})
        assert model.model is None
        assert model.scaler is not None

    def test_train(self):
        """Test model training."""
        X_train, y_train, _, _ = load_data()
        model = ModelWrapper({})

        metrics = model.train(
            X_train,
            y_train,
            {
                "n_estimators": 10,
                "max_depth": 3,
            },
        )

        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.8
        assert model.model is not None

    def test_predict(self, trained_model):
        """Test model prediction."""
        model, _, _ = trained_model

        # Setosa sample
        result = model.predict([5.1, 3.5, 1.4, 0.2])

        assert "prediction" in result
        assert "class_name" in result
        assert "probability" in result
        assert result["class_name"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= result["probability"] <= 1

    def test_predict_batch(self, trained_model):
        """Test predictions on multiple samples."""
        model, X_test, y_test = trained_model

        correct = 0
        for features, label in zip(X_test[:10], y_test[:10]):
            result = model.predict(features.tolist())
            if result["prediction"] == label:
                correct += 1

        accuracy = correct / 10
        assert accuracy >= 0.7  # At least 70% accuracy on test samples

    def test_evaluate(self, trained_model):
        """Test model evaluation."""
        model, X_test, y_test = trained_model

        metrics = model.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.7

    def test_save_load(self, trained_model, tmp_path):
        """Test model save and load."""
        model, _, _ = trained_model

        # Save
        save_path = tmp_path / "model"
        model.save(str(save_path))

        # Load into new model
        new_model = ModelWrapper({})
        new_model.load(str(save_path))

        # Compare predictions
        test_features = [5.1, 3.5, 1.4, 0.2]
        original_result = model.predict(test_features)
        loaded_result = new_model.predict(test_features)

        assert original_result["prediction"] == loaded_result["prediction"]

    def test_get_model_info(self, trained_model):
        """Test model info retrieval."""
        model, _, _ = trained_model

        info = model.get_model_info()

        assert "class" in info
        assert "framework" in info
        assert info["framework"] == "sklearn"


class TestDataLoader:
    """Tests for data loading."""

    def test_load_data(self):
        """Test data loading."""
        X_train, y_train, X_test, y_test = load_data()

        assert X_train.shape[1] == 4  # 4 features
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Check train/test split ratio
        total = len(X_train) + len(X_test)
        assert total == 150  # Iris dataset size

    def test_load_data_split_ratio(self):
        """Test custom split ratio."""
        X_train, y_train, X_test, y_test = load_data(test_size=0.3)

        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total

        assert abs(test_ratio - 0.3) < 0.05  # Within 5%
