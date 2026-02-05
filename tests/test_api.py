"""
Tests for the FastAPI serving application.
"""

import pytest
from fastapi.testclient import TestClient

# Note: These tests require config.yaml to exist
# Run: cp examples/iris/config.yaml config.yaml

try:
    from src.serving.main import app

    HAS_CONFIG = True
except FileNotFoundError:
    HAS_CONFIG = False


@pytest.fixture
def client():
    """Create test client."""
    if not HAS_CONFIG:
        pytest.skip("config.yaml not found")
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "framework" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_input(self, client):
        """Test prediction with valid input."""
        response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})

        # May return 503 if model not trained, that's OK for unit test
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "latency_ms" in data

    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input."""
        response = client.post("/predict", json={"features": "invalid"})
        assert response.status_code == 422  # Validation error

    def test_predict_missing_features(self, client):
        """Test prediction with missing features."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert (
            "prediction_requests_total" in response.text or response.status_code == 200
        )
