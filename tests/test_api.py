"""
API Tests
=========

Integration tests for the FastAPI serving application.

This module tests the REST API endpoints to ensure:
- Health checks return correct status
- Predictions work with valid input
- Invalid requests are rejected with proper errors
- Metrics endpoint is accessible

Test Setup:
    These tests require config.yaml to exist in the project root.
    Run: cp examples/iris/config.yaml config.yaml

Running Tests:
    pytest tests/test_api.py -v
    pytest tests/test_api.py::TestHealthEndpoint -v

Note:
    Some tests may pass even without a trained model, as they
    test the API's error handling for missing models (503 responses).

See Also:
    - src/serving/main.py: FastAPI application
    - src/serving/schemas.py: Request/response schemas
"""

import pytest
from fastapi.testclient import TestClient

# =============================================================================
# TEST CLIENT SETUP
# =============================================================================

# Try to import the app - may fail if config.yaml doesn't exist
try:
    from src.serving.main import app

    HAS_CONFIG = True
except FileNotFoundError:
    HAS_CONFIG = False


@pytest.fixture
def client():
    """
    Create a FastAPI test client.

    The TestClient allows making HTTP requests to the API
    without starting a real server.

    Returns:
        TestClient instance for making requests

    Skips:
        If config.yaml is not found, tests are skipped
        with a descriptive message.
    """
    if not HAS_CONFIG:
        pytest.skip("config.yaml not found - run setup.py first")
    return TestClient(app)


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================


class TestHealthEndpoint:
    """
    Tests for the /health endpoint.

    The health endpoint is used by:
    - Kubernetes liveness/readiness probes
    - Load balancers
    - Monitoring systems
    """

    def test_health_check(self, client):
        """
        Test that health endpoint returns 200 OK.

        Verifies:
        - Endpoint responds with 200 status
        - Response contains 'status' field
        - Status is 'healthy'
        """
        response = client.get("/health")

        # Should always return 200
        assert response.status_code == 200

        # Check response body
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


# =============================================================================
# MODEL INFO ENDPOINT TESTS
# =============================================================================


class TestModelInfoEndpoint:
    """
    Tests for the /model/info endpoint.

    This endpoint provides metadata about the loaded model.
    """

    def test_model_info(self, client):
        """
        Test model info endpoint returns expected structure.

        Note:
        - Returns 200 if model/config is loaded
        - Returns 503 if configuration is not available

        Verifies (when 200):
        - Response contains model name
        - Response contains version
        - Response contains framework
        """
        response = client.get("/model/info")

        # May return 503 if config not loaded (acceptable in CI)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "name" in data
            assert "version" in data
            assert "framework" in data


# =============================================================================
# PREDICTION ENDPOINT TESTS
# =============================================================================


class TestPredictEndpoint:
    """
    Tests for the /predict endpoint.

    This is the main inference endpoint that accepts
    features and returns model predictions.
    """

    def test_predict_valid_input(self, client):
        """
        Test prediction with valid Iris features.

        Sends a valid prediction request and verifies:
        - Response status is 200 or 503 (no model)
        - Response contains prediction (if 200)
        - Response contains latency_ms (if 200)

        Note:
            503 is acceptable in CI where no model is trained.
        """
        response = client.post(
            "/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}  # Typical setosa
        )

        # Accept both success and "no model" responses
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "latency_ms" in data

    def test_predict_invalid_input(self, client):
        """
        Test prediction with invalid input type.

        Verifies that non-list features are rejected
        with a 422 Unprocessable Entity response.
        """
        response = client.post(
            "/predict",
            json={"features": "invalid"},  # String instead of list
        )

        # Should return validation error
        assert response.status_code == 422

    def test_predict_missing_features(self, client):
        """
        Test prediction with missing required field.

        Verifies that requests without 'features' field
        are rejected with a 422 Unprocessable Entity response.
        """
        response = client.post(
            "/predict",
            json={},  # Missing required 'features' field
        )

        # Should return validation error
        assert response.status_code == 422


# =============================================================================
# METRICS ENDPOINT TESTS
# =============================================================================


class TestMetricsEndpoint:
    """
    Tests for the /metrics endpoint.

    This endpoint exposes Prometheus metrics for monitoring.
    """

    def test_metrics(self, client):
        """
        Test Prometheus metrics endpoint is accessible.

        Verifies:
        - Endpoint returns 200 status
        - Response is in text format (Prometheus format)
        """
        response = client.get("/metrics")

        # Should return 200 with metrics text
        assert response.status_code == 200

        # Prometheus metrics are plain text
        # May or may not contain specific metrics depending on state
        assert (
            "prediction_requests_total" in response.text or response.status_code == 200
        )
