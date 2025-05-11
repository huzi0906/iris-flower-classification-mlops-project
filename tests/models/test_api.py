"""
Tests for the Iris classification API.
"""

import json
import os
import pickle
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.predict_api import app


# Create test client
client = TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def predict(self, X):
            # Always predict class 0 (setosa) for testing
            return [0]
        
        def predict_proba(self, X):
            # Return fake probabilities
            return [[0.8, 0.1, 0.1]]
    
    with patch("src.models.predict_api.model", MockModel()):
        yield


def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Test the health endpoint."""
    with patch("src.models.predict_api.model", "mock_model"):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_predict_post(mock_model):
    """Test the POST predict endpoint."""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post(
        "/predict",
        json=test_data
    )
    
    # Check response
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] == 0
    assert result["predicted_species"] == "setosa"
    assert "probabilities" in result
    assert result["probabilities"]["setosa"] > 0.7


def test_predict_get(mock_model):
    """Test the GET predict endpoint."""
    response = client.get(
        "/predict",
        params={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    
    # Check response
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] == 0
    assert result["predicted_species"] == "setosa"
    assert "probabilities" in result