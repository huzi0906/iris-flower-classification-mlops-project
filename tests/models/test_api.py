"""
Test suite for the Iris Classification API
"""

import os
import sys
import json
import time
import logging
import requests
import unittest
from multiprocessing import Process

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import API module
from src.models.predict_api import app
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def start_server():
    """Start the API server for testing"""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


class TestIrisAPI(unittest.TestCase):
    """Test class for the Iris Classification API."""

    server_process = None
    base_url = "http://127.0.0.1:8000"

    @classmethod
    def setUpClass(cls):
        """Start the server before tests."""
        logger.info("Starting API server for testing...")
        cls.server_process = Process(target=start_server)
        cls.server_process.start()
        # Allow time for server to start
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        """Stop the server after tests."""
        if cls.server_process:
            logger.info("Stopping API server...")
            cls.server_process.terminate()
            cls.server_process.join()

    def test_root_endpoint(self):
        """Test that the root endpoint returns correct information."""
        response = requests.get(self.base_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("endpoints", data)
        self.assertIn("POST /predict", data["endpoints"])
        self.assertIn("GET /health", data["endpoints"])

    def test_health_endpoint(self):
        """Test that the health endpoint returns healthy status."""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)
        self.assertTrue(data["model_loaded"])

    def test_prediction_endpoint_setosa(self):
        """Test prediction for setosa sample."""
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        response = requests.post(f"{self.base_url}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("species", data)
        self.assertEqual(data["species"], "setosa")
        self.assertIn("species_id", data)
        self.assertEqual(data["species_id"], 0)
        self.assertIn("probability", data)
        self.assertGreaterEqual(data["probability"], 0.9)

    def test_prediction_endpoint_versicolor(self):
        """Test prediction for versicolor sample."""
        sample_data = {
            "sepal_length": 6.0,
            "sepal_width": 2.9,
            "petal_length": 4.5,
            "petal_width": 1.5,
        }

        response = requests.post(f"{self.base_url}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("species", data)
        self.assertEqual(data["species"], "versicolor")
        self.assertIn("species_id", data)
        self.assertEqual(data["species_id"], 1)

    def test_prediction_endpoint_virginica(self):
        """Test prediction for virginica sample."""
        sample_data = {
            "sepal_length": 7.2,
            "sepal_width": 3.2,
            "petal_length": 6.0,
            "petal_width": 1.8,
        }

        response = requests.post(f"{self.base_url}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("species", data)
        self.assertEqual(data["species"], "virginica")
        self.assertIn("species_id", data)
        self.assertEqual(data["species_id"], 2)

    def test_invalid_input(self):
        """Test API response to invalid input."""
        # Negative value not allowed
        sample_data = {
            "sepal_length": -1.0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        response = requests.post(f"{self.base_url}/predict", json=sample_data)

        self.assertEqual(response.status_code, 422)

        # Missing field
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            # Missing petal_width
        }

        response = requests.post(f"{self.base_url}/predict", json=sample_data)

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
