"""
Test module for the iris classification model
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import project modules
from src.models.train_model import train_model, evaluate_model


class TestIrisModel:
    """Test class for the Iris classification model."""

    @pytest.fixture
    def iris_data(self):
        """Fixture to provide iris data for tests."""
        # Load iris data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        return X, y

    def test_model_training(self, iris_data):
        """Test if the model trains without errors."""
        X, y = iris_data

        # Train the model
        model = train_model(X, y)

        # Check if model was created correctly
        assert model is not None
        assert isinstance(model, RandomForestClassifier)

    def test_model_evaluation(self, iris_data):
        """Test if the model evaluation works correctly."""
        X, y = iris_data

        # Train the model
        model = train_model(X, y)

        # Evaluate the model
        metrics = evaluate_model(model, X, y)

        # Check if metrics are as expected
        assert metrics is not None
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.9  # Expecting high accuracy on training data

    def test_model_prediction(self, iris_data):
        """Test if the model can make predictions."""
        X, y = iris_data

        # Train the model
        model = train_model(X, y)

        # Make a prediction
        sample = X.iloc[[0]]  # Get the first sample
        prediction = model.predict(sample)

        # Check if prediction shape is correct
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1, 2]  # Should be one of the three classes
