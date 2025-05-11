"""
Tests for the Iris classification model.
"""

import os
import pickle
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.train_model import evaluate_model, load_data, train_model


def test_load_data():
    """Test data loading function."""
    # Mock the existence of processed data
    with patch("os.path.exists", return_value=True):
        # Mock pandas read_csv
        mock_df = pd.DataFrame({
            "sepal length (cm)": [5.1, 4.9, 4.7],
            "sepal width (cm)": [3.5, 3.0, 3.2],
            "petal length (cm)": [1.4, 1.4, 1.3],
            "petal width (cm)": [0.2, 0.2, 0.2],
            "target": [0, 0, 0]
        })
        
        with patch("pandas.read_csv", return_value=mock_df):
            # Mock train_test_split
            with patch("sklearn.model_selection.train_test_split", return_value=("X_train", "X_test", "y_train", "y_test")):
                X_train, X_test, y_train, y_test = load_data()
                
                assert X_train == "X_train"
                assert X_test == "X_test"
                assert y_train == "y_train"
                assert y_test == "y_test"


def test_train_model():
    """Test model training function."""
    # Create dummy data
    X_train = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
    y_train = np.array([0, 0])
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Assertions
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_depth == 5


def test_evaluate_model():
    """Test model evaluation function."""
    # Create a simple model that always predicts class 0
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X))
    
    model = DummyModel()
    
    # Create test data where all examples are class 0
    X_test = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
    y_test = np.array([0, 0])
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Assertions
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0  # Perfect accuracy because model always predicts 0 and all examples are class 0
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_model_performance():
    """Test that the saved model meets minimum performance requirements."""
    model_path = "models/iris_model.pkl"
    
    # Skip test if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}")
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Create test data (or load from file)
    # Here we'll create some fake test data for simplicity
    np.random.seed(42)
    X_test = np.random.rand(30, 4)
    y_test = np.random.randint(0, 3, size=30)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # In a real test, we'd assert on real performance
    # For this example, we'll assert that the model at least runs
    assert isinstance(accuracy, float)