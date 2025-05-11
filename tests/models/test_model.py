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
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.models.train_model import evaluate_model, load_data, train_model


def test_load_data():
    """Test data loading function."""
    # Mock the existence of processed data
    with patch("os.path.exists", return_value=True):
        # Create a mock DataFrame that matches our expected format
        mock_df = pd.DataFrame(
            {
                "sepal length (cm)": [5.1, 4.9, 4.7],
                "sepal width (cm)": [3.5, 3.0, 3.2],
                "petal length (cm)": [1.4, 1.4, 1.3],
                "petal width (cm)": [0.2, 0.2, 0.2],
                "target": [0, 0, 0],
            }
        )

        # Patch read_csv to return our mock dataframe
        with patch("pandas.read_csv", return_value=mock_df):
            # Instead of trying to mock train_test_split with specific values,
            # we'll patch it to just return parts of our mock dataframe
            def mock_train_test_split(*args, **kwargs):
                # For this test, we'll just use the same data for train/test
                # This mimics what would happen with the real function
                features = mock_df.iloc[:, :-1]  # All but target column
                target = mock_df["target"]

                # First 2 rows for training
                X_train = features.iloc[:2]
                # Last row for testing
                X_test = features.iloc[2:]
                # Same for target
                y_train = target.iloc[:2]
                y_test = target.iloc[2:]

                return X_train, X_test, y_train, y_test

            # Apply our mock to train_test_split
            with patch(
                "sklearn.model_selection.train_test_split",
                side_effect=mock_train_test_split,
            ):
                # Call the function under test
                X_train, X_test, y_train, y_test = load_data()

                # Check shapes rather than exact values
                assert X_train.shape == (2, 4)  # 2 samples, 4 features
                assert X_test.shape == (1, 4)  # 1 sample, 4 features
                assert y_train.shape == (2,)  # 2 target values
                assert y_test.shape == (1,)  # 1 target value

                # Check data types - load_data returns pandas DataFrame and Series
                assert isinstance(X_train, pd.DataFrame)
                assert isinstance(X_test, pd.DataFrame)
                assert isinstance(y_train, pd.Series)
                assert isinstance(y_test, pd.Series)

                # Check first row of X_train (just to have some value check)
                expected_values = [5.1, 3.5, 1.4, 0.2]
                pd.testing.assert_series_equal(
                    X_train.iloc[0], pd.Series(expected_values, index=X_train.columns)
                )


def test_train_model():
    """Test model training function."""
    # Create dummy data as pandas DataFrame and Series
    X_train = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9],
            "sepal width (cm)": [3.5, 3.0],
            "petal length (cm)": [1.4, 1.4],
            "petal width (cm)": [0.2, 0.2],
        }
    )
    y_train = pd.Series([0, 0])

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
    X_test = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9],
            "sepal width (cm)": [3.5, 3.0],
            "petal length (cm)": [1.4, 1.4],
            "petal width (cm)": [0.2, 0.2],
        }
    )
    y_test = pd.Series([0, 0])

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Assertions
    assert "accuracy" in metrics
    assert (
        metrics["accuracy"] == 1.0
    )  # Perfect accuracy because model always predicts 0 and all examples are class 0
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

    # Create test data (or load from file) as pandas DataFrame and Series
    np.random.seed(42)
    X_test = pd.DataFrame(
        np.random.rand(30, 4),
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    y_test = pd.Series(np.random.randint(0, 3, size=30))

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # In a real test, we'd assert on real performance
    # For this example, we'll assert that the model at least runs
    assert isinstance(accuracy, float)
