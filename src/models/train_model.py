"""
Model Training Script for Iris Classification
This script trains a machine learning model on the processed Iris dataset and tracks experiments with MLflow.
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    """Load the processed training and test data."""
    logger.info("Loading processed data")

    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Prepare features and target
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    logger.info(f"Loaded training data with {X_train.shape[0]} samples")
    logger.info(f"Loaded test data with {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, params=None):
    """Train a RandomForest classifier."""
    logger.info("Training RandomForest classifier")

    # Default parameters
    if params is None:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )

    model.fit(X_train, y_train)
    logger.info("Model training completed")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    logger.info("Evaluating model")

    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def save_model(model, metrics):
    """Save the model to disk."""
    os.makedirs("models", exist_ok=True)

    model_path = "models/iris_classifier.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path


def main():
    """Main function to train and evaluate the model with MLflow tracking."""
    X_train, X_test, y_train, y_test = load_data()

    # Set MLflow experiment
    mlflow.set_experiment("iris-classification")

    # Parameters to try
    param_sets = [
        {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        {"n_estimators": 200, "max_depth": 15, "random_state": 42},
        {"n_estimators": 50, "max_depth": 5, "random_state": 42},
    ]

    best_model = None
    best_accuracy = 0
    best_metrics = None
    best_params = None

    for params in param_sets:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train and evaluate model
            model = train_model(X_train, y_train, params)
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Keep track of the best model
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_model = model
                best_metrics = metrics
                best_params = params

    logger.info(f"Best model parameters: {best_params}")
    logger.info(f"Best model accuracy: {best_accuracy:.4f}")

    # Save the best model
    if best_model:
        model_path = save_model(best_model, best_metrics)

        with mlflow.start_run(run_name="best-model"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(best_metrics)
            mlflow.sklearn.log_model(best_model, "best-model")


if __name__ == "__main__":
    main()
