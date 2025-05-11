"""
Model training script for Iris dataset.
"""

import os
import pickle
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.pipeline.mlflow_utils import log_model


def load_data():
    """Load processed data for model training."""
    data_path = "data/processed/iris_processed.csv"
    
    # Check if processed data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run process_data.py first.")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, hyperparameters=None):
    """
    Train a RandomForest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        hyperparameters: Optional dict of hyperparameters
    
    Returns:
        Trained model
    """
    if hyperparameters is None:
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
    
    # Initialize model with hyperparameters
    model = RandomForestClassifier(**hyperparameters)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    
    return metrics


def save_model(model, metrics, output_dir="models"):
    """
    Save model to disk.
    
    Args:
        model: Trained model
        metrics: Model evaluation metrics
        output_dir: Directory to save model
    
    Returns:
        Path to saved model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"iris_model.pkl")
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.txt")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metrics
    with open(metrics_path, "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
    
    return model_path


def main():
    """Main function to train and evaluate model."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print("Training model...")
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }
    model = train_model(X_train, y_train, hyperparameters)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Model metrics: {metrics}")
    
    print("Saving model...")
    model_path = save_model(model, metrics)
    print(f"Model saved to {model_path}")
    
    print("Logging to MLflow...")
    run_id = log_model(
        model=model,
        model_name="iris_classifier",
        params=hyperparameters,
        metrics=metrics,
        artifact_path="models"
    )
    print(f"MLflow run ID: {run_id}")
    
    return model, metrics


if __name__ == "__main__":
    main()