"""
MLflow utilities for experiment tracking.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def setup_mlflow():
    """Set up MLflow tracking."""
    # Set MLflow tracking URI if not already set
    if not mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    return mlflow.get_tracking_uri()


def log_model(
    model: BaseEstimator,
    model_name: str,
    params: Dict[str, Union[str, float, int]],
    metrics: Dict[str, float],
    artifact_path: Optional[str] = None
) -> str:
    """
    Log a model to MLflow with parameters and metrics.
    
    Args:
        model: The trained scikit-learn model
        model_name: Name of the model
        params: Dictionary of model parameters
        metrics: Dictionary of model metrics
        artifact_path: Path to save model artifacts
    
    Returns:
        str: The run ID of the MLflow run
    """
    # Set up MLflow
    setup_mlflow()
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        if artifact_path:
            # Save model to disk
            os.makedirs(artifact_path, exist_ok=True)
            model_path = os.path.join(artifact_path, f"{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Log model artifact
            mlflow.log_artifact(model_path, "model")
        else:
            # Log model directly to MLflow
            mlflow.sklearn.log_model(model, "model")
        
        return run.info.run_id


def get_best_run(experiment_name: str, metric_name: str, mode: str = "max") -> Tuple[str, dict, dict]:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric_name: Name of the metric to optimize
        mode: 'max' or 'min' depending on whether higher or lower metric is better
    
    Returns:
        Tuple containing run_id, parameters dict, and metrics dict
    """
    # Get experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    # Get runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    
    # Find best run based on metric
    if mode == "max":
        best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmax()]
    elif mode == "min":
        best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmin()]
    else:
        raise ValueError("Mode must be 'max' or 'min'")
    
    # Extract run_id, params, and metrics
    run_id = best_run["run_id"]
    
    # Get all params and metrics from the run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    return run_id, run.data.params, run.data.metrics


if __name__ == "__main__":
    setup_mlflow()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")