"""
MLflow setup and utility functions for experiment tracking
"""

import os
import mlflow
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow tracking settings
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris-classification"


def setup_mlflow():
    """Configure MLflow settings."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    # Set or create experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info(f"Created new experiment: {EXPERIMENT_NAME}")
    else:
        logger.info(f"Using existing experiment: {EXPERIMENT_NAME}")

    mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(run_name=None):
    """Start a new MLflow run with optional name."""
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return mlflow.start_run(run_name=run_name)


def log_parameters(params):
    """Log parameters to MLflow."""
    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} parameters to MLflow")


def log_metrics(metrics):
    """Log metrics to MLflow."""
    mlflow.log_metrics(metrics)
    logger.info(f"Logged {len(metrics)} metrics to MLflow")


def log_model(model, model_name="model"):
    """Log model to MLflow."""
    mlflow.sklearn.log_model(model, model_name)
    logger.info(f"Logged model as '{model_name}' to MLflow")


def log_artifact(filepath):
    """Log artifact to MLflow."""
    mlflow.log_artifact(filepath)
    logger.info(f"Logged artifact: {filepath}")


def end_run():
    """End the current MLflow run."""
    mlflow.end_run()
    logger.info("MLflow run completed")


def get_best_run(metric_name="accuracy", ascending=False):
    """Get the best run based on a metric."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        logger.warning(f"Experiment {EXPERIMENT_NAME} not found")
        return None

    # Get all runs for the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
    )

    if not runs:
        logger.warning(f"No runs found for experiment {EXPERIMENT_NAME}")
        return None

    # Return the best run
    best_run = runs[0]
    logger.info(
        f"Found best run (ID: {best_run.info.run_id}) with {metric_name} = {best_run.data.metrics.get(metric_name, 'N/A')}"
    )

    return best_run


def load_model_from_run(run_id, model_name="model"):
    """Load model from a specific MLflow run."""
    model_uri = f"runs:/{run_id}/{model_name}"
    logger.info(f"Loading model from {model_uri}")

    return mlflow.sklearn.load_model(model_uri)
