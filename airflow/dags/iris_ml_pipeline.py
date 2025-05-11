"""
Airflow DAG for Iris data processing and model training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Add project directory to path
sys.path.append("/home/huzaifa/Downloads/mlops-project")

# Import project modules
from src.data.process_data import main as process_data
from src.models.train_model import main as train_model

# Define default arguments
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    "iris_ml_pipeline",
    default_args=default_args,
    description="A pipeline for Iris data processing and model training",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 5, 11),
    catchup=False,
    tags=["mlops", "iris", "model"],
)

# Task 1: Process data
process_data_task = PythonOperator(
    task_id="process_data",
    python_callable=process_data,
    dag=dag,
)

# Task 2: Train model
train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

# Task 3: Run model evaluation
evaluate_model_task = BashOperator(
    task_id="evaluate_model",
    bash_command="python -m pytest /home/huzaifa/Downloads/mlops-project/src/models/test_model.py -v",
    dag=dag,
)

# Define task dependencies
process_data_task >> train_model_task >> evaluate_model_task
