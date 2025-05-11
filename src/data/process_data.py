"""
Data Processing Script for Iris Dataset
This script downloads the Iris dataset and prepares it for model training.
"""

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    logger.info("Created raw and processed data directories")


def download_iris_data():
    """Download Iris dataset and save as CSV."""
    logger.info("Downloading Iris dataset")

    # Load iris dataset
    iris = load_iris()

    # Create DataFrame
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add target column
    iris_df["target"] = iris.target
    iris_df["species"] = iris_df["target"].map(
        {0: "setosa", 1: "versicolor", 2: "virginica"}
    )

    # Save raw data
    raw_path = "data/raw/iris.csv"
    iris_df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to {raw_path}")

    return iris_df


def process_data(df):
    """Process the data and create train/test splits."""
    logger.info("Processing data and creating train/test splits")

    # Create features and target
    X = df.drop(["target", "species"], axis=1)
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save processed datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info(f"Training data saved to {train_path}")
    logger.info(f"Test data saved to {test_path}")

    return train_data, test_data


def main():
    """Main function to download and process Iris dataset."""
    create_dirs()
    df = download_iris_data()
    train_data, test_data = process_data(df)

    logger.info("Data processing completed successfully")
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")


if __name__ == "__main__":
    main()
