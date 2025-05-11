"""
Data processing script for Iris dataset.
Downloads and processes the Iris dataset.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def download_data():
    """Download Iris dataset and save as CSV."""
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    # Load Iris dataset
    iris = load_iris()

    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Save to CSV
    df.to_csv("data/raw/iris.csv", index=False)

    return df


def process_data():
    """Process Iris dataset for training."""
    # Check if raw data exists, if not download it
    if not os.path.exists("data/raw/iris.csv"):
        df = download_data()
    else:
        df = pd.read_csv("data/raw/iris.csv")

    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Process data (in this case, just splitting into train/test)
    df_processed = df.copy()

    # Save processed data
    df_processed.to_csv("data/processed/iris_processed.csv", index=False)

    return df_processed


def main():
    """Main function to execute data processing."""
    print("Processing Iris dataset...")
    process_data()
    print("Data processing complete!")


if __name__ == "__main__":
    main()
