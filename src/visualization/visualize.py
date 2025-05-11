"""
Visualization utilities for the Iris dataset.
This module provides functions to visualize the Iris dataset and model results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report


def load_data(raw=False):
    """Load either raw or processed Iris data."""
    if raw:
        path = "data/raw/iris.csv"
    else:
        path = "data/processed/train.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # If it's the raw data, add species names if not present
    if raw and "species" not in df.columns:
        species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
        if "target" in df.columns:
            df["species"] = df["target"].map(species_mapping)

    return df


def create_feature_distribution_plots():
    """Create plots showing the distribution of features by species."""
    print("Generating feature distribution plots...")

    # Load data
    df = load_data(raw=True)

    # Create output directory
    os.makedirs("docs/images", exist_ok=True)

    # Set plot style
    sns.set(style="whitegrid")

    # Plot histograms for each feature
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(df.columns[:4]):
        plt.subplot(2, 2, i + 1)
        for species in df["species"].unique():
            subset = df[df["species"] == species]
            sns.histplot(subset[feature], kde=True, label=species)
        plt.title(f"Distribution of {feature}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("docs/images/feature_distributions.png")
    print("Feature distribution plot saved to docs/images/feature_distributions.png")

    # Create pair plot
    plt.figure(figsize=(10, 8))
    pair_plot = sns.pairplot(df, hue="species", height=2)
    pair_plot.savefig("docs/images/pair_plot.png")
    print("Pair plot saved to docs/images/pair_plot.png")

    # Create box plots
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(df.columns[:4]):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x="species", y=feature, data=df)
        plt.title(f"Box plot of {feature}")

    plt.tight_layout()
    plt.savefig("docs/images/box_plots.png")
    print("Box plots saved to docs/images/box_plots.png")


def visualize_pca():
    """Create PCA visualization of the dataset."""
    print("Generating PCA visualization...")

    # Load data
    df = load_data(raw=True)

    # Extract features and target
    X = df.iloc[:, :4].values
    y = (
        df["target"].values
        if "target" in df.columns
        else df["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2}).values
    )

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["species"] = (
        df["species"]
        if "species" in df.columns
        else df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    )

    # Plot PCA results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="species",
        data=pca_df,
        palette="viridis",
        s=100,
        alpha=0.8,
    )

    plt.title("PCA of Iris Dataset")
    plt.xlabel(
        f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
    )
    plt.ylabel(
        f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
    )
    plt.tight_layout()

    # Save the plot
    os.makedirs("docs/images", exist_ok=True)
    plt.savefig("docs/images/pca_visualization.png")
    print("PCA visualization saved to docs/images/pca_visualization.png")


def visualize_model_results():
    """Visualize the model performance using confusion matrix and feature importances."""
    print("Generating model results visualization...")

    # Check if model exists
    model_path = "models/iris_classifier.joblib"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load model and test data
    model = joblib.load(model_path)

    # Load test data if available
    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        print(f"Test data file not found: {test_path}")
        return

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Create output directory
    os.makedirs("docs/images", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Setosa", "Versicolor", "Virginica"],
        yticklabels=["Setosa", "Versicolor", "Virginica"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("docs/images/confusion_matrix.png")
    print("Confusion matrix saved to docs/images/confusion_matrix.png")

    # Feature importances
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_test.columns

        plt.title("Feature Importances")
        plt.bar(range(X_test.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_test.shape[1]), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig("docs/images/feature_importances.png")
        print("Feature importances saved to docs/images/feature_importances.png")

    # Classification report
    report = classification_report(y_test, y_pred)
    with open("docs/classification_report.txt", "w") as f:
        f.write(report)
    print("Classification report saved to docs/classification_report.txt")


if __name__ == "__main__":
    create_feature_distribution_plots()
    visualize_pca()
    visualize_model_results()
