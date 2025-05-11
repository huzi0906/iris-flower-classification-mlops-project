"""
Visualization utilities for the Iris dataset.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix


def plot_iris_features():
    """Plot pairwise relationships between Iris features."""
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    # Create pairplot
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    plt.suptitle('Iris Dataset Features', y=1.02)
    plt.savefig('reports/figures/iris_pairplot.png')
    plt.close()
    
    return 'reports/figures/iris_pairplot.png'


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    
    Returns:
        str: Path to the saved confusion matrix plot
    """
    if class_names is None:
        iris = load_iris()
        class_names = iris.target_names
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('reports/figures', exist_ok=True)
    
    # Save the plot
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    
    return 'reports/figures/confusion_matrix.png'


if __name__ == "__main__":
    plot_iris_features()
    print("Visualization complete!")