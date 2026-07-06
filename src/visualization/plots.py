"""
=========================================================
HealthSense AI
Data Visualization

Author  : Deepshika
Version : 1.0
=========================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import logger


def plot_target_distribution(df):
    """
    Plot target variable distribution.
    """

    logger.info("Plotting target distribution...")

    plt.figure(figsize=(8, 5))
    sns.countplot(x=df.iloc[:, 0])

    plt.title("Target Variable Distribution")
    plt.xlabel(df.columns[0])
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap.
    """

    logger.info("Plotting correlation heatmap...")

    plt.figure(figsize=(15, 10))

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def plot_feature_distribution(df):
    """
    Plot first six feature distributions.
    """

    logger.info("Plotting feature distributions...")

    columns = df.columns[:6]

    plt.figure(figsize=(15, 8))

    for i, column in enumerate(columns, start=1):

        plt.subplot(2, 3, i)

        sns.histplot(df[column], kde=True)

        plt.title(column)

    plt.tight_layout()
    plt.show() 