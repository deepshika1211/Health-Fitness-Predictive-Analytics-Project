"""
=========================================================
HealthSense AI
Main Application

Author  : Deepshika
Version : 1.0
=========================================================
"""

from src.data.loader import load_data
from src.data.validation import validate_data
from src.utils.preprocessing import preprocess_data
from src.utils.logger import logger

from src.visualization.eda import perform_eda
from src.visualization.plots import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_feature_distribution
)

from src.models.train import train_model
from src.models.evaluate import evaluate_models


def main():

    logger.info("========== HealthSense AI Started ==========")

    # Load Dataset
    df = load_data()

    # Validate Dataset
    validate_data(df)

    # Preprocess Dataset
    processed_df, scaled_df, encoder, scaler = preprocess_data(df)

    # EDA
    perform_eda(processed_df)

    # Visualizations
    plot_target_distribution(processed_df)
    plot_correlation_heatmap(processed_df)
    plot_feature_distribution(processed_df)

    # Train Models
    trained_models, X_test, y_test = train_model(processed_df)

    # Evaluate Models
    results = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    print("\n")
    print("=" * 60)
    print("MODEL ACCURACY")
    print("=" * 60)

    for model_name, accuracy in results.items():
        print(f"{model_name:<25} : {accuracy:.4f}")

    logger.info("========== HealthSense AI Finished ==========")


if __name__ == "__main__":
    main()