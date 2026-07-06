"""
=========================================================
HealthSense AI
Data Validation

Author  : Deepshika
Version : 1.0
=========================================================
"""

from src.utils.logger import logger


def validate_data(df):
    """
    Validate the dataset.
    """

    logger.info("Starting data validation...")

    # Check if dataset is empty
    if df.empty:
        logger.error("Dataset is empty.")
        raise ValueError("Dataset is empty.")

    # Check missing values
    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        logger.warning("Missing values found in dataset.")
        logger.warning(f"\n{missing_values}")

    else:
        logger.info("No missing values found.")

    # Check duplicate rows
    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        logger.warning(f"{duplicate_rows} duplicate rows found.")

    else:
        logger.info("No duplicate rows found.")

    logger.info("Data validation completed successfully.")