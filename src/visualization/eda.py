"""
=========================================================
HealthSense AI
Exploratory Data Analysis (EDA)

Author  : Deepshika
Version : 1.0
=========================================================
"""

from src.utils.logger import logger


def perform_eda(df):
    """
    Perform Exploratory Data Analysis.
    """

    logger.info("========== EDA Started ==========")

    print("\n========== DATASET INFORMATION ==========")
    print(df.info())

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== FIRST FIVE RECORDS ==========")
    print(df.head())

    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE RECORDS ==========")
    print(df.duplicated().sum())

    logger.info("EDA completed successfully.")