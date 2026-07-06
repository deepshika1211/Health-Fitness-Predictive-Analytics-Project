"""
=========================================================
HealthSense AI
Data Preprocessing

Author  : Deepshika
Version : 1.0
=========================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import logger


def preprocess_data(df):
    """
    Preprocess the dataset.
    """

    logger.info("Starting data preprocessing...")

    # Drop unnecessary columns
    df = df.drop(columns=["Timestamp", "Email address"], errors="ignore")

    # Rename columns
    df.columns = [col.strip().split("?")[0] for col in df.columns]

    # Fill missing values
    df.ffill(inplace=True)

    # Label Encoding
    encoder = LabelEncoder()

    for column in df.columns:
        df[column] = encoder.fit_transform(df[column].astype(str))

    # Feature Scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    logger.info("Data preprocessing completed successfully.")

    return df, df_scaled, encoder, scaler