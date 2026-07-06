"""
=========================================================
HealthSense AI
Data Loader

Author  : Deepshika
Version : 1.0
=========================================================
"""

import pandas as pd

from config import DATASET_PATH
from src.utils.logger import logger


def load_data():
    """
    Load dataset from the raw data folder.
    """

    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info("Dataset loaded successfully.")
        return df

    except FileNotFoundError:
        logger.error(f"Dataset not found: {DATASET_PATH}")
        raise

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise