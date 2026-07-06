"""
=========================================================
HealthSense AI
Prediction Module

Author  : Deepshika
Version : 1.0
=========================================================
"""

import joblib

from config import MODEL_FILE
from src.utils.logger import logger


def load_model():
    """
    Load the trained model.
    """

    logger.info("Loading trained model...")

    model = joblib.load(MODEL_FILE)

    logger.info("Model loaded successfully.")

    return model


def predict(model, input_data):
    """
    Make prediction using the trained model.
    """

    prediction = model.predict(input_data)

    return prediction