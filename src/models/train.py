"""
=========================================================
HealthSense AI
Model Training

Author  : Deepshika
Version : 1.0
=========================================================
"""

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_FILE
)

from src.utils.logger import logger


def train_model(df):

    logger.info("========== Model Training Started ==========")

    # Features and Target
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE)
    }

    trained_models = {}

    for name, model in models.items():

        logger.info(f"Training {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    # Save Random Forest Model
    joblib.dump(trained_models["Random Forest"], MODEL_FILE)

    logger.info("Model saved successfully.")

    logger.info("========== Model Training Completed ==========")

    return trained_models, X_test, y_test