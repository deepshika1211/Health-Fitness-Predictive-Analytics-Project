"""
=========================================================
HealthSense AI
Model Evaluation

Author  : Deepshika
Version : 1.0
=========================================================
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.utils.logger import logger


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models.
    """

    logger.info("========== Model Evaluation Started ==========")

    results = {}

    for name, model in models.items():

        logger.info(f"Evaluating {name}...")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        results[name] = accuracy

        print("\n" + "=" * 60)
        print(f"{name}")
        print("=" * 60)
        print(f"Accuracy : {accuracy:.4f}")

        print("\nClassification Report")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

    logger.info("========== Model Evaluation Completed ==========")

    return results