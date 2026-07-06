"""
=========================================================
HealthSense AI
Configuration File

Author  : Deepshika
Version : 1.0
=========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# DATA PATHS
# ==========================================================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

DATASET_PATH = RAW_DATA_DIR / "health_dataset.csv"

# ==========================================================
# MODEL PATHS
# ==========================================================

MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
SCALERS_DIR = MODELS_DIR / "scalers"
ENCODERS_DIR = MODELS_DIR / "encoders"

MODEL_FILE = TRAINED_MODELS_DIR / "health_prediction_model.pkl"
SCALER_FILE = SCALERS_DIR / "scaler.pkl"
ENCODER_FILE = ENCODERS_DIR / "label_encoder.pkl"

# ==========================================================
# REPORT PATHS
# ==========================================================

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = REPORTS_DIR / "logs"
PDF_DIR = REPORTS_DIR / "pdf"

# ==========================================================
# ASSET PATHS
# ==========================================================

ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
ICONS_DIR = ASSETS_DIR / "icons"
DASHBOARD_DIR = ASSETS_DIR / "dashboard"

# ==========================================================
# PROJECT SETTINGS
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20