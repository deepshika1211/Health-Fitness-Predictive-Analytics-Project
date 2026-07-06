"""
=========================================================
HealthSense AI
Logger Configuration

Author  : Deepshika
Version : 1.0
=========================================================
"""

import logging
from pathlib import Path
from config import LOGS_DIR

# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = LOGS_DIR / "project.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Logger object
logger = logging.getLogger(__name__)