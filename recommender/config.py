# recommender/config.py
"""
Global configuration for the recommendation pipeline.
Values can be overridden using environment variables.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.getenv("PROJECT_ROOT", "/home/abdul960/kafka-docker"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODEL_REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_DIR", BASE_DIR / "model_registry/v0.3"))
MOVIELENS_PATH = Path(os.getenv("MOVIELENS_PATH", BASE_DIR / "ml-1m/ratings.csv"))

# Kafka simulation folders
REQUESTS_PATH = DATA_DIR / "reco_requests"
RESPONSES_PATH = DATA_DIR / "reco_responses"
WATCH_PATH = DATA_DIR / "watch"
RATE_PATH = DATA_DIR / "rate"

# Evaluation and training constants
KPI_WINDOW_MINUTES = int(os.getenv("KPI_WINDOW_MINUTES", 30))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))

MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
