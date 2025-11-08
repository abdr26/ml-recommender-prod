"""
Global configuration for the recommendation pipeline.
"""

import os
from pathlib import Path

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths configurable via environment variables 
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

# Ensure required directories exist 
for path in [DATA_DIR, MODEL_REGISTRY_DIR, REQUESTS_PATH, RESPONSES_PATH, WATCH_PATH, RATE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Optional Debug Print (can help in CI logs) 
if os.getenv("DEBUG_CONFIG", "false").lower() == "true":
    print(f"[Config] BASE_DIR={BASE_DIR}")
    print(f"[Config] DATA_DIR={DATA_DIR}")
    print(f"[Config] MODEL_REGISTRY_DIR={MODEL_REGISTRY_DIR}")
    print(f"[Config] MOVIELENS_PATH={MOVIELENS_PATH}")
