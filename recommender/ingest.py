import time
from pathlib import Path
from recommender.config import DATA_DIR

def ingest_kafka_snapshots(rate_limit_per_sec=2):
    """Simulate ingestion with backpressure control."""
    print(f" Ingesting Kafka snapshots into {DATA_DIR} (max {rate_limit_per_sec}/sec)...")
    Path(DATA_DIR).mkdir(exist_ok=True)

    # Simulate backpressure: limit event processing rate
    for i in range(5):
        print(f"   → Processing batch {i+1}/5 ...")
        time.sleep(1 / rate_limit_per_sec)
    print(" Ingestion simulation complete.")
