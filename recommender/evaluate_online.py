"""
Reads Kafka snapshots (Parquet files) for reco_requests, reco_responses, and watch events,
then computes a simple KPI:
    % of users who watched at least one recommended movie within N minutes.
Saves results to model_registry/v0.3/online_metrics.csv
"""

import os
import pandas as pd
from pathlib import Path

# CONFIG 
# Time window (in minutes) within which a watch counts as engagement
WINDOW_MINUTES = int(os.getenv("KPI_WINDOW_MINUTES", 30))

# Default data directories
BASE_DIR = Path(os.getenv("DATA_DIR", "data"))
REQUESTS_PATH = BASE_DIR / "reco_requests"
RESPONSES_PATH = BASE_DIR / "reco_responses"
WATCH_PATH = BASE_DIR / "watch"

OUTPUT_DIR = Path(os.getenv("MODEL_REGISTRY_DIR", "model_registry/v0.3"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =====================


def load_latest_parquet(folder: Path):
    """Load the latest parquet snapshot file from a folder."""
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return pd.DataFrame()
    files = sorted(folder.glob("*.parquet"), key=os.path.getmtime, reverse=True)
    if not files:
        print(f"No parquet files found in {folder}")
        return pd.DataFrame()
    latest = files[0]
    print(f" Loading: {latest.name}")
    return pd.read_parquet(latest)


def compute_online_kpi(requests_df, responses_df, watch_df):
    """Compute engagement KPI (% users who watched any recommended movie)."""
    if responses_df.empty or watch_df.empty:
        print("Missing responses or watch data.")
        return pd.DataFrame()

    # Prepare data 
    # Keep only essential columns
    responses_df = responses_df[["user_id", "movie_id", "timestamp"]].copy()
    watch_df = watch_df[["user_id", "movie_id", "timestamp"]].copy()

    # Convert timestamps to datetime (handles both string and numeric)
    for df in [responses_df, watch_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with missing timestamps
    responses_df = responses_df.dropna(subset=["timestamp"])
    watch_df = watch_df.dropna(subset=["timestamp"])

    # Merge recommendations with watch events
    merged = pd.merge(responses_df, watch_df, on=["user_id", "movie_id"], suffixes=("_reco", "_watch"))

    # Compute time difference (in minutes)
    # Proxy Success Definition:
    # A user is "engaged" if they watch any recommended movie within N minutes
    # after receiving the recommendation (default: 30 minutes).
    merged["delta_min"] = (merged["timestamp_watch"] - merged["timestamp_reco"]).dt.total_seconds() / 60

    # Keep only valid positive deltas within KPI window
    engaged = merged[(merged["delta_min"] >= 0) & (merged["delta_min"] <= WINDOW_MINUTES)]

    # KPI calculation 
    total_users = responses_df["user_id"].nunique()
    engaged_users = engaged["user_id"].nunique()
    engagement_rate = round((engaged_users / total_users) * 100, 2) if total_users > 0 else 0

    print(f"\n Total users: {total_users}")
    print(f" Engaged users: {engaged_users}")
    print(f" Engagement rate (watched within {WINDOW_MINUTES} min): {engagement_rate}%")

    results = pd.DataFrame(
        [{"metric": f"Engagement@{WINDOW_MINUTES}min", "value": engagement_rate}]
    )
    results.to_csv(OUTPUT_DIR / "online_metrics.csv", index=False)
    print(f" Online metrics saved to: {OUTPUT_DIR / 'online_metrics.csv'}")
    return results


def main():
    print(" Starting online evaluation...")

    requests_df = load_latest_parquet(REQUESTS_PATH)
    responses_df = load_latest_parquet(RESPONSES_PATH)
    watch_df = load_latest_parquet(WATCH_PATH)

    if responses_df.empty or watch_df.empty:
        print("Not enough data for KPI computation.")
        return

    compute_online_kpi(requests_df, responses_df, watch_df)


if __name__ == "__main__":
    main()
