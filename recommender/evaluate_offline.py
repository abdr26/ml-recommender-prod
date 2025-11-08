"""
Offline Evaluation 
Performs evaluation on MovieLens dataset and saves metrics.
"""

import pandas as pd
from recommender.config import MODEL_REGISTRY_DIR

def evaluate_offline_metrics():
    """Read offline metrics file and summarize numeric values."""
    path = MODEL_REGISTRY_DIR / "model_comparison.csv"
    df = pd.read_csv(path)
    print("\n Offline Evaluation Results:")
    print(df)

    # Extract numeric metrics only
    numeric_df = df[pd.to_numeric(df["value"], errors="coerce").notnull()]
    numeric_df = numeric_df.copy()
    numeric_df["value"] = numeric_df["value"].astype(float)
    summary = numeric_df["value"].describe()
    print("\n Summary of numeric metrics:")
    print(summary)

    # Save summary
    summary.to_csv(MODEL_REGISTRY_DIR / "offline_metrics_summary.csv")
    print(f"\n Offline metrics summary saved to {MODEL_REGISTRY_DIR}/offline_metrics_summary.csv")


def main():
    """Entry point for pipeline integration."""
    print(" Running offline evaluation...")
    evaluate_offline_metrics()


if __name__ == "__main__":
    main()
