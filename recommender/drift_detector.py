"""
Drift detection for user/movie distributions.
Compares new incoming data to baseline MovieLens distributions.
"""

import pandas as pd

def detect_drift(current_df, baseline_df, threshold=0.05):
    """
    Detect drift using user_id and movie_id frequency changes.
    threshold = acceptable relative difference (5% default)
    """
    print(" Running drift detection...")

    drift_report = {}

    for col in ["user_id", "movie_id"]:
        base_counts = baseline_df[col].value_counts(normalize=True)
        current_counts = current_df[col].value_counts(normalize=True)

        # Intersection of common keys (convert to list for pandas)
        common = list(set(base_counts.index).intersection(current_counts.index))

        if not common:
            drift_report[col] = 1.0  # complete drift (no overlap)
            print(f" Complete drift detected in {col} (no overlapping values).")
            continue

        diff = abs(base_counts.loc[common] - current_counts.loc[common]).mean()
        drift_report[col] = float(diff)

        if diff > threshold:
            print(f"Drift detected in {col} distribution: Δ={diff:.3f}")
        else:
            print(f" No significant drift in {col} distribution (Δ={diff:.3f})")

    return drift_report
