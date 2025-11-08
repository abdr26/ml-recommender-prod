"""
Unit tests for the recommender pipeline 
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from recommender.transform import load_and_transform
from recommender.schema_validation import validate_dataset
from recommender.drift_detector import detect_drift


def test_load_dataset():
    """Test that dataset loads properly and contains expected columns."""
    df = load_and_transform()
    assert len(df) > 1000, "Dataset too small!"
    expected_cols = {"user_id", "movie_id", "rating", "timestamp"}
    assert expected_cols.issubset(df.columns), f"Missing expected columns: {expected_cols - set(df.columns)}"


def test_schema_validation():
    """Test that Pandera schema validation passes."""
    df = load_and_transform()
    result = validate_dataset(df)
    assert result is True, "Schema validation failed — dataset does not match expected structure."


def test_drift_detector():
    """Test drift detector returns valid results with both keys present."""
    baseline = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "movie_id": [10, 20, 10, 30, 40]
    })
    current = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "movie_id": [10, 30, 40, 50, 60]
    })

    report = detect_drift(current, baseline, threshold=0.5)
    assert isinstance(report, dict), "Drift report should be a dictionary."
    assert "user_id" in report and "movie_id" in report, "Missing keys in drift report."
