from recommender.config import MOVIELENS_PATH
import pandas as pd

def load_and_transform():
    """Load MovieLens data and prepare for training."""
    print(f" Loading dataset from {MOVIELENS_PATH}")

    # Explicit column names
    col_names = ["user_id", "movie_id", "rating", "timestamp"]
    df = pd.read_csv(MOVIELENS_PATH, names=col_names)

    df = df.sort_values("timestamp")
    print(f" Loaded {len(df)} records with columns: {list(df.columns)}")
    return df
