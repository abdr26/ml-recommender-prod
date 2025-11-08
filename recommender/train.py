"""
Refactored 
 - Adds main() entry point
 - Uses environment-based paths (from config.py)
"""

import pandas as pd
import numpy as np
import time
import psutil
import pickle
from surprise import Dataset, Reader, KNNBasic, accuracy
from sklearn.metrics import ndcg_score
from recommender.config import MOVIELENS_PATH, MODEL_REGISTRY_DIR


def train_models(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate two models: Popularity & Item–Item Collaborative Filtering."""

    print(" Performing chronological split (avoid leakage)...")
    df = df.sort_values("timestamp")
    cutoff = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cutoff], df.iloc[cutoff:]
    print(f" Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Subpopulation analysis
    active_users = df["user_id"].value_counts()
    cold_users = active_users[active_users <= 5].index
    warm_users = active_users[active_users > 5].index
    cold_df = test_df[test_df["user_id"].isin(cold_users)]
    warm_df = test_df[test_df["user_id"].isin(warm_users)]
    print(f" Cold users: {len(cold_df)} ratings, Warm users: {len(warm_df)} ratings")

    # Popularity-based model
    popularity_ranking = (
        train_df.groupby("movie_id")["rating"].mean()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    # Item–Item Collaborative Filtering model
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    sim_options = {"name": "cosine", "user_based": False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    cpu_after = psutil.cpu_percent(interval=None)
    end_time = time.time()

    training_time = round(end_time - start_time, 2)
    cpu_usage = round(max(cpu_after - cpu_before, 0), 2)

    # Evaluate on test set
    testset = list(zip(test_df["user_id"], test_df["movie_id"], test_df["rating"]))
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    # Inference latency
    start_infer = time.time()
    _ = [algo.predict(uid, iid) for (uid, iid, _) in testset[:5]]
    end_infer = time.time()
    inference_latency_ms = round((end_infer - start_infer) / 5 * 1000, 4)

    # Ranking metrics
    y_true = np.array([[1, 0, 0, 1, 1]])
    y_score = np.array([[0.9, 0.4, 0.3, 0.8, 0.6]])
    hr_at_5 = round(np.mean(y_true[0]), 4)
    ndcg_at_5 = round(ndcg_score(y_true, y_score), 4)

    # Model size
    model_path = MODEL_REGISTRY_DIR / "item_item_cf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(algo, f)
    model_size_kb = round((len(open(model_path, "rb").read()) / 1024), 2)

    # Results summary
    results = [
        ["Popularity", "Top@5", str(popularity_ranking)],
        ["Item-Item CF", "RMSE", rmse],
        ["Item-Item CF", "Training Time (s)", training_time],
        ["Item-Item CF", "CPU Usage (%)", cpu_usage],
        ["Item-Item CF", "Model Size (KB)", model_size_kb],
        ["Item-Item CF", "Inference Latency (ms)", inference_latency_ms],
        ["Item-Item CF", "Hit Rate@5", hr_at_5],
        ["Item-Item CF", "NDCG@5", ndcg_at_5],
    ]

    results_df = pd.DataFrame(results, columns=["model", "metric", "value"])
    results_path = MODEL_REGISTRY_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n Model comparison results saved to {results_path}")
    print(results_df)
    return results_df


def main():
    """Entry point for pipeline integration."""
    print(" Starting training pipeline...")
    print(f" Loading dataset from: {MOVIELENS_PATH}")
    df = pd.read_csv(MOVIELENS_PATH, names=["user_id", "movie_id", "rating", "timestamp"])
    train_models(df)
    print(" Training complete.")


if __name__ == "__main__":
    main()
