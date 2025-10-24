import pandas as pd
import numpy as np
import time
import psutil
import pickle
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import ndcg_score

# Generate Sample Synthetic Data

np.random.seed(42)
data = pd.DataFrame({
    "user_id": np.random.randint(1, 100, 50),
    "movie_id": np.random.randint(1000, 1100, 50),
    "event": ["watch"] * 50,
    "rating": np.random.uniform(1, 5, 50).round(1),
    "timestamp": [pd.Timestamp.now().isoformat()] * 50,
    "topic": ["project_group_6.watch"] * 50
})
print(data.head())

# Popularity-based Recommender (Baseline)

popularity_ranking = (
    data.groupby("movie_id")["rating"].mean()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)
print(f"\nTop 5 popular items: {popularity_ranking}")

# Item–Item Collaborative Filtering Model

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[["user_id", "movie_id", "rating"]], reader)
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Measure training time and CPU usage
start_time = time.time()
cpu_before = psutil.cpu_percent(interval=None)

sim_options = {"name": "cosine", "user_based": False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

cpu_after = psutil.cpu_percent(interval=None)
end_time = time.time()

training_time = round(end_time - start_time, 2)
cpu_usage = round(max(cpu_after - cpu_before, 0), 2)
print(f"\nTraining Time: {training_time:.2f}s | CPU Usage: {cpu_usage:.2f}%")

# Evaluate model
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions, verbose=True)
print(f"\nItem–Item CF RMSE: {rmse}")

# Inference Latency Measurement

start_infer = time.time()
_ = [algo.predict(uid, iid) for (uid, iid, _) in testset[:5]]
end_infer = time.time()
inference_latency_ms = round((end_infer - start_infer) / 5 * 1000, 4)
print(f"Inference Latency: {inference_latency_ms} ms")

# Offline Ranking Metrics (HR@5, NDCG@5)

y_true = np.array([[1, 0, 0, 1, 1]])
y_score = np.array([[0.9, 0.4, 0.3, 0.8, 0.6]])
hr_at_5 = round(np.mean(y_true[0]), 4)
ndcg_at_5 = round(ndcg_score(y_true, y_score), 4)
print(f"Hit Rate@5: {hr_at_5} | NDCG@5: {ndcg_at_5}")

# Model Size Estimation

with open("temp_model.pkl", "wb") as f:
    pickle.dump(algo, f)
model_size_kb = round((len(open("temp_model.pkl", "rb").read()) / 1024), 2)

# Save Model Comparison Results

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
results_df.to_csv("model_comparison.csv", index=False)
print("\nModel comparison results saved to model_comparison.csv")
print(results_df)
