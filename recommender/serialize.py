from recommender.config import MODEL_REGISTRY_DIR
import pandas as pd

def save_metrics(df: pd.DataFrame, name: str):
    """Save metrics or model outputs to registry."""
    out = MODEL_REGISTRY_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f" Saved {name} → {out}")
