from recommender import ingest, transform, train, evaluate_offline, evaluate_online

def run_pipeline():
    print(" Starting full recommendation pipeline...")

    # Ingest
    ingest.ingest_kafka_snapshots()

    #Transform
    df = transform.load_and_transform()
    from recommender.schema_validation import validate_dataset

    print(" Running schema validation...")
    validate_dataset(df)
    from recommender.drift_detector import detect_drift

    # Example: use MovieLens data 
    baseline_df = df.sample(frac=0.1, random_state=42)
    detect_drift(df, baseline_df)


    # Train
    print(" Starting training...")
    from recommender.train import main as train_main
    train_main()

    #Evaluate Offline
    print(" Running offline evaluation...")
    from recommender.evaluate_offline import main as offline_main
    offline_main()

    # Evaluate Online
    print(" Running online evaluation...")
    from recommender.evaluate_online import main as online_main
    online_main()

    print(" Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
