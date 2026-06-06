from pathlib import Path

from src.merge_restaurant_dataset import build_restaurant_dataset


def main():
    processed_data_path = Path("data/processed/fake_reviews_processed.parquet")

    # Step 1: Ingest, Merge, and Preprocess Restaurant Data
    if not processed_data_path.exists():
        print(f"Processed restaurant dataset not found at {processed_data_path}. Running builder...")
        build_restaurant_dataset()
    else:
        print(f"Processed restaurant dataset found at {processed_data_path}. Skipping ingestion.")

    print("\nPipeline run complete. Ready for model training!")


if __name__ == "__main__":
    main()
