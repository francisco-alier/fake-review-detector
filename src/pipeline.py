import os
from pathlib import Path
import pandas as pd

from src.download_data import download_dataset
from src.preprocessing import preprocess_pipeline


def main():
    raw_data_path = Path("data/raw/fake_reviews.parquet")
    processed_dir = Path("data/processed")
    processed_data_path = processed_dir / "fake_reviews_processed.parquet"

    # Step 1: Ingest Data
    if not raw_data_path.exists():
        print(f"Raw data not found at {raw_data_path}. Downloading...")
        download_dataset()
    else:
        print(f"Raw data found at {raw_data_path}. Skipping download.")

    # Step 2: Load Data
    print("Loading raw data...")
    df = pd.read_parquet(raw_data_path)
    print(f"Loaded {len(df):,} rows.")

    # Step 3: Preprocess
    print("Preprocessing text and extracting features...")
    df_processed = preprocess_pipeline(df, text_col="text")

    # Step 4: Save Processed Data
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path.resolve()}")

    # Display some stats
    print("\n--- Processed Dataset Stats ---")
    print(f"Total reviews: {len(df_processed):,}")
    print("\nClass distribution (0 = Genuine, 1 = Fake):")
    print(df_processed["label"].value_counts())
    
    print("\nFeatures Summary:")
    feature_cols = [
        "char_count",
        "word_count",
        "avg_word_length",
        "cap_ratio",
        "exclamation_ratio",
        "question_ratio",
    ]
    print(df_processed[feature_cols].describe().round(4))

    print("\nSample processed review:")
    sample = df_processed.iloc[0]
    print(f"Original Text: {sample['text'][:150]}...")
    print(f"Cleaned Text:  {sample['clean_text'][:150]}...")
    print(f"Label:         {sample['label']}")
    print(f"Features:      char_count={sample['char_count']}, cap_ratio={sample['cap_ratio']:.4f}")


if __name__ == "__main__":
    main()
