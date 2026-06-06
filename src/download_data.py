import os
import sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def download_dataset():
    print("Initializing dataset download from Hugging Face...")
    try:
        # Load dataset from Hugging Face
        # theArijitDas/Fake-Reviews-Dataset has a default 'train' split
        dataset = load_dataset("theArijitDas/Fake-Reviews-Dataset", split="train")

        print(f"Dataset loaded successfully! Number of rows: {len(dataset):,}")

        # Convert to Pandas DataFrame
        df = pd.DataFrame(dataset)

        # Let's inspect the columns
        print("Dataset columns:", df.columns.tolist())
        print("Class distribution:\n", df["label"].value_counts(normalize=True))

        # Rename columns to standard snake_case if necessary
        # We'll rename 'text_' to 'text' for cleaner downstream code
        if "text_" in df.columns:
            df = df.rename(columns={"text_": "text"})
            print("Renamed column 'text_' to 'text'")

        # Save to Parquet format in data/raw/
        raw_data_dir = Path("data/raw")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = raw_data_dir / "fake_reviews.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Dataset successfully saved to: {output_path.resolve()}")
        
    except Exception as e:
        print(f"Error downloading or saving dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
