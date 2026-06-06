import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

from src.generate_synthetic_reviews import generate_reviews
from src.preprocessing import preprocess_pipeline


def load_env_file(dotenv_path=".env"):
    """Reads a local .env file and loads keys into os.environ."""
    path = Path(dotenv_path)
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    key, val = stripped.split("=", 1)
                    val = val.strip("\"'")
                    os.environ[key] = val


def build_restaurant_dataset():
    load_env_file()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("--- STEP 1: Ingesting Real Yelp Reviews ---")
    try:
        # Load a slice of real Yelp reviews from HF (test split is smaller/faster to download)
        print("Downloading a slice of real Yelp reviews (5,000 samples)...")
        yelp_dataset = load_dataset("Yelp/yelp_review_full", split="test[:5000]")
        df_real = pd.DataFrame(yelp_dataset)

        # Align columns
        # Yelp labels are 0-4 (mapping to 1-5 stars). Set label = 0 (Genuine)
        df_real_aligned = df_real.assign(
            rating=lambda d: d["label"] + 1.0,
            label=0,
            category="Restaurant",
            origin="Yelp"
        )[["category", "rating", "text", "label", "origin"]]
        print(f"Loaded {len(df_real_aligned):,} real Yelp reviews.")
    except Exception as e:
        print(f"Error downloading Yelp reviews: {e}")
        return

    print("\n--- STEP 2: Ingesting/Generating Synthetic Fake Reviews ---")
    # Generate 500 fake reviews to build a balanced-enough dataset
    synthetic_path = raw_dir / "synthetic_reviews.parquet"
    
    print("Generating 500 synthetic fake restaurant reviews...")
    df_fake = generate_reviews(num_reviews=500)
    df_fake.to_parquet(synthetic_path, index=False)
    print(f"Saved raw synthetic reviews to {synthetic_path}")

    print("\n--- STEP 3: Merging Datasets ---")
    df_merged = pd.concat([df_real_aligned, df_fake], ignore_index=True)
    print(f"Combined dataset size: {len(df_merged):,} reviews.")
    print("Class distribution:")
    print(df_merged["label"].value_counts().rename({0: "Genuine", 1: "Fake"}))
    print("Origin segmentation:")
    print(df_merged["origin"].value_counts())

    print("\n--- STEP 4: Preprocessing & Feature Extraction ---")
    df_processed = preprocess_pipeline(df_merged, text_col="text")

    output_path = processed_dir / "fake_reviews_processed.parquet"
    df_processed.to_parquet(output_path, index=False)
    print(f"\nFinal restaurant dataset successfully processed and saved to: {output_path.resolve()}")


if __name__ == "__main__":
    build_restaurant_dataset()
