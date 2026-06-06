import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from src.utils import setup_logging

from src.generate_synthetic_reviews import generate_reviews
from src.preprocessing import preprocess_pipeline
from src.utils import load_config, load_env_file

logger = logging.getLogger("builder")


def build_restaurant_dataset():
    load_env_file()
    
    # Load settings from config
    config = load_config()
    paths = config["paths"]
    data_params = config["data_params"]
    
    # Read values from config
    num_synthetic = data_params.get("num_synthetic_reviews", 500)
    processed_data_path = Path(paths["processed_data"])
    processed_dir = processed_data_path.parent
    
    raw_data_path = Path(paths["raw_data"])
    raw_dir = raw_data_path.parent
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- STEP 1: Ingesting Real Yelp Reviews ---")
    try:
        logger.info("Downloading a slice of real Yelp reviews from Hugging Face (5,000 samples)...")
        # yelp test split is smaller and faster to load
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
        logger.info(f"Loaded {len(df_real_aligned):,} real Yelp reviews.")
    except Exception as e:
        logger.error(f"Error downloading Yelp reviews: {e}")
        return

    logger.info("--- STEP 2: Generating Synthetic Fake Reviews from Config ---")
    synthetic_path = raw_dir / "synthetic_reviews.parquet"
    
    logger.info(f"Generating {num_synthetic} synthetic fake restaurant reviews...")
    df_fake = generate_reviews(num_reviews=num_synthetic)
    df_fake.to_parquet(synthetic_path, index=False)
    logger.info(f"Saved raw synthetic reviews to {synthetic_path.resolve()}")

    logger.info("--- STEP 3: Merging Datasets ---")
    # Align columns of df_fake (drop restaurant_name if present to match Yelp columns)
    df_fake_aligned = df_fake[["category", "rating", "text", "label", "origin"]]
    df_merged = pd.concat([df_real_aligned, df_fake_aligned], ignore_index=True)
    logger.info(f"Combined dataset size: {len(df_merged):,} reviews.")
    logger.info(f"Class distribution:\n{df_merged['label'].value_counts().rename({0: 'Genuine', 1: 'Fake'})}")
    logger.info(f"Origin segmentation:\n{df_merged['origin'].value_counts()}")

    logger.info("--- STEP 4: Preprocessing & Feature Extraction ---")
    df_processed = preprocess_pipeline(df_merged, text_col="text")

    df_processed.to_parquet(processed_data_path, index=False)
    logger.info(f"Final restaurant dataset successfully processed and saved to: {processed_data_path.resolve()}")


if __name__ == "__main__":
    setup_logging("builder.log")
    build_restaurant_dataset()
