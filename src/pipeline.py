import logging
from pathlib import Path

from src.merge_restaurant_dataset import build_restaurant_dataset
from src.utils import setup_logging

logger = logging.getLogger("pipeline")


def main():
    processed_data_path = Path("data/processed/fake_reviews_processed.parquet")

    # Step 1: Ingest, Merge, and Preprocess Restaurant Data
    if not processed_data_path.exists():
        logger.info(f"Processed restaurant dataset not found at {processed_data_path}. Running builder...")
        build_restaurant_dataset()
    else:
        logger.info(f"Processed restaurant dataset found at {processed_data_path}. Skipping ingestion.")

    logger.info("Pipeline run complete. Ready for model training!")


if __name__ == "__main__":
    setup_logging("pipeline.log")
    main()
