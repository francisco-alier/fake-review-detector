import logging
import os
import sys
from pathlib import Path
import yaml


def setup_logging(log_file_name="pipeline.log"):
    """
    Sets up logging to print to both stdout and a log file in logs/ directory.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / log_file_name

    # Clear existing handlers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Get logger for main thread
    logger = logging.getLogger("setup")
    logger.info(f"Logging initialized. Writing logs to: {log_file_path.resolve()}")


def load_config(config_path="config.yaml") -> dict:
    """Loads configuration parameters from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path.resolve()}")
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


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
