# utils/config_loader.py

import os
import json
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env
load_dotenv()

# Fallback config path if not passed explicitly
DEFAULT_CONFIG_PATH = os.path.join("config", "config.json")


def get_config(config_path: Optional[str] = None) -> dict:
    """
    Loads configuration from JSON and overlays it with environment variables.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Config file not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    # Override sensitive keys from .env instead of JSON
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if api_key:
        config["api_key"] = api_key
    if api_secret:
        config["api_secret"] = api_secret

    return config
