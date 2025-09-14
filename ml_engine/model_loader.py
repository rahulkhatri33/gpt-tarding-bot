# ml_engine/model_loader.py

import os
import joblib
from typing import Dict
from core.logger import global_logger as logger
from utils.discord_logger import send_discord_log

MODEL_DIR = "ml_models"

# Optional in-memory cache (can be removed if infer_dual_model handles it)
_model_cache: Dict[str, object] = {}

def load_model(pair: str, model_type: str = "xgb", direction: str = "long"):
    """
    Loads a joblib-pickled model from disk for the given pair and direction.
    """
    model_key = f"{pair}_{model_type}_{direction}"
    if model_key in _model_cache:
        return _model_cache[model_key]

    model_path = os.path.join(MODEL_DIR, pair, f"{pair}_{model_type}_{direction}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    try:
        _model_cache[model_key] = joblib.load(model_path)
        logger.log_info(f"📦 Model loaded and cached: {model_path}")
        return _model_cache[model_key]

    except Exception as e:
        send_discord_log("WARNING: FAILED to load ML models.")
        logger.log_error(f"{pair} ❌ Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model for {pair} - {direction}")
