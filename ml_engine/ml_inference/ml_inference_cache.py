# ml_engine/ml_inference/ml_inference_cache.py

import os
import json
import time
import threading
import atexit
from typing import Optional, Dict
from datetime import datetime
from core.logger import global_logger as logger
from utils.notifier import Notifier

notifier = Notifier()

CACHE_FILE = "logs/ml_inference_cache.json"
ML_CACHE = {}

def _make_key(symbol: str, direction: str) -> str:
    return f"{symbol.upper()}|{direction.lower()}"

def cache_result(symbol: str, direction: str, label: int, confidence: float):
    key = _make_key(symbol, direction)
    ML_CACHE[key] = {
        "label": label,
        "confidence": confidence,
        "timestamp": datetime.now().astimezone().isoformat()
    }
    logger.log_debug(f"ML Cache ➕ SET: {key} → label={label}, conf={confidence:.4f}")

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(ML_CACHE, f)
        logger.log_once(f"📁 ML cache saved. Entries: {len(ML_CACHE)}", level="INFO")
    except Exception as e:
        logger.log_error(f"❌ Failed to save ML cache: {e}")
        notifier.send_critical(f"❌ Critical: Failed to save ML cache to disk. Restart or manual check needed.\nError: {e}")

def load_cache():
    global ML_CACHE
    if not os.path.exists(CACHE_FILE):
        logger.log_debug("🗂️ No ML cache file found.")
        return

    try:
        with open(CACHE_FILE, "r") as f:
            raw_cache = json.load(f)
        now = datetime.now().astimezone()
        filtered = {}

        for k, v in raw_cache.items():
            if isinstance(v, dict) and all(key in v for key in ["label", "confidence", "timestamp"]):
                filtered[k] = v  # No age check here
            else:
                logger.log_error(f"❌ Bad cache entry at {k}: {v} — skipped.")
        ML_CACHE = filtered
        logger.log_info(f"✅ Loaded ML cache from disk. Valid entries: {len(ML_CACHE)}")
    except Exception as e:
        logger.log_error(f"❌ Failed to load ML cache: {e}")
        notifier.send_critical(f"❌ Critical: ML cache loading failed. Check disk integrity.\nError: {e}")

def get_latest_prediction(symbol: str, direction: str) -> Optional[float]:
    key = f"{symbol.upper()}|{direction.lower()}"
    entry = ML_CACHE.get(key)

    if entry and "confidence" in entry:
        return entry["confidence"]
    return None
