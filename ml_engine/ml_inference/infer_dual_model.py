# ml_engine/ml_inference/infer_dual_model.py

import os
import joblib
import numpy as np
import pandas as pd
import json

from ml_engine.model_loader import load_model
from core.logger import global_logger as logger
from core.config import get_confidence_thresholds

# === Globals ===
MODEL_CACHE = {}
CONF_THRESHOLDS = get_confidence_thresholds()
TOP_FEATURES_PATH = "config/top_features.json"

# Load top features per (symbol, direction) once
with open(TOP_FEATURES_PATH, "r") as f:
    TOP_FEATURES = json.load(f)

# === Preload all models into memory at startup ===
def preload_models(base_pairs):
    for pair in base_pairs:
        for direction in ["long", "short"]:
            key = f"{pair}_{direction}"
            try:
                MODEL_CACHE[key] = load_model(pair, "xgb", direction)
            except Exception as e:
                logger.log_error(f"{pair}-{direction} ❌ Failed to preload model: {e}")

# === Fetch model from cache ===
def get_model(pair: str, direction: str):
    key = f"{pair}_{direction}"
    if key not in MODEL_CACHE:
        logger.log_once(f"❌ Model cache miss: {key} — model not preloaded or failed to load.")
        raise RuntimeError(f"❌ Model not found in cache: {key}")
    return MODEL_CACHE[key]

# === Get expected features for model inference ===
def get_feature_list(pair: str, direction: str) -> list:
    key = f"{pair}_{direction}"
    top = TOP_FEATURES.get(key, [])

    # ✅ Do not include alt_btc_ratio for BTCUSDT
    if pair == "BTCUSDT":
        core = ["volume", "volume_ma_5", "atr_5", "ema_divergence"]
    else:
        core = ["volume", "volume_ma_5", "atr_5", "alt_btc_ratio", "ema_divergence"]

    return top + [f for f in core if f not in top]

# === Inference function for a given pair/direction ===
def infer_dual_model(pair: str, df: pd.DataFrame, direction: str) -> dict:
    if df.empty or df.shape[0] < 1:
        raise ValueError(f"{pair}-{direction} ❌ Input DataFrame is empty or malformed.")

    model = get_model(pair, direction)

    try:
        latest = df.copy()
        if pair == "BTCUSDT" and "alt_btc_ratio" in latest.columns:
            latest = latest.drop(columns=["alt_btc_ratio"])
    except Exception as e:
        raise RuntimeError(f"{pair}-{direction} ❌ Failed to extract latest row for inference: {e}")

    try:
        probs = model.predict_proba(latest)[0]
    except Exception as e:
        raise RuntimeError(
            f"❌ XGB prediction failed for {pair}-{direction}: {e}\n"
            f"Input shape: {latest.shape}\nHead:\n{latest.head()}"
        )

    confidence = float(probs[1])  # Binary: index 1 is TP
    label = int(confidence > 0.5)

    pair_conf = CONF_THRESHOLDS.get(pair, {})
    if isinstance(pair_conf, dict) and direction in pair_conf:
        threshold = pair_conf[direction]
    else:
        threshold = CONF_THRESHOLDS.get("default", 0.60)

    logger.log_ml(
        f"{pair} [{direction.upper()}] 🤖 Inference | Label: {label} | Conf: {confidence:.4f} | Threshold: {threshold:.2f}"
    )

    return {
        "confidence": confidence,
        "label": label,
        "model": "XGB",
        "threshold": threshold
    }
