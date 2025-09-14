# engine/entry_engine.py

from core.config import get_confidence_thresholds
from core.logger import global_logger as logger

# === Entry Decision Logic for ML Signals ===

def should_enter(label: int, confidence: float, symbol: str, direction: str) -> bool:
    """
    Determines if an ML trade should be entered based on label and confidence.

    Args:
        label (int): ML model output label (1 = take-profit expected, 0 = no action).
        confidence (float): Confidence level of prediction (0.0 to 1.0).
        symbol (str): Trading pair, e.g., "BTCUSDT".
        direction (str): "long" or "short".

    Returns:
        bool: True if entry should occur, False otherwise.
    """
    if label != 1:
        logger.log_debug(f"🔎 {symbol}-{direction} entry blocked: label != 1 (label={label}).")
        return False

    thresholds = get_confidence_thresholds()
    threshold_key = f"{symbol.upper()}-{direction.lower()}"

    # Fetch specific threshold if available, else default
    threshold = thresholds.get(threshold_key, thresholds.get("default", 0.60))

    if confidence >= threshold:
        logger.log_info(
            f"✅ {symbol}-{direction} entry allowed: confidence {confidence:.4f} >= threshold {threshold:.4f}."
        )
        return True
    else:
        logger.log_debug(
            f"🔎 {symbol}-{direction} entry blocked: confidence {confidence:.4f} < threshold {threshold:.4f}."
        )
        return False
