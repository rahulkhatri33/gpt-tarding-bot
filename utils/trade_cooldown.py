# utils/trade_cooldown.py

import time
from typing import Optional
from core.config import get_cooldown_minutes_by_source
from core.logger import global_logger as logger

# === Internal Cooldown Tracker (UNIX timestamp) ===
_COOLDOWN_STORE = {}

def _key(symbol: str, direction: str) -> str:
    return f"{symbol.upper()}|{direction.lower()}"

def set_cooldown(symbol: str, direction: str, source: str = "unknown") -> None:
    """
    Sets a cooldown for a specific symbol-direction pair based on source type.
    """
    cooldown_minutes = get_cooldown_minutes_by_source(source)
    key = _key(symbol, direction)
    expiry_time = time.time() + cooldown_minutes * 60
    _COOLDOWN_STORE[key] = expiry_time
    logger.log_info(f"⏳ Cooldown set for {key} until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expiry_time))}.")

def is_in_cooldown(symbol: str, direction: str, source: str = "unknown") -> bool:
    """
    Checks if the cooldown period is still active for the symbol-direction pair.
    """
    key = _key(symbol, direction)
    expiry = _COOLDOWN_STORE.get(key)
    return expiry is not None and time.time() < expiry

def clear_cooldown(symbol: str, direction: str) -> None:
    """
    Clears cooldown immediately (manual intervention or on exit if allowed).
    """
    key = _key(symbol, direction)
    if key in _COOLDOWN_STORE:
        del _COOLDOWN_STORE[key]
        logger.log_info(f"❌ Cooldown manually cleared for {key}.")
