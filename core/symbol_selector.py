# core/symbol_selector.py

from core.config import get_scalper_config

def get_active_symbols():
    """
    Returns a list of active symbols from the scalper config.
    """
    config = get_scalper_config()
    return config.get("symbols", ["BTCUSDT", "ETHUSDT"])
