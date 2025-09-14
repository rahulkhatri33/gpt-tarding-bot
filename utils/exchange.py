# utils/exchange.py
import os
from typing import Dict, Any, Optional, Tuple
from binance.client import Client
from dotenv import load_dotenv
from core.logger import global_logger as logger
from core.config import get_config

# Load configuration
config = get_config()
binance_config = config.get("binance", {})
base_url = binance_config.get("base_url", "https://fapi.binance.com")  # For reference, not passed to Client

# Single, shared client for futures
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_API_SECRET")
)

# Set leverage for all supported symbols
def set_leverage(symbol: str, leverage: int = 20):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.log_info(f"Set leverage to {leverage}x for {symbol}")
    except Exception as e:
        logger.log_error(f"Failed to set leverage for {symbol}: {e}")

# -------- Exchange info cache --------
_exchange_info_cache: Optional[Dict[str, Any]] = None
_symbol_map: Dict[str, Dict[str, Any]] = {}

def _refresh_exchange_info() -> None:
    """Fetch and cache futures exchange info once."""
    global _exchange_info_cache, _symbol_map
    try:
        _exchange_info_cache = client.futures_exchange_info()
        _symbol_map = {s["symbol"]: s for s in _exchange_info_cache.get("symbols", [])}
        # Set leverage for all symbols in config
        base_pairs = config.get("base_pairs", [])
        for symbol in base_pairs:
            set_leverage(symbol, 20)
    except Exception as e:
        logger.log_error(f"Failed to refresh futures exchange info: {e}")

def get_symbol_info(symbol: str, refresh: bool = False) -> Optional[Dict[str, Any]]:
    """
    Return the full Binance symbol info dict for `symbol`.
    """
    if refresh or _exchange_info_cache is None:
        _refresh_exchange_info()
    return _symbol_map.get(symbol.upper())

def get_price_tick_size(symbol: str) -> float:
    """
    Return the PRICE_FILTER tickSize for a symbol (0.0 if not found).
    """
    info = get_symbol_info(symbol)
    if not info:
        return 0.0
    for f in info.get("filters", []):
        if f.get("filterType") == "PRICE_FILTER":
            return float(f.get("tickSize", 0.0))
    return 0.0

def get_qty_step_size(symbol: str) -> float:
    """
    Return the LOT_SIZE stepSize for a symbol (0.0 if not found).
    """
    info = get_symbol_info(symbol)
    if not info:
        return 0.0
    for f in info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            return float(f.get("stepSize", 0.0))
    return 0.0

def round_to_step(value: float, step: float, precision: int = 8) -> float:
    """
    Generic rounding helper to a given step.
    """
    if step <= 0:
        return round(value, precision)
    return round((value // step) * step, precision)

def get_futures_balance() -> float:
    """
    Return the total USDT balance in the futures wallet.
    """
    try:
        balance = client.futures_account()
        return float(balance['totalWalletBalance'])
    except Exception as e:
        logger.log_error(f"Failed to fetch futures balance: {e}")
        return 0.0