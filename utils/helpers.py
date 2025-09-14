# utils/helpers.py
import math
from core.logger import global_logger as logger
from utils.exchange import get_price_tick_size, get_qty_step_size, round_to_step

def adjust_to_tick_size(symbol: str, price: float, precision: int = 8) -> float:
    """
    Adjust `price` to Binance PRICE_FILTER tickSize.
    Safe, never tries to convert the symbol to float.
    """
    try:
        tick = get_price_tick_size(symbol)
        if tick > 0:
            # floor to the nearest tick
            adjusted = math.floor(price / tick) * tick
            return round(adjusted, precision)
        return round(price, precision)
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Tick size adjustment failed: {e}")
        # Fallback so we don't crash the whole cycle
        return round(float(price), precision)


def adjust_to_step_size(symbol: str, qty: float, precision: int = 8) -> float:
    """
    Adjust `qty` to Binance LOT_SIZE stepSize.
    """
    try:
        step = get_qty_step_size(symbol)
        if step > 0:
            adjusted = math.floor(qty / step) * step
            return round(adjusted, precision)
        return round(qty, precision)
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Step size adjustment failed: {e}")
        return round(float(qty), precision)
