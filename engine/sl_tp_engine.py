# engine/sl_tp_engine.py
"""
SL/TP and trailing stop calculations for scalper.

Patched full file — enforces a minimum/fallback SL when the dynamically computed SL
is equal or too-close to entry (prevents zero-distance SL which caused skipped exits).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Any
from core.logger import global_logger as logger
# These should exist in your config module (they are used in your codebase)
from core.config import get_scalper_config, get_scalper_fixed_sl_tp_pct


# ========================
# Core SL/TP Calculations
# ========================


def calculate_fixed_sl_tp(entry_price: float, direction: str, sl_pct: float, tp_pct: float) -> Tuple[float, float]:
    """Fixed percentage SL/TP"""
    try:
        if direction == "long":
            return round(entry_price * (1 - sl_pct), 8), round(entry_price * (1 + tp_pct), 8)
        return round(entry_price * (1 + sl_pct), 8), round(entry_price * (1 - tp_pct), 8)
    except Exception as e:
        logger.log_error(f"calculate_fixed_sl_tp error: {e}")
        raise


def calculate_scalper_sl_tp(pair: str, entry_price: float, direction: str, df: pd.DataFrame) -> Tuple[float, float]:
    """
    Dynamic SL/TP with swing points.

    Defensive behavior:
      - If df is invalid, fall back to fixed SL/TP from config.
      - If computed SL is equal or too-close to entry (e.g. swing low == entry),
        apply a fallback SL computed from `fallback_sl_pct` in config so SL is meaningful.
    Returns (stop_loss, take_profit).
    """
    try:
        config = get_scalper_config() or {}
    except Exception:
        config = {}

    try:
        # Validate entry_price
        try:
            entry_price = float(entry_price)
            if entry_price <= 0:
                raise ValueError("invalid entry_price")
        except Exception:
            # invalid entry price -> fallback to fixed
            logger.log_warning(f"{pair} calculate_scalper_sl_tp: invalid entry_price={entry_price!r}; using fixed SL/TP fallback.")
            sl_pct, tp_pct = get_scalper_fixed_sl_tp_pct(pair)
            return calculate_fixed_sl_tp(entry_price or 0.0, direction, sl_pct, tp_pct)

        # guard: ensure df is DataFrame and has required columns
        if not isinstance(df, pd.DataFrame) or not {"high", "low"}.issubset(set(df.columns)):
            logger.log_debug(f"calculate_scalper_sl_tp: df invalid for pair={pair}. Falling back to fixed percentages.")
            sl_pct, tp_pct = get_scalper_fixed_sl_tp_pct(pair)
            return calculate_fixed_sl_tp(entry_price, direction, sl_pct, tp_pct)

        # config-driven params
        lookback = int(config.get("swing_sl_lookback", 5))
        if lookback < 1:
            lookback = 1
        rr_ratio = float(config.get("risk_reward_ratio", 2.0))

        # minimum SL distance (a tiny guard to avoid float-rounding issues)
        min_sl_pct = float(config.get("min_sl_distance_pct", 0.001))  # e.g. 0.001 => 0.1%
        # fallback SL percent (used if swing SL is too-close or equal to entry)
        fallback_sl_pct = float(config.get("fallback_sl_pct", 0.03))  # e.g. 0.03 => 3%
        # effective threshold for deciding "too close"
        effective_min_pct = max(min_sl_pct, 1e-8)

        # compute candidate SL from swing points
        if direction == "long":
            swing_low = float(df["low"].iloc[-lookback:].min())
            # Minimum absolute distance required by min_sl_pct
            min_dist_abs = entry_price * effective_min_pct
            # initial candidate is the lower of swing_low or entry - min_dist_abs,
            # but we want SL to be below entry so we take the higher of (swing_low) and (entry - min_dist_abs)
            initial_sl = max(swing_low, entry_price - min_dist_abs)

            # computed distance from entry to candidate SL
            sl_distance = entry_price - initial_sl

            # If the candidate is non-positive distance or too small (<= min_sl_pct), enforce fallback_sl_pct
            if sl_distance <= 0 or sl_distance < (entry_price * min_sl_pct):
                logger.log_warning(
                    f"{pair} ⚠ computed SL too close/equal to entry (entry={entry_price}, candidate_sl={initial_sl}, sl_distance={sl_distance}). "
                    f"Applying fallback_sl_pct={fallback_sl_pct}"
                )
                sl = entry_price * (1.0 - fallback_sl_pct)
            else:
                sl = initial_sl

            tp = entry_price + (entry_price - sl) * rr_ratio
        else:
            swing_high = float(df["high"].iloc[-lookback:].max())
            min_dist_abs = entry_price * effective_min_pct
            initial_sl = min(swing_high, entry_price + min_dist_abs)
            sl_distance = initial_sl - entry_price

            if sl_distance <= 0 or sl_distance < (entry_price * min_sl_pct):
                logger.log_warning(
                    f"{pair} ⚠ computed SL too close/equal to entry (entry={entry_price}, candidate_sl={initial_sl}, sl_distance={sl_distance}). "
                    f"Applying fallback_sl_pct={fallback_sl_pct}"
                )
                sl = entry_price * (1.0 + fallback_sl_pct)
            else:
                sl = initial_sl

            tp = entry_price - (sl - entry_price) * rr_ratio

        # safety rounding (high precision; your symbol precision layer will round further on order submission)
        return round(float(sl), 8), round(float(tp), 8)

    except Exception as e:
        logger.log_error(f"Dynamic SL/TP failed for {pair}: {e}")
        logger.log_debug("Falling back to fixed SL/TP.")
        try:
            sl_pct, tp_pct = get_scalper_fixed_sl_tp_pct(pair)
        except Exception:
            sl_pct, tp_pct = (0.02, 0.04)
        return calculate_fixed_sl_tp(entry_price, direction, sl_pct, tp_pct)


def calculate_ml_style_sl_tp(entry_price: float, prediction: float, direction: str) -> Tuple[float, float]:
    """ML-style dynamic SL/TP"""
    sl_buffer = 0.004
    tp_buffer = 0.006
    confidence = min(max(prediction, 0), 1)

    sl_adj = sl_buffer * (1 - confidence)
    tp_adj = tp_buffer * (0.8 + confidence)

    if direction == "long":
        return round(entry_price * (1 - sl_adj), 8), round(entry_price * (1 + tp_adj), 8)
    return round(entry_price * (1 + sl_adj), 8), round(entry_price * (1 - tp_adj), 8)


# ========================
# Trailing Stop Functions
# ========================


def _to_float_safe(v: Any) -> Optional[float]:
    """
    Try to convert v to float. Accept single-element lists/tuples, numeric strings, numpy/pandas scalars.
    Return None if not parseable.
    """
    try:
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        if hasattr(v, "item"):
            try:
                return float(v.item())
            except Exception:
                pass
        return float(v)
    except Exception:
        return None


def calculate_scalper_trailing_stop(
    a: Union[str, float, int, list, tuple],
    b: Optional[Union[float, int]] = None,
    c: Optional[str] = None,
    trailing_pct: float = 0.005,
) -> Optional[float]:
    """
    Defensive trailing stop function.

    Accepts either:
      - calculate_scalper_trailing_stop(entry_price, current_price, direction, trailing_pct=...)
      OR
      - calculate_scalper_trailing_stop(symbol, current_price, direction, trailing_pct=...)
        (in this mode function will try to read stored entry_price via core.position_manager)

    Returns:
      - float trailing stop price, rounded to 6 decimals
      - or None if it cannot compute a valid trailing stop (caller should handle None)
    """
    entry_price_raw = None
    current_price_raw = None
    direction = None

    # If first arg is a string, treat it as symbol and interpret as (symbol, current_price, direction)
    if isinstance(a, str):
        symbol = a
        current_price_raw = b
        direction = c
        # try to look up stored entry_price for this symbol/direction
        try:
            # local import to avoid circular imports at module import time
            from core.position_manager import position_manager

            pos = position_manager.get_position(symbol, direction) if direction else None
            if pos and isinstance(pos, dict):
                # prefer explicit numeric fields; if entry missing allow fallback
                entry_price_raw = pos.get("entry_price") or pos.get("entryPrice") or pos.get("entry_price_estimated") or None
        except Exception as e:
            logger.log_debug(f"{a} ⚠️ Could not read entry_price from position_manager: {e}")
            entry_price_raw = None
    else:
        # treat (entry_price, current_price, direction)
        entry_price_raw = a
        current_price_raw = b
        direction = c

    # Coerce numeric values safely
    entry_price = _to_float_safe(entry_price_raw)
    current_price = _to_float_safe(current_price_raw)
    trailing_pct_f = _to_float_safe(trailing_pct)
    if trailing_pct_f is None:
        trailing_pct_f = 0.005

    if current_price is None:
        logger.log_error(f"calculate_scalper_trailing_stop: invalid current_price {current_price_raw!r}")
        return None

    # If entry_price missing, attempt fallback: use current_price as anchor (conservative),
    # but log warning so operator can inspect.
    if entry_price is None:
        logger.log_warning(
            f"calculate_scalper_trailing_stop: missing entry_price, using current_price ({current_price}) as fallback. Original entry raw: {entry_price_raw!r}"
        )
        entry_price = current_price

    try:
        if not isinstance(direction, str):
            # make direction string if possible
            direction = str(direction) if direction is not None else "long"

        direction = direction.lower()
        if direction == "long":
            trailing_stop = float(current_price) * (1 - trailing_pct_f)
            bound = float(entry_price) * (1 - trailing_pct_f * 2)
            out = round(max(trailing_stop, bound), 6)
            return out
        else:
            trailing_stop = float(current_price) * (1 + trailing_pct_f)
            bound = float(entry_price) * (1 + trailing_pct_f * 2)
            out = round(min(trailing_stop, bound), 6)
            return out
    except Exception as e:
        logger.log_error(f"calculate_scalper_trailing_stop computation failed: {e}")
        logger.log_debug(f"inputs: entry_price={entry_price!r}, current_price={current_price!r}, direction={direction!r}, trailing_pct={trailing_pct_f!r}")
        return None


def calculate_trailing_stop_ml(entry_price: float, current_price: float, direction: str, prediction: float) -> Optional[float]:
    """
    ML-based dynamic trailing stop
    """
    try:
        base_pct = 0.004
        max_pct = 0.008
        confidence = min(max(prediction, 0), 1)
        trailing_pct = base_pct + (max_pct - base_pct) * (1 - confidence)

        trailing_pct_f = _to_float_safe(trailing_pct)
        entry_price_f = _to_float_safe(entry_price)
        current_price_f = _to_float_safe(current_price)
        if entry_price_f is None or current_price_f is None:
            logger.log_error("calculate_trailing_stop_ml: invalid entry/current price")
            return None

        if direction == "long":
            trailing_stop = current_price_f * (1 - trailing_pct_f)
            return round(max(trailing_stop, entry_price_f * (1 - max_pct)), 6)
        else:
            trailing_stop = current_price_f * (1 + trailing_pct_f)
            return round(min(trailing_stop, entry_price_f * (1 + max_pct)), 6)
    except Exception as e:
        logger.log_error(f"calculate_trailing_stop_ml error: {e}")
        logger.log_debug(f"inputs: entry_price={entry_price}, current_price={current_price}, direction={direction}, prediction={prediction}")
        return None


# ========================
# Position Management helpers
# ========================


def calculate_vertical_barrier(df: pd.DataFrame, entry_index: int, max_hold_minutes: int = 60, interval: str = "5m") -> int:
    """Calculate max hold duration in bars"""
    interval_map = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    step = interval_map.get(interval, 5)
    return min(entry_index + (max_hold_minutes // step), max(0, len(df) - 1))


# Quick self-test when invoked directly
if __name__ == "__main__":
    print("=== SL/TP Engine quick test ===")
    # Example where swing low == entry (should force fallback SL)
    import pandas as pd
    df = pd.DataFrame({"high": [0.232, 0.233, 0.231], "low": [0.22808, 0.22808, 0.22808]})
    entry = 0.22808
    sl, tp = calculate_scalper_sl_tp("HBARUSDT", entry, "long", df)
    print("entry:", entry, "computed SL:", sl, "TP:", tp)
