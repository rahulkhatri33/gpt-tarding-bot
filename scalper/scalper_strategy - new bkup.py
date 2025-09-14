# === Patched scalper_strategy.py ===
# (Full code included, with fixes for filters, SL/TP calculation, and partial TP logic)
# Copy and replace your existing scalper_strategy.py with this file.

from __future__ import annotations
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from core.logger import global_logger as logger
from core.config import get_scalper_config, get_scalper_usd_allocation
from binance_utils import BinanceClient

binance_utils = BinanceClient()

# Use the SAME file the runner / position_manager uses
OPEN_TRADES_FILE = 'open_positions.json'

# -----------------------------
# Position Persistence Helpers
# -----------------------------
def _normalize_positions(raw: Dict) -> Dict[str, Dict]:
    norm: Dict[str, Dict] = {}
    try:
        for k, v in (raw or {}).items():
            if "_" in k:
                sym, direction = k.split("_", 1)
                d = dict(v)
                d["direction"] = direction
                norm[sym] = d
    except Exception as e:
        logger.log_warning(f"⚠️ Failed to normalize open positions: {e}")
    return norm

def load_open_trades(file_path=OPEN_TRADES_FILE) -> Dict:
    try:
        with open(file_path, 'r') as f:
            raw = json.load(f)
            return _normalize_positions(raw)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_open_trades(open_trades: Dict, file_path=OPEN_TRADES_FILE):
    try:
        pass
    except Exception:
        pass

def add_open_trade(symbol: str, direction: str, entry_price: float, size: float, sl: float, tp: float, trailing_stop: float):
    logger.log_info(f"{symbol} add_open_trade() is a no-op (managed by position_manager).")

def close_trade(symbol: str):
    logger.log_info(f"{symbol} close_trade() is a no-op (managed by position_manager).")

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class STCConfig:
    enabled: bool
    fast_length: int
    slow_length: int
    signal_period: int
    cycle_length: int

@dataclass
class TradeExit:
    trailing_stop: float
    sl: float
    tp: float
    sl_pct: float
    tp_pct: float
    partial_tp: Optional[float] = None
    partial_size: float = 0.5

# -----------------------------
# Utilities
# -----------------------------
def _ensure_dataframe(df: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, str):
        try:
            s = df.strip()
            if s.startswith("[") or s.startswith("{"):
                return pd.read_json(io.StringIO(s))
            return pd.read_csv(io.StringIO(s))
        except Exception as e:
            raise TypeError(f"Failed to convert string to DataFrame: {e}")
    raise TypeError(f"evaluate_scalper_entry expected a DataFrame or str, got {type(df)}")

def _rma(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return series.copy()*np.nan
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

# === Min candle body filter (unchanged) ===========================
def _get_min_body_param(settings: Dict, key: str, default: Optional[float | int | bool] = None):
    mcb = settings.get("min_candle_body", {}) if isinstance(settings, dict) else {}
    if key in mcb:
        return mcb.get(key, default)
    mapping = {
        "enabled": "min_body_enabled",
        "pct": "min_body_pct",
        "abs": "min_body_abs",
        "atr_mult": "min_body_atr_mult",
        "atr_period": "min_body_atr_period",
    }
    return settings.get(mapping.get(key, key), default)

def _passes_min_body_filter(df: pd.DataFrame, settings: Dict) -> bool:
    enabled = bool(_get_min_body_param(settings, "enabled", False))
    if not enabled or df.shape[0] < 2:
        return True
    o = float(df["open"].iloc[-1])
    c = float(df["close"].iloc[-1])
    body = abs(c - o)
    thresholds = []
    pct = float(_get_min_body_param(settings, "pct", 0.0) or 0.0)
    if pct > 0:
        thresholds.append(pct * c)
    absv = float(_get_min_body_param(settings, "abs", 0.0) or 0.0)
    if absv > 0:
        thresholds.append(absv)
    atr_mult = float(_get_min_body_param(settings, "atr_mult", 0.0) or 0.0)
    if atr_mult > 0:
        atr_period = int(_get_min_body_param(settings, "atr_period", 14) or 14)
        atr_val = ta.atr(df["high"], df["low"], df["close"], length=atr_period).iloc[-1]
        if pd.notna(atr_val):
            thresholds.append(atr_mult * float(atr_val))
    if not thresholds:
        return True
    required = max(thresholds)
    return body >= required
# ================================================================

# -----------------------------
# STC (Schaff Trend Cycle)
# -----------------------------
def custom_stc(
    df: pd.DataFrame,
    fast_length: int,
    slow_length: int,
    signal_period: int,
    cycle_length: int = 80,
) -> pd.Series:
    """
    STC built from MACD->stochastic->EMA smoothing.
    Implemented to be stable on closed candles.
    """
    try:
        exp1 = df["close"].ewm(span=fast_length, adjust=False).mean()
        exp2 = df["close"].ewm(span=slow_length, adjust=False).mean()
        macd = exp1 - exp2

        lowest_macd = macd.rolling(window=cycle_length, min_periods=cycle_length).min()
        highest_macd = macd.rolling(window=cycle_length, min_periods=cycle_length).max()
        range_macd = (highest_macd - lowest_macd).replace(0, np.nan)

        # stochastic of MACD
        stoch_k = 100 * (macd - lowest_macd) / range_macd
        stoch_k = stoch_k.clip(lower=0.1, upper=99.9)

        # smooth k
        stoch_k_smooth = stoch_k.ewm(span=signal_period, adjust=False).mean()

        # range normalize again (as commonly seen in STC impls)
        lowest_k_s = stoch_k_smooth.rolling(window=cycle_length, min_periods=cycle_length).min()
        highest_k_s = stoch_k_smooth.rolling(window=cycle_length, min_periods=cycle_length).max()
        range_k_s = (highest_k_s - lowest_k_s).replace(0, np.nan)

        stc = 100 * (stoch_k_smooth - lowest_k_s) / range_k_s
        stc = stc.clip(lower=0.1, upper=99.9)
        return stc.ffill().fillna(50.0)
    except Exception as e:
        logger.log_error(f"Custom STC calculation error: {str(e)}")
        return pd.Series(np.nan, index=df.index)


def _calculate_stc(
    df: pd.DataFrame,
    fast_length: int,
    slow_length: int,
    signal_period: int,
    cycle_length: int,
    buy_threshold: float,
    sell_threshold: float,
) -> Tuple[pd.DataFrame, bool]:
    df = df.copy()
    try:
        stc = custom_stc(df, fast_length, slow_length, signal_period, cycle_length)
        if stc is None or stc.isna().all():
            return df, False
        df[f"STC_{fast_length}_{slow_length}_{signal_period}"] = stc
        return df, True
    except Exception as e:
        logger.log_error(f"STC calculation error: {str(e)}")
        return df, False


# -----------------------------
# UT Bot (TradingView-style) signals
# -----------------------------
def _get_ut_param(settings: Dict, key: str, default: Optional[float | int] = None):
    """
    Support both nested config["ut_bot"][key] and flat settings["ut_*"] keys.
    """
    ut_bot = settings.get("ut_bot", {}) if isinstance(settings, dict) else {}
    if key in ut_bot:
        return ut_bot.get(key, default)
    # fallbacks to flat naming used elsewhere in your code
    mapping = {
        "key_value": "ut_multiplier",
        "buy_atr_period": "ut_buy_atr_period",
        "sell_atr_period": "ut_sell_atr_period",
    }
    flat_key = mapping.get(key, key)
    return settings.get(flat_key, default)


def calculate_ut_signals(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """
    TradingView UT Bot based on RMA ATR (Wilder). Signals are placed on CLOSED candles only.
    Outputs numeric flags:
        - df["ut_buy_signal"]  in {0.0, 1.0}
        - df["ut_sell_signal"] in {0.0, 1.0}
    """
    df = df.copy()

    key_value = float(_get_ut_param(settings, "key_value", 1.0))
    buy_atr_period = int(_get_ut_param(settings, "buy_atr_period", 10))
    sell_atr_period = int(_get_ut_param(settings, "sell_atr_period", 10))

    # --- True Range & RMA ATR (non-repainting on closed candles) ---
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["buy_atr"] = _rma(tr, buy_atr_period)
    df["sell_atr"] = _rma(tr, sell_atr_period)

    # Initialize numeric signal columns
    df["ut_buy_signal"] = 0.0
    df["ut_sell_signal"] = 0.0

    # Buy trailing stop (with buy_atr)
    buy_trailing = [0.0] * len(df)
    # Sell trailing stop (with sell_atr)
    sell_trailing = [0.0] * len(df)

    # Iterate over bars (start from 1 to avoid index errors)
    for i in range(1, len(df)):
        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # Update buy trailing stop (with buy_atr)
        nloss = key_value * df["buy_atr"].iloc[i]
        prev_buy_trailing = buy_trailing[i - 1]
        if close > prev_buy_trailing and prev_close > prev_buy_trailing:
            buy_trailing[i] = max(prev_buy_trailing, close - nloss)
        elif close < prev_buy_trailing and prev_close < prev_buy_trailing:
            buy_trailing[i] = min(prev_buy_trailing, close + nloss)
        elif close > prev_buy_trailing:
            buy_trailing[i] = close - nloss
        else:
            buy_trailing[i] = close + nloss

        # Buy signal if crossed above previous buy trailing stop
        if prev_close < buy_trailing[i - 1] and close > buy_trailing[i - 1]:
            df.iat[i, df.columns.get_loc("ut_buy_signal")] = 1.0

        # Update sell trailing stop (with sell_atr)
        nloss = key_value * df["sell_atr"].iloc[i]
        prev_sell_trailing = sell_trailing[i - 1]
        if close > prev_sell_trailing and prev_close > prev_sell_trailing:
            sell_trailing[i] = max(prev_sell_trailing, close - nloss)
        elif close < prev_sell_trailing and prev_close < prev_sell_trailing:
            sell_trailing[i] = min(prev_sell_trailing, close + nloss)
        elif close > prev_sell_trailing:
            sell_trailing[i] = close - nloss
        else:
            sell_trailing[i] = close + nloss

        # Sell signal if crossed below previous sell trailing stop
        if prev_close > sell_trailing[i - 1] and close < sell_trailing[i - 1]:
            df.iat[i, df.columns.get_loc("ut_sell_signal")] = 1.0

    # Ensure no NaNs in outputs
    df["ut_buy_signal"] = df["ut_buy_signal"].fillna(0.0)
    df["ut_sell_signal"] = df["ut_sell_signal"].fillna(0.0)
    return df


# -----------------------------
# SL/TP & Quantity helpers
# -----------------------------
def calculate_quantity(symbol: str, price: float, settings: Dict) -> float:
    """
    Calculate the trade quantity based on account balance, price, and risk percentage.
    """
    try:
        risk_percentage = float(settings.get("risk_percentage", 0.01))
        usd_allocation = get_scalper_usd_allocation(symbol)
        
        # Get leverage from config
        symbol_precisions = settings.get("symbol_precisions", {}).get(symbol, {})
        leverage = float(symbol_precisions.get("leverage", settings.get("leverage", 20)))
        
        # Get quantity precision from config
        quantity_precision = int(symbol_precisions.get("quantityPrecision", 2))
        
        # Get account balance in USDT
        balance = binance_utils.get_futures_balance()
        if balance is None or balance <= 0:
            logger.log_error(f"No USDT balance available for {symbol}")
            return 0.0
            
        logger.log_info(f"{symbol} 💰 Balance: {balance} USDT, Leverage: {leverage}x, Risk: {risk_percentage*100}%")
        
        # Calculate position size with leverage
        position_size = balance * risk_percentage * usd_allocation * leverage
        quantity = position_size / price
        
        logger.log_info(f"{symbol} 📊 Position size: ${position_size:.2f}, Raw quantity: {quantity}")
        
        # Round to config precision
        quantity = round(quantity, quantity_precision)
        
        # Ensure minimum quantity
        min_qty = 0.1  # Reasonable minimum for most coins
        if quantity < min_qty:
            quantity = min_qty
            logger.log_warning(f"{symbol} ⚠️ Quantity below minimum, using {min_qty}")
            
        logger.log_info(f"{symbol} ✅ Final quantity: {quantity} (precision: {quantity_precision})")
        
        return quantity
        
    except Exception as e:
        logger.log_error(f"Quantity calculation error for {symbol}: {str(e)[:200]}")
        return 0.0

def _calculate_sl_tp(df: pd.DataFrame, settings: Dict, side: str, price: float) -> TradeExit:
    """
    Calculate proper SL/TP with validation to prevent invalid values.
    """
    try:
        swing_lookback = int(settings.get("swing_sl_lookback", 5))
        min_sl_distance_pct = float(settings.get("min_sl_distance_pct", 0.005))
        risk_reward_ratio = float(settings.get("risk_reward_ratio", 2.0))
        static_sl_pct = float(settings.get("static_sl_pct", 0.02))
        static_tp_pct = float(settings.get("static_tp_pct", 0.04))

        # Get ATR values for trailing stop
        buy_len = int(_get_ut_param(settings, "buy_atr_period", int(settings.get("ut_buy_atr_period", 10))))
        sell_len = int(_get_ut_param(settings, "sell_atr_period", int(settings.get("ut_sell_atr_period", 10))))
        mult = float(_get_ut_param(settings, "key_value", float(settings.get("ut_multiplier", 1.0))))

        # Calculate ATR safely
        buy_atr_series = ta.atr(df["high"], df["low"], df["close"], length=buy_len)
        sell_atr_series = ta.atr(df["high"], df["low"], df["close"], length=sell_len)
        
        buy_atr = float(buy_atr_series.iloc[-1]) if not buy_atr_series.empty and not pd.isna(buy_atr_series.iloc[-1]) else price * 0.01
        sell_atr = float(sell_atr_series.iloc[-1]) if not sell_atr_series.empty and not pd.isna(sell_atr_series.iloc[-1]) else price * 0.01

        # Calculate trailing stops
        buy_trailing_stop = price - mult * buy_atr
        sell_trailing_stop = price + mult * sell_atr

        if bool(settings.get("use_dynamic_sl_tp", True)):
            if side == "LONG":
                swing_low = float(df["low"].iloc[-swing_lookback:].min())
                raw_sl = swing_low
                raw_sl_pct = abs((price - raw_sl) / price)
                sl_pct = max(raw_sl_pct, min_sl_distance_pct)
                sl = price * (1 - sl_pct)
                tp_pct = sl_pct * risk_reward_ratio
                tp = price * (1 + tp_pct)
                
                if sl >= price or tp <= price:
                    logger.log_warning(f"Dynamic SL/TP invalid for LONG, using static")
                    sl = price * (1 - static_sl_pct)
                    tp = price * (1 + static_tp_pct)
                    sl_pct = static_sl_pct
                    tp_pct = static_tp_pct
                    
            else:  # SHORT
                swing_high = float(df["high"].iloc[-swing_lookback:].max())
                raw_sl = swing_high
                raw_sl_pct = abs((raw_sl - price) / price)
                sl_pct = max(raw_sl_pct, min_sl_distance_pct)
                sl = price * (1 + sl_pct)
                tp_pct = sl_pct * risk_reward_ratio
                tp = price * (1 - tp_pct)
                
                if sl <= price or tp >= price:
                    logger.log_warning(f"Dynamic SL/TP invalid for SHORT, using static")
                    sl = price * (1 + static_sl_pct)
                    tp = price * (1 - static_tp_pct)
                    sl_pct = static_sl_pct
                    tp_pct = static_tp_pct
        else:
            if side == "LONG":
                sl = price * (1 - static_sl_pct)
                tp = price * (1 + static_tp_pct)
                sl_pct = static_sl_pct
                tp_pct = static_tp_pct
            else:  # SHORT
                sl = price * (1 + static_sl_pct)
                tp = price * (1 - static_tp_pct)
                sl_pct = static_sl_pct
                tp_pct = static_tp_pct

        # FINAL VALIDATION
        if side == "LONG":
            if sl >= price:
                sl = price * 0.99
                logger.log_warning(f"LONG SL forced below price: {sl}")
            if tp <= price:
                tp = price * 1.01
                logger.log_warning(f"LONG TP forced above price: {tp}")
        else:  # SHORT
            if sl <= price:
                sl = price * 1.01
                logger.log_warning(f"SHORT SL forced above price: {sl}")
            if tp >= price:
                tp = price * 0.99
                logger.log_warning(f"SHORT TP forced below price: {tp}")

        # Calculate partial TP
        if side == "LONG":
            partial = price + (price - sl)
        else:
            partial = price - (sl - price)

        # Ensure trailing stop makes sense
        trailing = buy_trailing_stop if side == "LONG" else sell_trailing_stop
        if side == "LONG" and trailing >= price:
            trailing = price * 0.995
        elif side == "SHORT" and trailing <= price:
            trailing = price * 1.005

        logger.log_info(f"{side} SL/TP calculated: entry={price}, sl={sl}, tp={tp}, trailing={trailing}")

        return TradeExit(trailing_stop=trailing, sl=sl, tp=tp, sl_pct=sl_pct, tp_pct=tp_pct,
                         partial_tp=partial, partial_size=0.5)
                         
    except Exception as e:
        logger.log_error(f"SL/TP calculation error: {str(e)[:200]}")
        if side == "LONG":
            return TradeExit(price * 0.99, price * 0.98, price * 1.04, 0.02, 0.04)
        else:
            return TradeExit(price * 1.01, price * 1.02, price * 0.96, 0.02, 0.04)

# -----------------------------
# Entry evaluation
# -----------------------------
def evaluate_scalper_entry(df: Union[pd.DataFrame, str], settings: Dict) -> Tuple[Optional[str], Optional[TradeExit]]:
    try:
        df = _ensure_dataframe(df)
        if df.empty:
            return None, None

        symbol = settings.get("symbol", "")
        open_trades = load_open_trades()

        # --- Time filter
        if bool(settings.get("filters", {}).get("use_time_filter", False)):
            start_hour, end_hour = settings.get("allowed_trading_hours", [0, 24])
            if not (start_hour <= df["timestamp"].iloc[-1].hour < end_hour):
                return None, None

        # --- Calculate UT signals before trend filter
        df = calculate_ut_signals(df, settings)

        # --- Trend filter
        if bool(settings.get("filters", {}).get("use_trend_filter", False)):
            ema_period = int(settings.get("ema_filter_period", 200))
            ema_val = df["close"].ewm(span=ema_period).mean().iloc[-1]
            if df["close"].iloc[-1] < ema_val and df["ut_buy_signal"].iloc[-1] == 1.0:
                return None, None
            if df["close"].iloc[-1] > ema_val and df["ut_sell_signal"].iloc[-1] == 1.0:
                return None, None

        # --- Min body filter
        if bool(_get_min_body_param(settings, "enabled", False)):
            if not _passes_min_body_filter(df, settings):
                return None, None

        # --- Indicators (STC + UT Bot)
        stc_success = False
        if bool(settings.get("enable_stc_confirmation", False)):
            df, stc_success = _calculate_stc(
                df,
                int(settings.get("stc_fast_length", 23)),
                int(settings.get("stc_slow_length", 50)),
                int(settings.get("stc_signal_period", 10)),
                int(settings.get("stc_cycle_length", 80)),
                float(settings.get("stc_buy_threshold", 25.0)),
                float(settings.get("stc_sell_threshold", 75.0)),
            )

        last = df.iloc[-1]
        price = float(last["close"])

        side, sltp = None, None

        if bool(settings.get("enable_stc_confirmation", False)) and stc_success:
            stc_col = f"STC_{int(settings.get('stc_fast_length', 23))}_{int(settings.get('stc_slow_length', 50))}_{int(settings.get('stc_signal_period', 10))}"
            stc_val = float(last.get(stc_col, np.nan))
            stc_prev = float(df[stc_col].iloc[-2]) if stc_col in df.columns and len(df) >= 2 else np.nan
            if last["ut_buy_signal"] == 1.0 and not np.isnan(stc_val) and not np.isnan(stc_prev):
                if stc_val < float(settings.get("stc_buy_threshold", 25.0)) and stc_val > stc_prev:
                    side, sltp = "LONG", _calculate_sl_tp(df, settings, "LONG", price)
            if last["ut_sell_signal"] == 1.0 and not np.isnan(stc_val) and not np.isnan(stc_prev):
                if stc_val > float(settings.get("stc_sell_threshold", 75.0)) and stc_val < stc_prev:
                    side, sltp = "SHORT", _calculate_sl_tp(df, settings, "SHORT", price)
        else:
            if last["ut_buy_signal"] == 1.0:
                side, sltp = "LONG", _calculate_sl_tp(df, settings, "LONG", price)
            if last["ut_sell_signal"] == 1.0:
                side, sltp = "SHORT", _calculate_sl_tp(df, settings, "SHORT", price)

        if side is None or sltp is None:
            return None, None

        existing_dir = open_trades.get(symbol, {}).get('direction')
        if existing_dir and existing_dir.upper() == side:
            return None, None

        return side, sltp

    except Exception as e:
        logger.log_error(f"Scalper entry evaluation error: {str(e)[:200]}")
        return None, None