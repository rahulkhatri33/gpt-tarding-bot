from __future__ import annotations
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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

        # Update buy trailing stop (flipping logic with buy_atr)
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

        # Update sell trailing stop (flipping logic with sell_atr)
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
    Calculate the trade quantity based on USD allocation from config.
    """
    try:
        # Get allocation from config - FIXED: use proper config structure
        usd_allocation_dict = settings.get("usd_allocation_scalper", {})
        if isinstance(usd_allocation_dict, dict):
            usd_allocation = float(usd_allocation_dict.get(symbol, 50.0))  # Default to 50 USDT
        else:
            usd_allocation = 50.0  # fallback
        
        # Get account balance in USDT
        balance = binance_utils.get_futures_balance()
        if balance is None or balance <= 0:
            logger.log_error(f"No USDT balance available for {symbol}")
            return 0.0
            
        # Use the smaller of allocation or available balance
        amount_to_use = min(usd_allocation, balance)
        quantity = amount_to_use / price
        
        # Adjust for Binance quantity precision and minimum quantity
        quantity_precision = binance_utils.get_quantity_precision(symbol)
        symbol_info = binance_utils.get_symbol_info(symbol)
        
        if quantity_precision is not None:
            quantity = round(quantity, quantity_precision)
        
        # Check minimum quantity requirement
        if symbol_info and "minQuantity" in symbol_info:
            min_qty = symbol_info["minQuantity"]
            if quantity < min_qty:
                logger.log_warning(f"{symbol} Quantity {quantity} below minimum {min_qty}, adjusting...")
                quantity = max(quantity, min_qty)
                # Re-round after adjustment
                if quantity_precision is not None:
                    quantity = round(quantity, quantity_precision)
        
        # Final validation - ensure quantity meets Binance minimum notional value (5 USDT)
        notional_value = quantity * price
        if notional_value < 5.0:
            logger.log_warning(f"{symbol} Notional value {notional_value} below 5 USDT, adjusting quantity...")
            min_quantity = 5.0 / price
            if quantity_precision is not None:
                min_quantity = round(min_quantity, quantity_precision)
            quantity = max(quantity, min_quantity)
        
        # Final validation
        if quantity <= 0:
            logger.log_error(f"{symbol} Invalid quantity after adjustments: {quantity}")
            return 0.0
            
        logger.log_info(f"{symbol} ✅ Calculated quantity: {quantity} (allocation: {usd_allocation} USDT, price: {price})")
        return quantity
        
    except Exception as e:
        logger.log_error(f"Quantity calculation error for {symbol}: {str(e)[:200]}")
        return 0.0

def _calculate_sl_tp(df: pd.DataFrame, settings: Dict, side: str, price: float) -> TradeExit:
    try:
        swing_lookback = int(settings.get("swing_sl_lookback", 5))
        min_sl_distance_pct = float(settings.get("min_sl_distance_pct", 0.02))
        risk_reward_ratio = float(settings.get("risk_reward_ratio", 2.0))
        static_sl_pct = float(settings.get("static_sl_pct", 0.03))
        static_tp_pct = float(settings.get("static_tp_pct", 0.06))

        buy_len = int(_get_ut_param(settings, "buy_atr_period", int(settings.get("ut_buy_atr_period", 10))))
        sell_len = int(_get_ut_param(settings, "sell_atr_period", int(settings.get("ut_sell_atr_period", 10))))
        mult = float(_get_ut_param(settings, "key_value", float(settings.get("ut_multiplier", 1.0))))

        buy_atr = ta.atr(df["high"], df["low"], df["close"], length=buy_len).iloc[-1]
        sell_atr = ta.atr(df["high"], df["low"], df["close"], length=sell_len).iloc[-1]

        buy_trailing_stop = price - mult * buy_atr
        sell_trailing_stop = price + mult * sell_atr

        if bool(settings.get("use_dynamic_sl_tp", True)):
            swing_low = df["low"].iloc[-swing_lookback:].min()
            swing_high = df["high"].iloc[-swing_lookback:].max()
            raw_sl = swing_low if side == "LONG" else swing_high
            raw_sl_pct = abs((price - raw_sl) / price)
            sl_pct = max(raw_sl_pct, min_sl_distance_pct)
            sl = price * (1 - sl_pct) if side == "LONG" else price * (1 + sl_pct)
            tp_pct = sl_pct * risk_reward_ratio
            tp = price * (1 + tp_pct) if side == "LONG" else price * (1 - tp_pct)
        else:
            sl_pct, tp_pct = static_sl_pct, static_tp_pct
            sl = price * (1 - sl_pct) if side == "LONG" else price * (1 + sl_pct)
            tp = price * (1 + tp_pct) if side == "LONG" else price * (1 - tp_pct)

        # ✅ NEW partial TP at 1:1
        if side == "LONG":
            partial = price + (price - sl)
        else:
            partial = price - (sl - price)

        trailing = buy_trailing_stop if side == "LONG" else sell_trailing_stop
        return TradeExit(trailing_stop=trailing, sl=sl, tp=tp, sl_pct=sl_pct, tp_pct=tp_pct,
                         partial_tp=partial, partial_size=0.5)
    except Exception as e:
        logger.log_error(f"SL/TP calculation error: {str(e)[:200]}")
        return TradeExit(price, price, price, 0.0, 0.0)

# -----------------------------
# Entry evaluation (patched)
# -----------------------------
def evaluate_scalper_entry(df: Union[pd.DataFrame, str], settings: Dict) -> Tuple[Optional[str], Optional[TradeExit]]:
    try:
        df = _ensure_dataframe(df)
        if df.empty:
            return None, None

        symbol = settings.get("symbol", "")
        open_trades = load_open_trades()

        # --- ✅ Time filter ---
        if bool(settings.get("filters", {}).get("use_time_filter", False)):
            start_hour, end_hour = settings.get("allowed_trading_hours", [0, 24])
            current_hour = pd.Timestamp.now(tz=timezone.utc).hour
            if not (start_hour <= current_hour < end_hour):
                logger.log_info(f"{symbol} ⏰ Time filter active: {current_hour}h not in [{start_hour}, {end_hour})")
                return None, None

        # --- ✅ Calculate UT signals before trend filter ---
        df = calculate_ut_signals(df, settings)

        # --- ✅ Trend filter ---
        if bool(settings.get("filters", {}).get("use_trend_filter", False)):
            ema_period = int(settings.get("ema_filter_period", 200))
            ema_val = df["close"].ewm(span=ema_period).mean().iloc[-1]
            current_price = df["close"].iloc[-1]
            
            # Only allow LONG when price above EMA, SHORT when below
            if current_price < ema_val and df["ut_buy_signal"].iloc[-1] == 1.0:
                logger.log_info(f"{symbol} 📉 Trend filter: Price below EMA {ema_val:.2f}, rejecting LONG")
                return None, None
            if current_price > ema_val and df["ut_sell_signal"].iloc[-1] == 1.0:
                logger.log_info(f"{symbol} 📈 Trend filter: Price above EMA {ema_val:.2f}, rejecting SHORT")
                return None, None

        # --- Min body filter ---
        if bool(_get_min_body_param(settings, "enabled", False)):
            if not _passes_min_body_filter(df, settings):
                logger.log_info(f"{symbol} 🔍 Min body filter: Candle body too small")
                return None, None

        # --- Indicators (STC + UT Bot) ---
        stc_success = False
        if bool(settings.get("filters", {}).get("use_stc_confirmation", False)):
            stc_fast = int(settings.get("stc_fast_length", 23))
            stc_slow = int(settings.get("stc_slow_length", 50))
            stc_signal = int(settings.get("stc_signal_period", 10))
            
            df, stc_success = _calculate_stc(
                df, stc_fast, stc_slow, stc_signal,
                int(settings.get("stc_cycle_length", 80)),
                float(settings.get("stc_buy_threshold", 25.0)),
                float(settings.get("stc_sell_threshold", 75.0)),
            )

        last = df.iloc[-1]
        price = float(last["close"])

        side, sltp = None, None

        if bool(settings.get("filters", {}).get("use_stc_confirmation", False)) and stc_success:
            stc_col = f"STC_{int(settings.get('stc_fast_length', 23))}_{int(settings.get('stc_slow_length', 50))}_{int(settings.get('stc_signal_period', 10))}"
            stc_val = float(last.get(stc_col, np.nan))
            stc_prev = float(df[stc_col].iloc[-2]) if stc_col in df.columns and len(df) >= 2 else np.nan
            
            if last["ut_buy_signal"] == 1.0 and not np.isnan(stc_val) and not np.isnan(stc_prev):
                if stc_val < float(settings.get("stc_buy_threshold", 25.0)) and stc_val > stc_prev:
                    side, sltp = "LONG", _calculate_sl_tp(df, settings, "LONG", price)
                    logger.log_info(f"{symbol} ✅ STC confirmed LONG: STC={stc_val:.1f} < {settings.get('stc_buy_threshold', 25.0)} and rising")
            
            if last["ut_sell_signal"] == 1.0 and not np.isnan(stc_val) and not np.isnan(stc_prev):
                if stc_val > float(settings.get("stc_sell_threshold", 75.0)) and stc_val < stc_prev:
                    side, sltp = "SHORT", _calculate_sl_tp(df, settings, "SHORT", price)
                    logger.log_info(f"{symbol} ✅ STC confirmed SHORT: STC={stc_val:.1f} > {settings.get('stc_sell_threshold', 75.0)} and falling")
        else:
            if last["ut_buy_signal"] == 1.0:
                side, sltp = "LONG", _calculate_sl_tp(df, settings, "LONG", price)
                logger.log_info(f"{symbol} ✅ UT Bot LONG signal")
            
            if last["ut_sell_signal"] == 1.0:
                side, sltp = "SHORT", _calculate_sl_tp(df, settings, "SHORT", price)
                logger.log_info(f"{symbol} ✅ UT Bot SHORT signal")

        if side is None or sltp is None:
            return None, None

        existing_dir = open_trades.get(symbol, {}).get('direction')
        if existing_dir and existing_dir.upper() == side:
            logger.log_info(f"{symbol} ⚠️ Already have {side} position, skipping")
            return None, None

        logger.log_info(f"{symbol} 🚀 Signal confirmed: {side} at {price}, SL={sltp.sl}, TP={sltp.tp}")
        return side, sltp

    except Exception as e:
        logger.log_error(f"Scalper entry evaluation error: {str(e)[:200]}")
        return None, None