# === Patched scalper_strategy.py (2025-09-04, rev IST+TP/SL guard) ===
# - Min body filter now wired to config.json -> scalper_settings.filters.use_min_body
#   and scalper_settings.{min_body_pct,min_body_abs,min_body_atr_mult,min_body_atr_period}
# - Time filter obeys scalper_settings.filters.use_time_filter and logs UTC hour used
# - Trend filter obeys scalper_settings.filters.use_trend_filter and logs pass/fail
# - Added explicit, human-readable DEBUG lines for every filter decision and final entry
# - Supports IST (or any TZ) via settings.trading_hours_tz_offset_min (e.g., 330 for IST)
# - Guards against TP==SL or too-close TP/SL via settings.min_tp_sl_gap_pct (default 0.001 = 0.1%)
# - UT signals still evaluated ONLY on CLOSED candles
# - No business-logic changes beyond filter wiring + logs

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

# ===================== MIN BODY FILTER (wired to config.json) =====================
# config.json shape (excerpt):
#   "scalper_settings": {
#       "filters": { "use_min_body": true, ... },
#       "min_body_pct": 0.0020,
#       "min_body_abs": 0,
#       "min_body_atr_mult": 0.2,
#       "min_body_atr_period": 14,
#       ...
#   }
# ---------------------------------------------------------------------------------

def _min_body_enabled(settings: Dict) -> bool:
    return bool(settings.get("filters", {}).get("use_min_body", False))

def _min_body_params(settings: Dict) -> Tuple[float, float, float, int]:
    pct = float(settings.get("min_body_pct", 0.0) or 0.0)
    absv = float(settings.get("min_body_abs", 0.0) or 0.0)
    atr_mult = float(settings.get("min_body_atr_mult", 0.0) or 0.0)
    atr_period = int(settings.get("min_body_atr_period", 14) or 14)
    return pct, absv, atr_mult, atr_period

def _passes_min_body_filter(df: pd.DataFrame, settings: Dict) -> Tuple[bool, str]:
    if not _min_body_enabled(settings) or df.shape[0] < 2:
        return True, "disabled"

    o = float(df["open"].iloc[-1])
    c = float(df["close"].iloc[-1])
    body = abs(c - o)

    pct, absv, atr_mult, atr_period = _min_body_params(settings)
    thresholds = []
    if pct > 0:
        thresholds.append(pct * c)
    if absv > 0:
        thresholds.append(absv)
    if atr_mult > 0:
        atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period).iloc[-1]
        if pd.notna(atr):
            thresholds.append(atr_mult * float(atr))

    if not thresholds:
        return True, "no-thresholds"

    required = max(thresholds)
    ok = body >= required
    detail = f"body={body:.6f} >= required={required:.6f} (pct*price={pct*c:.6f}, abs={absv:.6f}, atr_mult={atr_mult}*ATR)"
    return ok, detail

# -----------------------------
# STC (Schaff Trend Cycle) — optional (kept for parity)
# -----------------------------

def custom_stc(
    df: pd.DataFrame,
    fast_length: int,
    slow_length: int,
    signal_period: int,
    cycle_length: int = 80,
) -> pd.Series:
    try:
        exp1 = df["close"].ewm(span=fast_length, adjust=False).mean()
        exp2 = df["close"].ewm(span=slow_length, adjust=False).mean()
        macd = exp1 - exp2

        lowest_macd = macd.rolling(window=cycle_length, min_periods=cycle_length).min()
        highest_macd = macd.rolling(window=cycle_length, min_periods=cycle_length).max()
        range_macd = (highest_macd - lowest_macd).replace(0, np.nan)

        stoch_k = 100 * (macd - lowest_macd) / range_macd
        stoch_k = stoch_k.clip(lower=0.1, upper=99.9)
        stoch_k_smooth = stoch_k.ewm(span=signal_period, adjust=False).mean()

        lowest_k_s = stoch_k_smooth.rolling(window=cycle_length, min_periods=cycle_length).min()
        highest_k_s = stoch_k_smooth.rolling(window=cycle_length, min_periods=cycle_length).max()
        range_k_s = (highest_k_s - lowest_k_s).replace(0, np.nan)

        stc = 100 * (stoch_k_smooth - lowest_k_s) / range_k_s
        stc = stc.clip(lower=0.1, upper=99.9)
        return stc.ffill().fillna(50.0)
    except Exception as e:
        logger.log_error(f"Custom STC calculation error: {str(e)}")
        return pd.Series(np.nan, index=df.index)

# -----------------------------
# UT Bot signals (closed-candle)
# -----------------------------

def _get_ut(settings: Dict, key: str, default: Optional[float|int]=None):
    mapping = {
        "key_value": "ut_multiplier",
        "buy_atr_period": "ut_buy_atr_period",
        "sell_atr_period": "ut_sell_atr_period",
    }
    return settings.get(mapping.get(key, key), default)


def calculate_ut_signals(df: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    df = df.copy()
    key_value = float(_get_ut(settings, "key_value", 1.0))
    buy_atr_period = int(_get_ut(settings, "buy_atr_period", 10))
    sell_atr_period = int(_get_ut(settings, "sell_atr_period", 10))

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["buy_atr"] = _rma(tr, buy_atr_period)
    df["sell_atr"] = _rma(tr, sell_atr_period)

    df["ut_buy_signal"] = 0.0
    df["ut_sell_signal"] = 0.0

    buy_trailing = [0.0] * len(df)
    sell_trailing = [0.0] * len(df)

    for i in range(1, len(df)):
        close = df["close"].iloc[i]
        pclose = df["close"].iloc[i-1]

        nloss = key_value * df["buy_atr"].iloc[i]
        prev_bt = buy_trailing[i-1]
        if close > prev_bt and pclose > prev_bt:
            buy_trailing[i] = max(prev_bt, close - nloss)
        elif close < prev_bt and pclose < prev_bt:
            buy_trailing[i] = min(prev_bt, close + nloss)
        elif close > prev_bt:
            buy_trailing[i] = close - nloss
        else:
            buy_trailing[i] = close + nloss
        if pclose < buy_trailing[i-1] and close > buy_trailing[i-1]:
            df.iat[i, df.columns.get_loc("ut_buy_signal")] = 1.0

        nloss = key_value * df["sell_atr"].iloc[i]
        prev_st = sell_trailing[i-1]
        if close > prev_st and pclose > prev_st:
            sell_trailing[i] = max(prev_st, close - nloss)
        elif close < prev_st and pclose < prev_st:
            sell_trailing[i] = min(prev_st, close + nloss)
        elif close > prev_st:
            sell_trailing[i] = close - nloss
        else:
            sell_trailing[i] = close + nloss
        if pclose > sell_trailing[i-1] and close < sell_trailing[i-1]:
            df.iat[i, df.columns.get_loc("ut_sell_signal")] = 1.0

    df["ut_buy_signal"] = df["ut_buy_signal"].fillna(0.0)
    df["ut_sell_signal"] = df["ut_sell_signal"].fillna(0.0)
    return df

# -----------------------------
# SL/TP & Quantity helpers (unchanged logic)
# -----------------------------

def calculate_quantity(symbol: str, price: float, settings: Dict) -> float:
    try:
        risk_percentage = float(settings.get("risk_percentage", 0.01))
        usd_allocation = get_scalper_usd_allocation(symbol)
        symbol_precisions = settings.get("symbol_precisions", {}).get(symbol, {})
        leverage = float(symbol_precisions.get("leverage", settings.get("leverage", 20)))
        quantity_precision = int(symbol_precisions.get("quantityPrecision", 2))

        balance = binance_utils.get_futures_balance()
        if balance is None or balance <= 0:
            logger.log_error(f"No USDT balance available for {symbol}")
            return 0.0

        logger.log_info(f"{symbol} 💰 Balance: {balance} USDT, Leverage: {leverage}x, Risk: {risk_percentage*100}%")
        position_size = balance * risk_percentage * usd_allocation * leverage
        quantity = position_size / price
        logger.log_info(f"{symbol} 📊 Position size: ${position_size:.2f}, Raw quantity: {quantity}")
        quantity = round(quantity, quantity_precision)
        min_qty = 0.1
        if quantity < min_qty:
            quantity = min_qty
            logger.log_warning(f"{symbol} ⚠️ Quantity below minimum, using {min_qty}")
        logger.log_info(f"{symbol} ✅ Final quantity: {quantity} (precision: {quantity_precision})")
        return quantity
    except Exception as e:
        logger.log_error(f"Quantity calculation error for {symbol}: {str(e)[:200]}")
        return 0.0


def _calculate_sl_tp(df: pd.DataFrame, settings: Dict, side: str, price: float) -> TradeExit:
    try:
        swing_lookback = int(settings.get("swing_sl_lookback", 5))
        min_sl_distance_pct = float(settings.get("min_sl_distance_pct", 0.005))
        risk_reward_ratio = float(settings.get("risk_reward_ratio", 2.0))
        static_sl_pct = float(settings.get("static_sl_pct", 0.02))
        static_tp_pct = float(settings.get("static_tp_pct", 0.04))
        min_tp_sl_gap_pct = float(settings.get("min_tp_sl_gap_pct", 0.001))  # 0.1%

        buy_len = int(_get_ut(settings, "buy_atr_period", int(settings.get("ut_buy_atr_period", 10))))
        sell_len = int(_get_ut(settings, "sell_atr_period", int(settings.get("ut_sell_atr_period", 10))))
        mult = float(_get_ut(settings, "key_value", float(settings.get("ut_multiplier", 1.0))))

        buy_atr_series = ta.atr(df["high"], df["low"], df["close"], length=buy_len)
        sell_atr_series = ta.atr(df["high"], df["low"], df["close"], length=sell_len)
        buy_atr = float(buy_atr_series.iloc[-1]) if not buy_atr_series.empty and not pd.isna(buy_atr_series.iloc[-1]) else price * 0.01
        sell_atr = float(sell_atr_series.iloc[-1]) if not sell_atr_series.empty and not pd.isna(sell_atr_series.iloc[-1]) else price * 0.01

        buy_trailing_stop = price - mult * buy_atr
        sell_trailing_stop = price + mult * sell_atr

        if bool(settings.get("use_dynamic_sl_tp", True)):
            if side == "LONG":
                swing_low = float(df["low"].iloc[-swing_lookback:].min())
                raw_sl_pct = abs((price - swing_low) / price)
                sl_pct = max(raw_sl_pct, min_sl_distance_pct)
                sl = price * (1 - sl_pct)
                tp_pct = sl_pct * risk_reward_ratio
                tp = price * (1 + tp_pct)
                if sl >= price or tp <= price:
                    logger.log_warning("Dynamic SL/TP invalid for LONG, using static")
                    sl = price * (1 - static_sl_pct)
                    tp = price * (1 + static_tp_pct)
                    sl_pct = static_sl_pct
                    tp_pct = static_tp_pct
            else:
                swing_high = float(df["high"].iloc[-swing_lookback:].max())
                raw_sl_pct = abs((swing_high - price) / price)
                sl_pct = max(raw_sl_pct, min_sl_distance_pct)
                sl = price * (1 + sl_pct)
                tp_pct = sl_pct * risk_reward_ratio
                tp = price * (1 - tp_pct)
                if sl <= price or tp >= price:
                    logger.log_warning("Dynamic SL/TP invalid for SHORT, using static")
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
            else:
                sl = price * (1 + static_sl_pct)
                tp = price * (1 - static_tp_pct)
                sl_pct = static_sl_pct
                tp_pct = static_tp_pct

        # Sanity: keep SL/TP on the correct side of price
        if side == "LONG":
            if sl >= price:
                sl = price * (1 - max(min_sl_distance_pct, 0.001))
                logger.log_warning("LONG SL forced below price")
            if tp <= price:
                tp = price * (1 + max(min_sl_distance_pct, 0.001))
                logger.log_warning("LONG TP forced above price")
        else:
            if sl <= price:
                sl = price * (1 + max(min_sl_distance_pct, 0.001))
                logger.log_warning("SHORT SL forced above price")
            if tp >= price:
                tp = price * (1 - max(min_sl_distance_pct, 0.001))
                logger.log_warning("SHORT TP forced below price")

        # Guard: ensure TP and SL are sufficiently apart
        gap = abs(tp - sl) / max(price, 1e-9)
        if gap < min_tp_sl_gap_pct:
            logger.log_warning(f"TP/SL gap {gap:.6f} < min {min_tp_sl_gap_pct:.6f}; widening targets")
            if side == "LONG":
                sl = min(sl, price * (1 - min_tp_sl_gap_pct))
                tp = max(tp, price * (1 + min_tp_sl_gap_pct))
            else:
                sl = max(sl, price * (1 + min_tp_sl_gap_pct))
                tp = min(tp, price * (1 - min_tp_sl_gap_pct))

        partial = price + (price - sl) if side == "LONG" else price - (sl - price)
        trailing = buy_trailing_stop if side == "LONG" else sell_trailing_stop
        if side == "LONG" and trailing >= price:
            trailing = price * 0.995
        elif side == "SHORT" and trailing <= price:
            trailing = price * 1.005

        logger.log_info(f"{side} SL/TP calculated: entry={price}, sl={sl}, tp={tp}, trailing={trailing}")
        return TradeExit(trailing_stop=trailing, sl=sl, tp=tp, sl_pct=sl_pct, tp_pct=tp_pct, partial_tp=partial, partial_size=0.5)

    except Exception as e:
        logger.log_error(f"SL/TP calculation error: {str(e)[:200]}")
        if side == "LONG":
            return TradeExit(price * 0.99, price * 0.98, price * 1.04, 0.02, 0.04)
        else:
            return TradeExit(price * 1.01, price * 1.02, price * 0.96, 0.02, 0.04)

    except Exception as e:
        logger.log_error(f"SL/TP calculation error: {str(e)[:200]}")
        if side == "LONG":
            return TradeExit(price * 0.99, price * 0.98, price * 1.04, 0.02, 0.04)
        else:
            return TradeExit(price * 1.01, price * 1.02, price * 0.96, 0.02, 0.04)

# -----------------------------
# Entry evaluation (with explicit filter logging)
# -----------------------------

def evaluate_scalper_entry(df: Union[pd.DataFrame, str], settings: Dict) -> Tuple[Optional[str], Optional[TradeExit]]:
    try:
        df = _ensure_dataframe(df)
        if df.empty:
            return None, None

        symbol = settings.get("symbol", "")
        open_trades = load_open_trades()

        last_ts = df["timestamp"].iloc[-1]
        ts = pd.to_datetime(last_ts, utc=True)
        hour_utc = int(ts.hour)

        # --- Time filter ---
        use_time = bool(settings.get("filters", {}).get("use_time_filter", False))
        # support timezone offset in minutes (e.g., 330 for IST)
        tz_off_min = int(settings.get("trading_hours_tz_offset_min", 0) or 0)
        local_ts = ts + pd.Timedelta(minutes=tz_off_min)
        local_hour = int(local_ts.hour)
        if use_time:
            start_hour, end_hour = settings.get("allowed_trading_hours", [0, 24])
            in_window = start_hour <= local_hour < end_hour
            logger.log_debug(f"{symbol} ⏰ Time filter: UTC={hour_utc}, local={local_hour} (offset {tz_off_min} min), window=[{start_hour},{end_hour}) => {'PASS' if in_window else 'BLOCK'}")
            if not in_window:
                return None, None

        # --- Calculate UT signals ---
        df = calculate_ut_signals(df, settings)

        # --- Trend filter (EMA) ---
        use_trend = bool(settings.get("filters", {}).get("use_trend_filter", False))
        if use_trend:
            ema_period = int(settings.get("ema_filter_period", 200))
            ema_val = df["close"].ewm(span=ema_period).mean().iloc[-1]
            px = float(df["close"].iloc[-1])
            buy_sig = df["ut_buy_signal"].iloc[-1] == 1.0
            sell_sig = df["ut_sell_signal"].iloc[-1] == 1.0
            if buy_sig and px < ema_val:
                logger.log_debug(f"{symbol} 📉 Trend filter BLOCK: buy_sig with px<{ema_period}EMA ({px:.6f}<{float(ema_val):.6f})")
                return None, None
            if sell_sig and px > ema_val:
                logger.log_debug(f"{symbol} 📈 Trend filter BLOCK: sell_sig with px>{ema_period}EMA ({px:.6f}>{float(ema_val):.6f})")
                return None, None
            logger.log_debug(f"{symbol} ✅ Trend filter PASS: px={px:.6f}, EMA{ema_period}={float(ema_val):.6f}")

        # --- Min body filter ---
        ok_body, detail = _passes_min_body_filter(df, settings)
        logger.log_debug(f"{symbol} 🧱 Min-body filter: {'PASS' if ok_body else 'BLOCK'}; {detail}")
        if not ok_body:
            return None, None

        last = df.iloc[-1]
        price = float(last["close"])

        side: Optional[str] = None
        sltp: Optional[TradeExit] = None

        if last["ut_buy_signal"] == 1.0:
            side, sltp = "LONG", _calculate_sl_tp(df, settings, "LONG", price)
        elif last["ut_sell_signal"] == 1.0:
            side, sltp = "SHORT", _calculate_sl_tp(df, settings, "SHORT", price)
        else:
            logger.log_debug(f"{symbol} 💤 No UT signal on the last CLOSED candle.")
            return None, None

        if side is None or sltp is None:
            return None, None

        existing_dir = open_trades.get(symbol, {}).get('direction')
        if existing_dir and existing_dir.upper() == side:
            logger.log_debug(f"{symbol} 🔁 Existing {side} position open — skipping new entry.")
            return None, None

        logger.log_info(f"{symbol} ✅ ENTRY decision: {side} at {price} (UTC {ts.isoformat()}) after filters PASS")
        return side, sltp

    except Exception as e:
        logger.log_error(f"Scalper entry evaluation error: {str(e)[:200]}")
        return None, None
