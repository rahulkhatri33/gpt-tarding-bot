# engine/indicator_engine.py

import pandas as pd
import numpy as np
from core.logger import global_logger as logger
from utils.notifier import Notifier, notifier

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Computes the Exponential Moving Average (EMA) for a given series and period.
    """
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Computes the Average True Range (ATR) over the specified period.
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes the Relative Strength Index (RSI) over the specified period.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_ema_trend_score(ema3: float, ema10: float, ema50: float) -> int:
    """
    Calculates the EMA trend score based on EMA3, EMA10, and EMA50.
    +1 if EMA3 > EMA10, +1 if EMA3 > EMA50, +1 if EMA10 > EMA50.
    """
    score = 0
    if ema3 > ema10:
        score += 1
    if ema3 > ema50:
        score += 1
    if ema10 > ema50:
        score += 1
    return score


def enrich_indicators(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Adds EMA, RSI, ATR, volume, and candlestick structure features to the DataFrame."""
    try:
        df = df.copy()

        if len(df) < 5:
            logger.log_warning("📉 Not enough data to enrich indicators (minimum 5 rows required).")
            return pd.DataFrame()

        # === EMA indicators ===
        df["ema_7"] = df["close"].ewm(span=7, adjust=False).mean()
        df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()
        df["ema_7_slope"] = df["ema_7"].diff()
        df["ema_14_slope"] = df["ema_14"].diff()
        df["ema_divergence"] = df["ema_7"] - df["ema_14"]
        df["ema_divergence_slope"] = df["ema_divergence"].diff()
        df["ema_7_prev"] = df["ema_7"].shift(1)
        df["ema_14_prev"] = df["ema_14"].shift(1)

        # === RSI 7 + slope + pivots ===
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=7).mean()
        avg_loss = pd.Series(loss).rolling(window=7).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_7"] = 100 - (100 / (1 + rs))
        df["rsi_7_prev"] = df["rsi_7"].shift(1)
        df["rsi_7_slope"] = df["rsi_7"] - df["rsi_7_prev"]

        df["rsi_pivot_up"] = (
            (df["rsi_7"].shift(2) < df["rsi_7"].shift(1)) & 
            (df["rsi_7"].shift(1) < df["rsi_7"])
        ).astype(int)
        df["rsi_pivot_down"] = (
            (df["rsi_7"].shift(2) > df["rsi_7"].shift(1)) & 
            (df["rsi_7"].shift(1) > df["rsi_7"])
        ).astype(int)

        # === Volume indicators ===
        df["volume_slope"] = df["volume"].diff()
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()

        # === ATR (5) ===
        hl = df["high"] - df["low"]
        hc = abs(df["high"] - df["close"].shift())
        lc = abs(df["low"] - df["close"].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr_5"] = tr.rolling(window=5).mean()

        # === Candle structure ===
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"] + 1e-8
        df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"]
        df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"]
        df["body_to_range"] = df["body_size"] / df["candle_range"]

        df["hammer_like"] = (
            (df["lower_wick_ratio"] > 0.5) &
            (df["body_to_range"] < 0.3)
        ).astype(int)

        df["engulfing"] = (
            (df["close"] > df["open"]) &
            (df["close"].shift(1) < df["open"].shift(1)) &
            (df["close"] > df["open"].shift(1)) &
            (df["open"] < df["close"].shift(1))
        ).astype(int)

        if dropna:
            df.dropna(inplace=True)

        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.log_error(f"❌ Indicator enrichment failed: {e}")
        notifier.send_critical(f"❌ IndicatorEngine: Failed during enrich_indicators execution.\nError: {e}")
        return pd.DataFrame()  # Graceful return to avoid crash
