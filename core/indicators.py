# core/indicators.py

import pandas as pd
import numpy as np
from core.logger import global_logger as logger

def calculate_ut_signals(df: pd.DataFrame, buy_atr_period: int, sell_atr_period: int, multiplier: float, relax_cross: bool):
    """
    Calculate UT Bot signals for buy (ATR period 1) and sell (ATR period 300).
    Args:
        df: DataFrame with OHLC data
        buy_atr_period: ATR period for buy signals (1)
        sell_atr_period: ATR period for sell signals (300)
        multiplier: ATR multiplier (2)
        relax_cross: If False, require strict price crossing for signals
    Returns:
        DataFrame with buy/sell trailing stops and signals
    """
    try:
        df = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # Calculate ATR
        atr_buy = calculate_atr(df, buy_atr_period)
        atr_sell = calculate_atr(df, sell_atr_period)

        # Initialize trailing stops and signals
        buy_trailing_stop = pd.Series(index=df.index, dtype=float)
        sell_trailing_stop = pd.Series(index=df.index, dtype=float)
        buy_signal = pd.Series(0, index=df.index, dtype=float)
        sell_signal = pd.Series(0, index=df.index, dtype=float)

        for i in range(1, len(df)):
            # Buy trailing stop (lower)
            buy_trailing_stop.iloc[i] = max(
                low.iloc[i] - atr_buy.iloc[i] * multiplier,
                buy_trailing_stop.iloc[i-1] if not pd.isna(buy_trailing_stop.iloc[i-1]) else low.iloc[i]
            )
            # Sell trailing stop (upper)
            sell_trailing_stop.iloc[i] = min(
                high.iloc[i] + atr_sell.iloc[i] * multiplier,
                sell_trailing_stop.iloc[i-1] if not pd.isna(sell_trailing_stop.iloc[i-1]) else high.iloc[i]
            )

            # Signals
            if relax_cross:
                buy_signal.iloc[i] = 1 if close.iloc[i] > buy_trailing_stop.iloc[i] else 0
                sell_signal.iloc[i] = 1 if close.iloc[i] < sell_trailing_stop.iloc[i] else 0
            else:
                buy_signal.iloc[i] = 1 if close.iloc[i] > buy_trailing_stop.iloc[i] and close.iloc[i-1] <= buy_trailing_stop.iloc[i-1] else 0
                sell_signal.iloc[i] = 1 if close.iloc[i] < sell_trailing_stop.iloc[i] and close.iloc[i-1] >= sell_trailing_stop.iloc[i-1] else 0

        return pd.DataFrame({
            "buy_trailing_stop": buy_trailing_stop,
            "sell_trailing_stop": sell_trailing_stop,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal
        })
    except Exception as e:
        logger.log_error(f"❌ Error calculating UT signals: {e}")
        return pd.DataFrame()

def calculate_atr(df: pd.DataFrame, period: int):
    """
    Calculate Average True Range (ATR).
    Args:
        df: DataFrame with OHLC data
        period: ATR period
    Returns:
        Series with ATR values
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_stc(close: pd.Series, fast_length: int, slow_length: int, signal_period: int):
    """
    Calculate Schaff Trend Cycle (STC) indicator.
    Args:
        close: Series of closing prices
        fast_length: Fast EMA period for MACD (227)
        slow_length: Slow EMA period for MACD (80)
        signal_period: Signal EMA period (9)
    Returns:
        Series with STC values (0-100)
    """
    try:
        close = close.astype(float)
        # Calculate MACD
        ema_fast = close.ewm(span=fast_length, adjust=False).mean()
        ema_slow = close.ewm(span=slow_length, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        # Normalize MACD to 0-100
        window = max(fast_length, slow_length, signal_period)
        macd_min = macd.rolling(window=window).min()
        macd_max = macd.rolling(window=window).max()
        macd_normalized = 100 * (macd - macd_min) / (macd_max - macd_min + 1e-10)  # Avoid division by zero
        
        # Apply signal EMA
        stc = macd_normalized.ewm(span=signal_period, adjust=False).mean()
        
        # Ensure STC is within 0-100
        stc = stc.clip(0, 100).fillna(50)  # Fill NaN with neutral value
        return stc
    except Exception as e:
        logger.log_error(f"❌ Error calculating STC: {e}")
        return pd.Series()