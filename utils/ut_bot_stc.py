# utils/ut_bot_stc.py
import pandas as pd
import numpy as np
import talib

def compute_ut_bot_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute UT Bot signals with parameters from the video
    """
    # UT Bot Buy settings (key=2, atr_period=1)
    df['ut_buy_atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=1)
    df['ut_buy_signal'] = np.where(
        df['close'] > (df['close'].shift(1) + 2 * df['ut_buy_atr']),
        1, 0
    )
    
    # UT Bot Sell settings (key=2, atr_period=300)
    df['ut_sell_atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=300)
    df['ut_sell_signal'] = np.where(
        df['close'] < (df['close'].shift(1) - 2 * df['ut_sell_atr']),
        1, 0
    )
    
    return df

def compute_stc_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute STC oscillator with custom settings (length=80, fast_length=227)
    """
    # Simplified STC calculation (actual implementation may be more complex)
    df['stc_fast_ema'] = df['close'].ewm(span=227).mean()
    df['stc_slow_ema'] = df['close'].ewm(span=80).mean()
    df['stc_line'] = 100 * ((df['stc_fast_ema'] - df['stc_slow_ema']) / df['stc_slow_ema'])
    df['stc_signal'] = df['stc_line'].rolling(window=10).mean()
    
    # Determine direction and zones
    df['stc_direction'] = np.where(df['stc_line'] > df['stc_line'].shift(1), 1, -1)
    df['stc_buy_zone'] = df['stc_line'] < df['stc_signal']
    df['stc_sell_zone'] = df['stc_line'] > df['stc_signal']
    
    return df