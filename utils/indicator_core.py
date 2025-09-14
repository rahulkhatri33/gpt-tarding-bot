import pandas as pd
import numpy as np


def compute_obv(close_series, volume_series):
    obv = [0]
    for i in range(1, len(close_series)):
        if close_series[i] > close_series[i - 1]:
            obv.append(obv[-1] + volume_series[i])
        elif close_series[i] < close_series[i - 1]:
            obv.append(obv[-1] - volume_series[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close_series.index)  # <- FIXED



def compute_vortex(df: pd.DataFrame, period: int = 14):
    tr = df['high'].combine(df['low'], np.subtract).abs()
    vm_plus = (df['high'] - df['low'].shift()).abs()
    vm_minus = (df['low'] - df['high'].shift()).abs()
    tr_sum = tr.rolling(window=period).sum()
    vp = vm_plus.rolling(window=period).sum()
    vm = vm_minus.rolling(window=period).sum()
    return pd.DataFrame({
        'VORTEX_POS': vp / tr_sum,
        'VORTEX_NEG': vm / tr_sum
    })


def compute_mfi(df: pd.DataFrame, period: int = 14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = mf.where(tp > tp.shift(1), 0)
    neg = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos.rolling(window=period).sum()
    neg_sum = neg.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + (pos_sum / (neg_sum + 1e-9))))
    return mfi


def compute_ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()


def compute_hma(series: pd.Series, period: int):
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    wma_half = series.rolling(window=half_length).apply(lambda x: np.average(x, weights=np.arange(1, half_length + 1)), raw=True)
    wma_full = series.rolling(window=period).apply(lambda x: np.average(x, weights=np.arange(1, period + 1)), raw=True)
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=sqrt_length).apply(lambda x: np.average(x, weights=np.arange(1, sqrt_length + 1)), raw=True)
    return hma


def compute_atr(df: pd.DataFrame, period: int = 14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return mid, upper, lower
def compute_ut_bot(df: pd.DataFrame, key_value: float, atr_period: int):
    """
    Custom UT Bot logic based on ATR and a key multiplier.
    Returns buy/sell signal columns.
    """
    atr = compute_atr(df, period=atr_period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (key_value * atr)
    lower_band = hl2 - (key_value * atr)

    direction = [None]
    buy_signal = [False]
    sell_signal = [False]

    for i in range(1, len(df)):
        if df['close'][i] > upper_band[i - 1]:
            direction.append('buy')
            buy_signal.append(True)
            sell_signal.append(False)
        elif df['close'][i] < lower_band[i - 1]:
            direction.append('sell')
            sell_signal.append(True)
            buy_signal.append(False)
        else:
            direction.append(direction[-1])
            buy_signal.append(False)
            sell_signal.append(False)

    df['ut_direction'] = direction
    df['ut_buy'] = buy_signal
    df['ut_sell'] = sell_signal
    return df


def compute_stc(df: pd.DataFrame, length=80, fast_length=227):
    """
    Compute STC Oscillator approximation.
    """
    macd = compute_ema(df['close'], fast_length) - compute_ema(df['close'], length)
    signal = compute_ema(macd, 15)  # Use standard MACD signal period

    stc_line = compute_ema((macd - signal), 10)  # STC smoothed line

    df['stc'] = stc_line
    return df
