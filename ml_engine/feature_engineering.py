# ml_engine/feature_engineering.py

import pandas as pd
import numpy as np

def extract_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Extract the final aligned features used in live inference or model training.
    Must be consistent with `indicator_engine.py` and match `top_features.json`.
    """
    df = df.copy()

    # === Required base columns ===
    required_cols = ["open", "high", "low", "close", "volume", "alt_btc_ratio"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # === EMAs and Divergences ===
    df["ema_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()
    df["ema_7_slope"] = df["ema_7"].diff()
    df["ema_14_slope"] = df["ema_14"].diff()
    df["ema_7_prev"] = df["ema_7"].shift(1)
    df["ema_14_prev"] = df["ema_14"].shift(1)
    df["ema_divergence"] = df["ema_7"] - df["ema_14"]
    df["ema_divergence_slope"] = df["ema_divergence"].diff()

    # === RSI 7 and derived features ===
    df["rsi_7"] = compute_rsi(df["close"], window=7)
    df["rsi_7_prev"] = df["rsi_7"].shift(1)
    df["rsi_7_slope"] = df["rsi_7"] - df["rsi_7_prev"]
    df["rsi_pivot_up"] = ((df["rsi_7"].shift(2) < df["rsi_7"].shift(1)) & (df["rsi_7"].shift(1) < df["rsi_7"])).astype(int)
    df["rsi_pivot_down"] = ((df["rsi_7"].shift(2) > df["rsi_7"].shift(1)) & (df["rsi_7"].shift(1) > df["rsi_7"])).astype(int)

    # === Volume dynamics ===
    df["volume_slope"] = df["volume"].diff()
    df["volume_ma_5"] = df["volume"].rolling(window=5).mean()

    # === ATR ===
    df["atr_5"] = compute_atr(df, window=5)

    # === Candle structure features ===
    df["body_size"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"] + 1e-8
    df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"]
    df["body_to_range"] = df["body_size"] / df["candle_range"]

    # === Patterns ===
    df["hammer_like"] = ((df["lower_wick_ratio"] > 0.5) & (df["body_to_range"] < 0.3)).astype(int)
    df["engulfing"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] > df["open"].shift(1)) &
        (df["open"] < df["close"].shift(1))
    ).astype(int)

    # === Final Cleanup ===
    if dropna:
        df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def compute_rsi(series: pd.Series, window: int = 7) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int = 5) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()
