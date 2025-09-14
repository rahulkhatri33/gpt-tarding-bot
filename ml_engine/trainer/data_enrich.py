# ml_engine/trainer/data_enrich.py

import os
import pandas as pd
import numpy as np

RAW_DIR = "data/historical_1h/"
OUT_DIR = "data/enriched_1h/"
ALT_COIN = "BTCUSDT"

os.makedirs(OUT_DIR, exist_ok=True)

def compute_rsi(series: pd.Series, window: int = 7) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(window=window).mean()
    roll_down = pd.Series(down).rolling(window=window).mean()
    rs = roll_up / (roll_down + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_atr(df: pd.DataFrame, window: int = 5) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def enrich(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop unused Binance fields
    df.drop(columns=[
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ], errors='ignore', inplace=True)

    # EMAs
    df["ema_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_7_slope"] = df["ema_7"].diff()
    df["ema_14_slope"] = df["ema_14"].diff()
    df["ema_divergence"] = df["ema_7"] - df["ema_14"]
    df["ema_divergence_slope"] = df["ema_divergence"].diff()
    df["ema_7_prev"] = df["ema_7"].shift(1)
    df["ema_14_prev"] = df["ema_14"].shift(1)
    df["ema_cross_distance"] = df["ema_7"] - df["ema_14"]

    # RSI
    df["rsi_7"] = compute_rsi(df["close"], 7)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_7_prev"] = df["rsi_7"].shift(1)
    df["rsi_7_slope"] = df["rsi_7"] - df["rsi_7_prev"]
    df["rsi_pivot_up"] = ((df["rsi_7"].shift(2) < df["rsi_7"].shift(1)) & (df["rsi_7"].shift(1) < df["rsi_7"])).astype(int)
    df["rsi_pivot_down"] = ((df["rsi_7"].shift(2) > df["rsi_7"].shift(1)) & (df["rsi_7"].shift(1) > df["rsi_7"])).astype(int)

    # Stochastic RSI
    rsi_14 = compute_rsi(df["close"], 14)
    stoch_min = rsi_14.rolling(14).min()
    stoch_max = rsi_14.rolling(14).max()
    stoch_rsi = (rsi_14 - stoch_min) / (stoch_max - stoch_min + 1e-8)
    df["stoch_rsi_k"] = stoch_rsi.rolling(3).mean()
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume
    df["volume_slope"] = df["volume"].diff()
    df["volume_ma_5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / (df["volume_ma_5"] + 1e-8)

    # ATR
    df["atr_5"] = compute_atr(df, 5)
    df["atr_14"] = compute_atr(df, 14)

    # Bollinger Band Width
    df["bb_ma"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_ma"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_ma"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_ma"] + 1e-8)

    # Candle anatomy
    df["body_size"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"] + 1e-8
    df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"]
    df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"]
    df["body_to_range"] = df["body_size"] / df["candle_range"]
    df["hammer_like"] = ((df["lower_wick_ratio"] > 0.5) & (df["body_to_range"] < 0.3)).astype(int)

    # Structure patterns
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    df["is_bearish"] = (df["close"] < df["open"]).astype(int)
    df["marubozu"] = ((df["upper_wick_ratio"] < 0.05) & (df["lower_wick_ratio"] < 0.05)).astype(int)
    df["doji"] = (df["body_to_range"] < 0.1).astype(int)
    df["spinning_top"] = ((df["body_to_range"] < 0.3) &
                          (df["upper_wick_ratio"] > 0.3) &
                          (df["lower_wick_ratio"] > 0.3)).astype(int)
    df["hammer"] = ((df["lower_wick_ratio"] > 0.5) & (df["upper_wick_ratio"] < 0.2)).astype(int)
    df["inverted_hammer"] = ((df["upper_wick_ratio"] > 0.5) & (df["lower_wick_ratio"] < 0.2)).astype(int)
    df["engulfing"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] > df["open"].shift(1)) &
        (df["open"] < df["close"].shift(1))
    ).astype(int)
    df["bearish_engulfing"] = (
        (df["close"] < df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["close"] < df["open"].shift(1)) &
        (df["open"] > df["close"].shift(1))
    ).astype(int)
    df["candle_strength"] = df["body_to_range"] * df["is_bullish"] - df["body_to_range"] * df["is_bearish"]

    # Custom interaction
    df["trend_strength"] = df["ema_divergence"] * df["rsi_7_slope"]

    # Hour of day
    df["time_bucket"] = df["timestamp"].dt.hour

    # BTC RSI
    if "btc_close" in df.columns:
        df["btc_rsi7"] = compute_rsi(df["btc_close"], 7)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_btc_reference() -> pd.DataFrame:
    btc_path = os.path.join(RAW_DIR, "BTCUSDT_1h.csv")
    btc_df = pd.read_csv(btc_path)
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
    btc_df = btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    btc_df["btc_rsi7"] = compute_rsi(btc_df["btc_close"], 7)
    return btc_df

def enrich_all():
    print("🔍 Starting enrichment process...")
    btc_df = load_btc_reference()

    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".csv"):
            continue

        symbol = fname.replace("_1h.csv", "")
        path = os.path.join(RAW_DIR, fname)
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if symbol == ALT_COIN:
            df["alt_btc_ratio"] = 1.0
        else:
            df = pd.merge(df, btc_df, on="timestamp", how="left")
            df["alt_btc_ratio"] = df["close"] / df["btc_close"]
            df["alt_btc_ratio"] = df["alt_btc_ratio"].ffill()  # 🔧 Safe fix

        enriched = enrich(df, btc_df=btc_df)
        enriched.to_csv(os.path.join(OUT_DIR, f"{symbol}_1h_enriched.csv"), index=False)
        print(f"✅ {symbol} enriched → {OUT_DIR}{symbol}_1h_enriched.csv | Shape: {enriched.shape}")

if __name__ == "__main__":
    enrich_all()
