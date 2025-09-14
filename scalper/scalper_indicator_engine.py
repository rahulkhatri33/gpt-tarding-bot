import pandas as pd
import numpy as np
from core.logger import global_logger as logger
from utils.notifier import Notifier, notifier
from scalper import scalper_runner

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def enrich_dataframe(symbol: str, df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    try:
        if not isinstance(df, pd.DataFrame):
            if scalper_runner.shutdown_flag.is_set():
                logger.log_info(f"🛑 Shutdown detected during indicator enrichment for {symbol} — exiting gracefully.")
                return pd.DataFrame()
            logger.log_error(f"❌ Indicator enrichment failed for {symbol}: Not a DataFrame.")
            notifier.send_critical(f"❌ Indicator enrichment error: Received non-DataFrame input for {symbol}.")
            return pd.DataFrame()

        df = df.copy()
        if len(df) < 5:
            logger.log_warning(f"{symbol} 📉 Not enough data to enrich indicators (min 5 rows).")
            return pd.DataFrame()

        # Only compute indicators needed for scalper_strategy.py (UT Bot and STC)
        # UT Bot and STC are computed in scalper_strategy.py, so minimal enrichment needed
        df['ATR_14'] = compute_atr(df)

        if dropna:
            df = df.dropna(subset=['ATR_14']).copy()

        df.reset_index(drop=True, inplace=True)
        logger.log_debug(f"{symbol} Enriched DataFrame: {df.iloc[-1][['open', 'high', 'low', 'close', 'volume', 'ATR_14']].to_dict()}")
        return df

    except Exception as e:
        logger.log_error(f"❌ Indicator enrichment failed for {symbol}: {e}")
        notifier.send_critical(f"❌ Indicator enrichment failed for {symbol}: {e}")
        return pd.DataFrame()