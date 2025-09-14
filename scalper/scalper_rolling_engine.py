import pandas as pd
from typing import Dict
from core.logger import global_logger as logger
from core.config import CONFIG
import threading

class RollingEngine:
    """Manages rolling candle data for scalper."""
    def __init__(self):
        self.candles: Dict[str, pd.DataFrame] = {}
        self.maxlen = CONFIG["scalper_settings"].get("min_candles", 300)
        self._lock = threading.Lock()
        logger.log_debug(f"RollingEngine initialized with maxlen={self.maxlen}")

    def update_candles(self, symbol: str, df: pd.DataFrame) -> None:
        """Update candle data for a symbol."""
        try:
            with self._lock:
                logger.log_debug(
                    f"{symbol} Updating candles: input df size={len(df)}, "
                    f"current cache size={len(self.candles.get(symbol, pd.DataFrame()))}, "
                    f"maxlen={self.maxlen}, "
                    f"df_columns={df.columns.tolist() if not df.empty else 'empty'}"
                )
                if symbol not in self.candles:
                    self.candles[symbol] = df
                else:
                    combined = pd.concat([self.candles[symbol], df]).drop_duplicates(subset=['timestamp'])
                    self.candles[symbol] = combined.tail(self.maxlen)
                    logger.log_debug(f"{symbol} After concat and dedupe: {len(combined)} candles, trimmed to {len(self.candles[symbol])}")
                logger.log_info(f"{symbol} 🔁 Rolling cache updated: {len(self.candles[symbol])} candles")
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to update candles: {str(e)}")

    def get_candles(self, symbol: str) -> pd.DataFrame:
        """Retrieve candle data for a symbol."""
        try:
            with self._lock:
                if symbol in self.candles and not self.candles[symbol].empty:
                    logger.log_debug(f"{symbol} Retrieving {len(self.candles[symbol])} candles")
                    return self.candles[symbol]
                logger.log_warning(f"{symbol} 📉 No candle data available.")
                return pd.DataFrame()
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to get candles: {str(e)}")
            return pd.DataFrame()

    def restore_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Restore candle cache from saved data."""
        try:
            from scalper.scalper_candle_listener import fetch_5m_data, convert_klines_to_dataframe
            with self._lock:
                logger.log_debug(
                    f"{symbol} Restoring cache: input df size={len(df)}, "
                    f"maxlen={self.maxlen}, "
                    f"df_columns={df.columns.tolist() if not df.empty else 'empty'}, "
                    f"df_head={df.head(2).to_dict() if not df.empty else 'empty'}, source='external'"
                )
                if df.empty or len(df) < self.maxlen:
                    logger.log_warning(f"{symbol} Insufficient candles ({len(df)}), fetching {self.maxlen} candles")
                    klines = fetch_5m_data(symbol, self.maxlen)
                    df = convert_klines_to_dataframe(klines)
                if df.empty:
                    logger.log_warning(f"{symbol} Empty DataFrame provided for cache restoration")
                    self.candles[symbol] = pd.DataFrame()
                else:
                    self.candles[symbol] = df.tail(self.maxlen)
                    logger.log_info(f"{symbol} 🔁 Rolling cache restored: {len(self.candles[symbol])} candles")
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to restore cache: {str(e)}")
            with self._lock:
                self.candles[symbol] = pd.DataFrame()

    def save_all(self):
        """Save all candle data to disk."""
        try:
            with self._lock:
                for symbol, df in self.candles.items():
                    logger.log_debug(f"{symbol} Saving {len(df)} candles to cache")
                    # Placeholder for cache saving logic
        except Exception as e:
            logger.log_error(f"Failed to save candle cache: {str(e)}")

scalper_rolling = RollingEngine()
