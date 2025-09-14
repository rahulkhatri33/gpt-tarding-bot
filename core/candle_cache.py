# core/candle_cache.py

from threading import Lock
from core.logger import global_logger as logger
from utils.notifier import Notifier

notifier = Notifier()

class CandleCache:
    def __init__(self):
        self.last_seen = {}
        self.lock = Lock()

    def should_process(self, symbol: str, ts: int, timeframe: str = "1h") -> bool:
        """
        Returns True if this candle timestamp for the given timeframe has not been processed yet.
        """
        try:
            key = f"{symbol}|{timeframe}"
            with self.lock:
                if self.last_seen.get(key) == ts:
                    logger.log_debug(f"{symbol} ⏩ Duplicate candle TS {ts} for {timeframe} — skipping.")
                    return False
                self.last_seen[key] = ts
                logger.log_debug(f"{symbol} 🟢 New candle TS {ts} for {timeframe} — processing allowed.")
                return True
        except Exception as e:
            logger.log_error(f"{symbol} ❌ CandleCache error in should_process: {e}")
            notifier.send_critical(f"{symbol} ❌ CandleCache: Failure in should_process(). Restart or manual check recommended.\nError: {e}")
            return False  # Safe fallback: don't process to avoid duplication on error

    def mark_processed(self, symbol: str, timestamp: int, timeframe: str = "1h"):
        """
        Explicitly marks a candle timestamp as processed for the given timeframe.
        """
        try:
            key = f"{symbol}|{timeframe}"
            with self.lock:
                self.last_seen[key] = timestamp
                logger.log_debug(f"{symbol} ✅ Marked TS {timestamp} as processed for {timeframe}.")
        except Exception as e:
            logger.log_error(f"{symbol} ❌ CandleCache error in mark_processed: {e}")
            notifier.send_critical(f"{symbol} ❌ CandleCache: Failure in mark_processed(). Manual check advised.\nError: {e}")

# === Singleton Instance ===
candle_cache = CandleCache()
