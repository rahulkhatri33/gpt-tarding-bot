# engine/rolling_engine.py

import os
import json
import pandas as pd
from collections import deque
from typing import Dict, Optional
from core.logger import global_logger as logger
from utils.notifier import Notifier, notifier


class RollingEngine:
    def __init__(self, pairs, interval: str = "1h", maxlen: int = 100, cache_dir: str = "cache/rolling"):
        self.interval = interval
        self.maxlen = maxlen
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.pair_data: Dict[str, deque] = {
            pair: deque(maxlen=maxlen) for pair in pairs
        }
        self._load_all_pairs(pairs)

    def update(self, pair: str, candle: dict):
        if pair not in self.pair_data:
            self.pair_data[pair] = deque(maxlen=self.maxlen)
        self.pair_data[pair].append(candle)

    def get_df(self, pair: str) -> Optional[pd.DataFrame]:
        if pair not in self.pair_data or len(self.pair_data[pair]) < 48:
            return None
        return pd.DataFrame(list(self.pair_data[pair]))

    def get_latest(self, pair: str) -> Optional[dict]:
        if pair not in self.pair_data or not self.pair_data[pair]:
            return None
        return self.pair_data[pair][-1]

    def save_all(self):
        for pair, candles in self.pair_data.items():
            filename = f"{pair}_{self.interval}.json"
            filepath = os.path.join(self.cache_dir, filename)
            try:
                # Convert timestamps to strings
                safe_candles = []
                for c in candles:
                    safe_c = c.copy()
                    if isinstance(safe_c.get("timestamp"), pd.Timestamp):
                        safe_c["timestamp"] = safe_c["timestamp"].isoformat()
                    safe_candles.append(safe_c)

                with open(filepath, "w") as f:
                    json.dump(safe_candles, f, indent=2)

                logger.log_once(f"{pair} 💾 Rolling cache saved ({len(candles)} entries).")
            except Exception as e:
                logger.log_error(f"{pair} ❌ Failed to save rolling cache: {e}")
                notifier.send_critical(f"❌ RollingEngine: Failed to save cache for {pair}. Error: {e}")

    def _load_all_pairs(self, pairs):
        for pair in pairs:
            filename = f"{pair}_{self.interval}.json"
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        candles = json.load(f)

                    # Rehydrate ISO timestamp strings into pd.Timestamp
                    for c in candles:
                        ts = c.get("timestamp")
                        if isinstance(ts, str):
                            try:
                                c["timestamp"] = pd.Timestamp(ts)
                            except Exception as e:
                                logger.log_warning(f"{pair} ⚠️ Invalid timestamp in cache: {ts} — {e}")

                    self.pair_data[pair] = deque(candles[-self.maxlen:], maxlen=self.maxlen)
                    logger.log_once(f"{pair} 🔁 Rolling cache restored ({len(candles)} candles).")
                except Exception as e:
                    logger.log_error(f"{pair} ⚠️ Failed to load rolling cache: {e}")
                    notifier.send_critical(f"⚠️ RollingEngine: Failed to load cache for {pair}. Error: {e}")

    def get_btc_enriched(self) -> Optional[pd.DataFrame]:
        if self.interval != "1h":
            return None

        btc_df = self.get_df("BTCUSDT")
        if btc_df is None or len(btc_df) < 48:
            return None

        try:
            enriched_btc = btc_df.copy().reset_index(drop=True)
            from engine.indicator_engine import enrich_indicators
            return enrich_indicators(enriched_btc)
        except Exception as e:
            logger.log_error(f"BTCUSDT ❌ Failed to enrich BTC data for alt ratio: {e}")
            notifier.send_critical(f"❌ RollingEngine: Failed to enrich BTC data.\nError: {e}")
            return None


# === Singleton instance for 1H model runner ===
rolling_engine: Optional[RollingEngine] = None

def init_rolling_engine(pairs):
    global rolling_engine
    if rolling_engine is None:
        rolling_engine = RollingEngine(pairs, interval="1h")
