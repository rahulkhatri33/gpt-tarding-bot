import os
import sys
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from typing import List, Optional
from binance.client import Client
from binance.enums import KLINE_INTERVAL_5MINUTE
from core.logger import global_logger as logger
from core.config import CONFIG


class BinanceClient:
    def __init__(self):
        self.client = Client(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )
        self.config = CONFIG
        self._time_offset_ms = 0  # local offset vs Binance

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def sync_time_with_binance(self) -> int:
        """Sync local offset with Binance server time (no system clock change)."""
        try:
            srv = self.client.futures_time()
            server_time = int(srv.get("serverTime", 0))
            local_ms = self._now_ms()
            self._time_offset_ms = local_ms - server_time
            if hasattr(self.client, "TIME_OFFSET"):
                self.client.TIME_OFFSET = int(self._time_offset_ms)
            logger.log_info(
                f"Binance offset sync: server={server_time}, local={local_ms}, offset={self._time_offset_ms}ms"
            )
            return int(self._time_offset_ms)
        except Exception as e:
            logger.log_warning(f"Failed to sync offset with Binance: {e}")
            return int(self._time_offset_ms)

    def _sign(self, params: dict) -> dict:
        """Sign params with API secret."""
        query = urlencode(params)
        secret = os.getenv("BINANCE_API_SECRET")
        signature = hmac.new(
            secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def fetch_klines(
        self, symbol: str, timeframe: str = KLINE_INTERVAL_5MINUTE, limit: int = None
    ) -> List:
        if limit is None:
            limit = self.config["scalper_settings"].get("min_candles", 300)
        try:
            klines = self.client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
            logger.log_debug(f"{symbol} fetched {len(klines)} klines")
            return klines
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to fetch klines: {e}")
            return []

    def get_futures_balance(self) -> Optional[float]:
        """
        Fetch USDT futures wallet balance using manual signing.
        - Always uses adjusted timestamp (local - offset)
        - Adds recvWindow=5000
        - Retries once on -1021
        """
        url = "https://fapi.binance.com/fapi/v2/balance"
        headers = {"X-MBX-APIKEY": os.getenv("BINANCE_API_KEY")}

        for attempt in range(2):
            try:
                ts = int(time.time() * 1000) - int(self._time_offset_ms)
                params = {"timestamp": ts, "recvWindow": 5000}
                signed = self._sign(params)
                r = requests.get(url, headers=headers, params=signed, timeout=5)
                r.raise_for_status()
                data = r.json()
                for asset in data:
                    if asset["asset"] == "USDT":
                        return float(
                            asset.get("balance", asset.get("walletBalance", 0.0))
                        )
                return 0.0
            except requests.HTTPError as e:
                try:
                    j = e.response.json()
                    code = j.get("code")
                    msg = j.get("msg", "")
                except Exception:
                    code = None
                    msg = str(e)

                logger.log_error(
                    f"Failed to fetch futures balance (attempt {attempt+1}): {msg}"
                )
                if code == -1021 or "Timestamp" in msg:
                    logger.log_warning("Timestamp error. Syncing offset with Binance...")
                    self.sync_time_with_binance()
                    time.sleep(0.2)
                    continue
                return None
            except Exception as e:
                logger.log_error(f"Unexpected error fetching balance: {e}")
                return None

        logger.log_error("Exhausted retries fetching futures balance.")
        return None

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                return None
            filters = {f["filterType"]: f for f in info["filters"]}
            return {
                "quantityPrecision": info["quantityPrecision"],
                "pricePrecision": info["pricePrecision"],
                "minQuantity": float(filters["LOT_SIZE"]["minQty"]),
            }
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to fetch symbol info: {e}")
            return None

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to fetch price: {e}")
            return None
