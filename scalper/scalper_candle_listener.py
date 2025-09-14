import os
import time
import pandas as pd
from typing import Callable, List
from datetime import timezone
from binance.client import Client
from binance import ThreadedWebsocketManager
from core.logger import global_logger as logger
from core.config import CONFIG
from scalper.scalper_rolling_engine import scalper_rolling

client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_API_SECRET"),
)

def fetch_5m_data(symbol: str, limit: int) -> List:
    """Fetch 5m candle data from Binance."""
    try:
        max_limit = 1000  # Binance API max limit per request
        klines = []
        remaining = limit
        end_time = client.get_server_time()['serverTime']
        
        while remaining > 0:
            fetch_limit = min(remaining, max_limit)
            logger.log_debug(f"{symbol} Fetching {fetch_limit} 5m candles, remaining={remaining}, end_time={end_time}")
            batch = client.get_klines(
                symbol=symbol,
                interval='5m',
                limit=fetch_limit,
                endTime=end_time
            )
            if not batch:
                logger.log_warning(f"{symbol} No candles fetched in batch")
                break
            klines = batch + klines  # Prepend to maintain chronological order
            remaining -= len(batch)
            end_time = int(batch[0][0]) - 1  # Set end_time to earliest timestamp - 1ms
            logger.log_debug(f"{symbol} Fetched {len(batch)} candles, total={len(klines)}, remaining={remaining}")
            if len(batch) < fetch_limit:
                break  # No more data available
        logger.log_info(f"{symbol} Fetched {len(klines)} 5m candles with requested limit={limit}")
        return klines
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to fetch 5m data: {str(e)}")
        return []

def convert_klines_to_dataframe(klines: List) -> pd.DataFrame:
    """Convert Binance klines to DataFrame."""
    try:
        if not klines:
            logger.log_warning("No klines provided for DataFrame conversion")
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        logger.log_debug(f"Converted klines to DataFrame: rows={len(df)}, columns={df.columns.tolist()}")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.log_error(f"Failed to convert klines to DataFrame: {str(e)}")
        return pd.DataFrame()

def scalper_warm_start_cache() -> None:
    """Warm start the scalper cache with historical 5m candles."""
    base_pairs = CONFIG.get("base_pairs", [])
    min_candles = CONFIG["scalper_settings"].get("min_candles", 300)
    for symbol in base_pairs:
        try:
            logger.log_info(f"{symbol} 🧊 Warming up 5M scalper cache with min_candles={min_candles}...")
            klines = fetch_5m_data(symbol, min_candles)
            if not klines:
                logger.log_warning(f"{symbol} 📉 No 5M candle data retrieved.")
                continue
            df = convert_klines_to_dataframe(klines)
            if df.empty:
                logger.log_warning(f"{symbol} 📊 Empty DataFrame after conversion.")
                continue
            scalper_rolling.update_candles(symbol, df)
            logger.log_info(f"{symbol} ✅ 5M candles loaded: {len(df)}")
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to warm start cache: {str(e)}")

def on_candle_close(symbol: str, interval: str, callback: Callable[[str], None]) -> None:
    """Set up WebSocket to listen for 5m candle closes."""
    twm = ThreadedWebsocketManager(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))
    twm.start()

    def handle_message(msg):
        try:
            if msg.get("e") == "error":
                logger.log_error(f"{symbol} WebSocket error: {msg.get('m')}")
                twm.stop()
                time.sleep(5)
                logger.log_info(f"{symbol} 5M WS Reconnected")
                twm.start_kline_socket(symbol=symbol, interval=interval, callback=handle_message)
                return

            candle = msg["k"]
            if candle["x"]:  # Candle is closed
                df = convert_klines_to_dataframe([candle])
                if not df.empty:
                    scalper_rolling.update_candles(symbol, df)
                    logger.log_debug(f"{symbol} New 5M candle closed: {df.iloc[-1]['close']}")
                    callback(symbol)
        except Exception as e:
            logger.log_error(f"{symbol} WebSocket processing error: {str(e)}")

    twm.start_kline_socket(symbol=symbol, interval=interval, callback=handle_message)
    logger.log_info(f"{symbol} 🕒 5M WebSocket listener started.")

def start_scalper_listeners(callback: Callable[[str], None]) -> None:
    """Start WebSocket listeners for all base pairs."""
    base_pairs = CONFIG.get("base_pairs", [])
    interval = CONFIG["scalper_settings"]["timeframe"]
    for symbol in base_pairs:
        try:
            on_candle_close(symbol, interval, callback)
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to start 5M listener: {str(e)}")