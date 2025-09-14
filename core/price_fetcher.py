# core/price_fetcher.py
from binance.client import Client
from core.logger import global_logger as logger
from utils.exchange import client
import pandas as pd
from datetime import datetime, timezone
import time
from core.config import get_config

def get_price(symbol: str) -> float:
    """
    Fetch the current price for a given symbol from Binance futures.
    Returns the price as a float or 0.0 if the fetch fails.
    """
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker.get("price", 0.0))
        logger.log_info(f"{symbol} ✅ Current price: {price}")
        return price
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to fetch price: {e}")
        return 0.0

def get_recent_klines(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    logger.log_info(f"{symbol} 🧊 Fetching {timeframe} candles...")
    config = get_config()
    retries = config.get("binance", {}).get("api_retries", 3)
    rate_limit_buffer = config.get("binance", {}).get("rate_limit_buffer", 0.5)
    
    for attempt in range(retries):
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit + 417  # Buffer for STC(417) and ATR(300)
            )
            logger.log_debug(f"{symbol} Raw klines last: {klines[-1] if klines else 'Empty'}")
            if not klines or len(klines) < limit:
                logger.log_warning(f"{symbol} ⚠️ Not enough candles: {len(klines)}")
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit + 600
                )
                if not klines or len(klines) < limit:
                    logger.log_error(f"{symbol} ❌ Still not enough candles: {len(klines)}")
                    return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignored"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df[["open_time", "open", "high", "low", "close", "volume"]]
            df = df.astype({
                "open": float, "high": float, "low": float, "close": float, "volume": float
            })
            latest_time = df["open_time"].iloc[-1]
            current_time = datetime.now(timezone.utc)
            time_diff = (current_time - latest_time).total_seconds()
            if time_diff > 600:
                logger.log_warning(f"{symbol} ⚠️ Latest candle too old: {latest_time} UTC, current: {current_time} UTC, diff: {time_diff}s")
            else:
                logger.log_info(f"{symbol} ✅ Latest candle: {latest_time} UTC, current: {current_time} UTC, diff: {time_diff}s")
            logger.log_info(f"{symbol} ✅ {timeframe} candles loaded: {len(df)}")
            return df
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to fetch klines (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(rate_limit_buffer)
            continue
    logger.log_error(f"{symbol} ❌ Failed to fetch klines after {retries} attempts")
    return pd.DataFrame()