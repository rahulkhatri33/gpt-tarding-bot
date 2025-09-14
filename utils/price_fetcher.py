# utils/price_fetcher.py

import os
from typing import Optional
from binance.client import Client
from dotenv import load_dotenv

from core.logger import global_logger as logger

# === Init Binance Client ===
load_dotenv()
client = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

def get_latest_price(symbol: str) -> Optional[float]:
    """
    Fetch the latest price for a given symbol using Binance REST API.
    Returns None if price cannot be retrieved or parsed.
    """
    try:
        data = client.get_symbol_ticker(symbol=symbol)
        price = float(data.get("price", 0))
        return price
    except Exception as e:
        logger.log_once(f"{symbol} ❌ Failed to fetch latest price: {e}", level="ERROR")
        return None
