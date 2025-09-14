# live/candle_listener

import json
import threading
import websocket
import ssl
import os
import certifi
import time
from binance.client import Client
from core.logger import global_logger as logger
from live.order_watcher import update_heartbeat
from engine.rolling_engine import rolling_engine
from ml_engine.feature_engineering import compute_atr
from data.atr_cache import atr_cache
from core.config import CONFIG

# === Init Binance client ===
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_API_SECRET")
)

_stream_started = False
_last_ping = time.time()

# === Callbacks ===
def on_candle_close(client):
    pairs = CONFIG.get("base_pairs", [])
    for symbol in pairs:
        history = rolling_engine.get_df(symbol)
        if history is not None and len(history) >= 14:
            atr_value = compute_atr(history.tail(14))
            if atr_value is not None:
                atr_cache.update_atr(symbol, atr_value)
                logger.log_debug(f"{symbol} 🔄 ATR recalculated and updated at 1H close: {atr_value:.6f}")
            else:
                logger.log_error(f"{symbol} ❌ Failed to calculate ATR on 1H close (ATR calculation returned None)")
        else:
            logger.log_warning(f"{symbol} ⚠️ Insufficient history for ATR calculation on 1H close (need at least 14 candles)")


def on_message(ws, message, on_candle_close):
    global _last_ping
    update_heartbeat()
    _last_ping = time.time()
    try:
        msg = json.loads(message)
        kline = msg.get("data", {}).get("k")
        if kline and kline.get("x"):
            symbol = msg["data"]["s"]
            close_time = kline["T"]
            logger.log_info(f"⏳ 1H candle CLOSED | {symbol} @ {close_time}")
            threading.Thread(target=on_candle_close, args=(client,), daemon=True).start()
    except Exception as e:
        logger.log_error(f"❌ WebSocket message error: {e}")

def on_error(ws, error):
    logger.log_error(f"❌ WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    global _stream_started
    logger.log_error(f"📴 WebSocket closed ({close_status_code}) — {close_msg}")
    _stream_started = False

def on_open(ws):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    logger.log_info(f"📡 [1H WS] Reconnected at {timestamp}")

# === Watchdog ===
def watchdog():
    global _last_ping
    while True:
        time.sleep(10)
        silence = time.time() - _last_ping
        if silence > 150:
            logger.log_error(f"🛑 1H WebSocket silent for {silence:.1f}s — restarting...")
            os._exit(1)

# === Entry Point ===
def stream_1h_closes(pairs, on_candle_close):
    global _stream_started
    if _stream_started:
        logger.log_info("⚠️ 1H WebSocket already running. Ignoring duplicate start.")
        return
    _stream_started = True

    threading.Thread(target=watchdog, daemon=True).start()
    logger.log_info("👁️ WebSocket Watchdog armed for 1H feed...")

    streams = "/".join([f"{pair.lower()}@kline_1h" for pair in pairs])
    url = f"wss://fstream.binance.com/stream?streams={streams}"
    logger.log_once("🔌 Launching Binance WebSocketApp for 1H...")

    ws = websocket.WebSocketApp(
        url,
        on_message=lambda ws, msg: on_message(ws, msg, on_candle_close),
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    ws.run_forever(
        ping_interval=30,
        ping_timeout=10,
        ping_payload="ping",
        sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": certifi.where()}
    )
