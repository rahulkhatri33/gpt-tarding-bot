import os
import sys

# Get root folder (f_trading_bot/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Optionally change working dir
os.chdir(BASE_DIR)

import time
import threading
from threading import Lock
from binance.client import Client
from utils.discord_logger import send_discord_log
from engine.rolling_engine import RollingEngine
from engine.indicator_engine import enrich_indicators
from core.logger import YogiLogger
from core.config import (
    CONFIG,
    get_usd_allocation,
    is_dry_run_enabled
)
from ml_engine.ml_inference.infer_dual_model import preload_models
from engine.model_runner import run_hourly_model
from core.position_manager import position_manager
from data.atr_cache import ATRCache
from data.init_seed import warm_start_cache
from live.order_watcher import order_monitor
from live.exit_manager import price_poll_exit_loop
from core.logger import global_logger as logger
from core import order_tracker 
from scalper.scalper_runner import run_scalper
from scalper.scalper_candle_listener import scalper_warm_start_cache
from scalper.scalper_candle_listener import on_candle_close as scalper_on_candle_close
from ml_engine.ml_inference.ml_inference_cache import load_cache
from binance_utils import BinanceClient

try:
    from live.recover_open_positions import main as recover_positions
    logger.log_info("✅ recover_open_positions, successfully imported ")
except ImportError as e:
    logger.log_error(f"❌ Failed to import recover_open_positions: {e}")
    recover_positions = lambda: None

# === INIT ===

client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_API_SECRET")
)

binance_utils = BinanceClient()  # Fixed: Removed api_key and api_secret arguments
atr_cache = ATRCache()
BASE_PAIRS = CONFIG["base_pairs"]
preload_models(BASE_PAIRS)
rolling_engine = RollingEngine(BASE_PAIRS)
run_lock = Lock()

def run_safe_hourly_model(client):
    with run_lock:
        run_hourly_model(client)

# === Restart-safe Wrapper ===
def run_with_restart(target_fn, label="stream"):
    while True:
        try:
            logger.log_live_feed(f"🔄 {label} thread started.")
            target_fn()
        except Exception as e:
            logger.log_error(f"❌ {label} crashed: {e}. Restarting in 15s...")
            time.sleep(15)

# === Background Threads ===
def start_exit_manager():
    price_poll_exit_loop()

def start_order_monitor():
    order_monitor()

def start_candle_listener():
    from live.candle_listener import stream_1h_closes
    logger.log_info("🧠 Launching 1H ML listener thread...")
    run_with_restart(lambda: stream_1h_closes(BASE_PAIRS, on_candle_close=lambda _: run_safe_hourly_model(client)), label="1H stream")
    
def start_heartbeat():
    logger.log_info("                                                                        built for profits ...🚀, with precision⏱  ")
    send_discord_log(" ⏱️ firing up and ready for trading...  ")
    while True:
        time.sleep(1800)
        logger.log_live_feed("🛰️ Bot beat check-in.")
        send_discord_log(" 📶 Actively sniffing for profits  ")

# === Config Validation (Minimal) ===
def validate_scalper_config():
    """Ensure scalper config has required values"""
    scalper_settings = CONFIG.get("scalper_settings", {})
    
    # Set defaults if not present
    if "leverage" not in scalper_settings:
        scalper_settings["leverage"] = 20
        logger.log_info("📊 Set default leverage = 20")
    
    if "risk_percentage" not in scalper_settings:
        scalper_settings["risk_percentage"] = 0.01
        logger.log_info("📊 Set default risk_percentage = 0.01")
    
    # Ensure symbol_precisions exists
    if "symbol_precisions" not in scalper_settings:
        scalper_settings["symbol_precisions"] = {}
        logger.log_info("📊 Created empty symbol_precisions dict")
    
    # Ensure each symbol has required precision fields
    base_pairs = CONFIG.get("base_pairs", [])
    for symbol in base_pairs:
        if symbol not in scalper_settings["symbol_precisions"]:
            scalper_settings["symbol_precisions"][symbol] = {}
            logger.log_info(f"📊 Created config entry for {symbol}")
        
        symbol_config = scalper_settings["symbol_precisions"][symbol]
        
        # Set default precision values if missing
        if "leverage" not in symbol_config:
            symbol_config["leverage"] = scalper_settings["leverage"]
            logger.log_info(f"📊 Set {symbol} leverage = {scalper_settings['leverage']}")
        
        if "quantityPrecision" not in symbol_config:
            symbol_config["quantityPrecision"] = 3
            logger.log_info(f"📊 Set {symbol} quantityPrecision = 3")
        
        if "pricePrecision" not in symbol_config:
            symbol_config["pricePrecision"] = 4
            logger.log_info(f"📊 Set {symbol} pricePrecision = 4")
    
    return True

# === Graceful Shutdown ===
def graceful_shutdown():
    logger.log_live_feed("🛑 Manual shutdown triggered — saving rolling cache...")
    from scalper.scalper_rolling_engine import scalper_rolling
    scalper_rolling.save_all()  # ✅ Save 5M scalper cache
    run_scalper.shutdown_flag.set()  # ✅ Set shutdown flag
    rolling_engine.save_all()
    try:
        send_discord_log(" shutdown detected ... going offline ")
    except Exception as e:
        logger.log_error(f"⚠️ Failed to notify Discord on shutdown: {e}")
    logger.log_live_feed("✅ All rolling cache saved. now going... offline.")

# === MAIN ===
if __name__ == "__main__":
    logger.log_info("🚀 Booting 1 Hour Machine Learning models...")

    # Validate config first
    logger.log_info("🔧 Validating configuration...")
    validate_scalper_config()
    
    # Log config summary
    scalper_settings = CONFIG.get("scalper_settings", {})
    logger.log_info(f"📊 Global leverage: {scalper_settings.get('leverage')}x")
    logger.log_info(f"📊 Risk percentage: {scalper_settings.get('risk_percentage')*100}%")

    # Sync positions with Binance at startup
    logger.log_info("Syncing positions with Binance...")
    position_manager.sync_with_binance()

    # === Load ML inference cache ===
    load_cache()  # 🔁 Load ML predictions from disk into memory
    warm_start_cache(BASE_PAIRS, rolling_engine)
    scalper_warm_start_cache()
    
    threading.Thread(target=start_exit_manager, daemon=True).start()
    threading.Thread(target=start_order_monitor, daemon=True).start()
    threading.Thread(target=start_candle_listener, daemon=True).start()
    threading.Thread(target=start_heartbeat, daemon=True).start()

    try:
        while True:
            logger.log_debug(f"CONFIG in runner.py: {CONFIG}")
            run_scalper()
            time.sleep(15)
    except KeyboardInterrupt:
        graceful_shutdown()