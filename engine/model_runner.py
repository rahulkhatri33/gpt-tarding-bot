# engine/model_runner.py

import time
from core.logger import global_logger as logger
from core.config import (
    CONFIG,
    get_usd_allocation,
    is_dry_run_enabled,
    get_max_concurrent_trades_by_source
)

from datetime import datetime
from core.position_manager import position_manager
from core.symbol_precision import symbol_precision
from core.trade_executor import execute_trade
from engine.indicator_engine import enrich_indicators
from engine.entry_engine import should_enter
from engine.rolling_engine import RollingEngine
from core.candle_cache import candle_cache
from ml_engine.ml_inference.infer_dual_model import infer_dual_model, get_feature_list
from data.atr_cache import ATRCache
from utils.trade_cooldown import is_in_cooldown
from core.config import get_confidence_thresholds
from ml_engine.ml_inference.ml_inference_cache import cache_result

thresholds = get_confidence_thresholds()
atr_cache = ATRCache()
BASE_PAIRS = CONFIG["base_pairs"]
rolling_engine = RollingEngine(BASE_PAIRS)

def get_last_closed_1h_candle(client, symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval="1h", limit=2)
        return klines[-2]
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to fetch 1H candle via REST: {e}")
        return None
    
def _get_threshold(pair: str, direction: str) -> float:
    thresholds = get_confidence_thresholds()
    pair_conf = thresholds.get(pair, {})
    return pair_conf.get(direction, thresholds.get("default", 0.60))

def run_hourly_model(client):
    modified = False
    btc_enriched = None
    trade_candidates = []
    max_trades = get_max_concurrent_trades_by_source("ML")

    for pair in BASE_PAIRS:
        try:
            last_candle = get_last_closed_1h_candle(client, pair)
            if not last_candle:
                continue

            close_time = int(last_candle[6])
            if not candle_cache.should_process(pair, close_time, timeframe="1h"):
                continue


            rolling_engine.update(pair, {
                "timestamp": close_time,
                "open": float(last_candle[1]),
                "high": float(last_candle[2]),
                "low": float(last_candle[3]),
                "close": float(last_candle[4]),
                "volume": float(last_candle[5])
            })

            modified = True
            df = rolling_engine.get_df(pair)
            if df is None or len(df) < 50:
                logger.log_debug(f"{pair} ⚠️ Insufficient data for enrichment.")
                continue

            enriched = enrich_indicators(df)
            if enriched.empty:
                logger.log_error(f"{pair} ❌ Enrichment failed.")
                continue

            if pair == "BTCUSDT":
                enriched["alt_btc_ratio"] = 1.0
            else:
                if btc_enriched is None:
                    btc_df = rolling_engine.get_df("BTCUSDT")
                    if btc_df is None or len(btc_df) < len(enriched):
                        logger.log_error(f"{pair} ❌ BTC enrichment failed for alt_btc_ratio.")
                        continue
                    btc_enriched = enrich_indicators(btc_df).reset_index(drop=True)
                enriched = enriched.reset_index(drop=True)
                enriched["alt_btc_ratio"] = enriched["close"] / btc_enriched["close"]

            latest = enriched.iloc[-1]
            high, low, close = latest["high"], latest["low"], latest["close"]
            atr = max(high - low, abs(high - close), abs(low - close))
            atr_cache.update_atr(pair, atr)

            for direction in ["long", "short"]:
                try:
                    feature_list = get_feature_list(pair, direction)
                    features_only = enriched[feature_list].iloc[[-1]].copy()

                    if features_only.isnull().any().any():
                        missing = features_only.columns[features_only.isnull().any()].tolist()
                        logger.log_error(f"{pair}-{direction} ❌ Missing features: {missing}")
                        continue

                    # ✅ Always re-infer and overwrite cache on every 1H close
                    result = infer_dual_model(pair, features_only, direction)
                    cache_result(pair, direction, result["label"], result["confidence"])
                    from ml_engine.ml_inference.ml_inference_cache import save_cache
                    save_cache()
                    threshold = thresholds.get(pair, {}).get(direction, thresholds.get("default", 0.6))
                  
                    if not position_manager.can_open_trade(pair, max_trades):
                        logger.log_debug(f"{pair} ⛔ {direction.upper()} blocked — max trades or already active.")
                        break

                    if should_enter(result['label'], result['confidence'], pair, direction):
                        trade_candidates.append({
                            "pair": pair,
                            "direction": direction,
                            "confidence": result["confidence"],
                            "label": result["label"],
                            "price": close
                        })
                        break  # One direction per symbol
                    else:
                        logger.log_debug(f"{pair} [{direction.upper()}] 🚫 Rejected | Label: {result['label']} | Conf: {result['confidence']:.4f}")

                except Exception as e:
                    logger.log_error(f"{pair}-{direction} ❌ Inference error: {e}")
                    continue
            candle_cache.mark_processed(pair, close_time, timeframe="1h")

        except Exception as e:
            logger.log_error(f"{pair} ❌ Model runner exception: {e}")

        time.sleep(1)

    trade_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    open_count = position_manager.get_open_trade_count()
    available_slots = max_trades - open_count

    if trade_candidates:
        logger.log_info(f"🎯 Submitting {len(trade_candidates[:available_slots])} ML trades to Gatekeeper.")

        trade_requests = []
        for trade in trade_candidates[:available_slots]:
            symbol = trade["pair"]
            direction = trade["direction"]
            price = trade["price"]

            if is_in_cooldown(symbol, direction, source="ML"):
                logger.log_debug(f"{symbol} ⏳ ML cooldown active for {direction.upper()} — skipping.")
                continue

            if price <= 0:
                logger.log_error(f"{symbol} ❌ Invalid price: {price} — skipping.")
                continue

            quantity = get_usd_allocation(symbol, source="ML") / price
            trade_requests.append({
                "symbol": symbol,
                "direction": direction,
                "confidence": trade["confidence"],
                "label": trade["label"],
                "entry_price": price,
                "quantity": quantity,
                "source": "ML",
                "timestamp": datetime.now().astimezone().isoformat()  # ✅ Timestamp patched for ML entry tracking
            })
        from engine.gatekeeper import submit_trade_requests
        submit_trade_requests(trade_requests)
