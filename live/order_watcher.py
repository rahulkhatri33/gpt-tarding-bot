import time
from typing import Optional
from core.logger import global_logger as logger
from core.position_manager import position_manager
from live.exit_manager import full_exit
from utils.price_fetcher import get_latest_price
from core.config import get_heartbeat_timeout_sec, get_watchdog_poll_interval_sec
from core import order_tracker
from scalper.sl_tracker import record_scalper_sl_hit

_last_heartbeat = time.time()

def update_heartbeat() -> None:
    """Updates system heartbeat timestamp."""
    global _last_heartbeat
    _last_heartbeat = time.time()

def order_monitor() -> None:
    """Monitors active orders and triggers exit if SL/TP/Trailing conditions are met."""
    while True:
        for key, pos in list(position_manager.get_all_positions().items()):
            if not all(k in pos for k in ["symbol", "direction", "stop_loss", "take_profit"]):
                logger.log_error(f"Invalid position {key}: Missing required fields {pos}")
                position_manager.remove_position(key)
                continue

            symbol = pos["symbol"]
            direction = pos["direction"]
            sl = pos["stop_loss"]
            tp = pos["take_profit"]
            trail_active = pos.get("trail_active", False)
            peak_price = pos.get("peak_price", pos["entry_price"])
            exit_pending = pos.get("exit_pending", False)

            if exit_pending:
                continue

            price = None
            for _ in range(2):
                price = get_latest_price(symbol)
                if price:
                    break
                time.sleep(0.5)
            if price is None:
                logger.log_warning(f"⚠️ Cannot fetch price for {symbol}. Skipping this cycle.")
                continue

            try:
                if not trail_active and (
                    (direction == "long" and price <= sl) or
                    (direction == "short" and price >= sl)
                ):
                    logger.log_info(f"🚨 {symbol} SL HIT detected.")
                    if order_tracker.is_exit_pending(symbol, direction):
                        logger.log_debug(f"{symbol}-{direction} ⏳ Exit already pending (order_monitor).")
                        continue
                    full_exit(symbol, direction, price, reason="SL")

                    if pos.get("source") == "5M_SCALPER" and not pos.get("tp1_triggered", False):
                        record_scalper_sl_hit()

                    continue

                if direction == "long" and price > peak_price:
                    position_manager.set_peak_price(symbol, direction, price)
                elif direction == "short" and price < peak_price:
                    position_manager.set_peak_price(symbol, direction, price)

                if trail_active:
                    current_pos = pos
                    trail_sl = current_pos.get("trailing_sl", sl)

                    if direction == "long" and price <= trail_sl:
                        logger.log_info(f"🚨 {symbol} TRAILING SL HIT.")
                        if order_tracker.is_exit_pending(symbol, direction):
                            logger.log_debug(f"{symbol}-{direction} ⏳ Exit already pending (order_monitor).")
                            continue
                        full_exit(symbol, direction, price, reason="TRAILING")
                        continue

                    elif direction == "short" and price >= trail_sl:
                        logger.log_info(f"🚨 {symbol} TRAILING SL HIT.")
                        if order_tracker.is_exit_pending(symbol, direction):
                            logger.log_debug(f"{symbol}-{direction} ⏳ Exit already pending (order_monitor).")
                            continue
                        full_exit(symbol, direction, price, reason="TRAILING")
                        continue

            except Exception as e:
                logger.log_critical(f"❌ Order monitor error for {symbol}: {e}")

        update_heartbeat()
        time.sleep(2)

def time_exit_loop() -> None:
    """Forces exit if hold time limit exceeded for any open position."""
    while True:
        now = time.time()
        for key, pos in position_manager.get_all_positions().items():
            if not all(k in pos for k in ["symbol", "direction", "stop_loss", "take_profit"]):
                logger.log_error(f"Invalid position {key}: Missing required fields {pos}")
                position_manager.remove_position(key)
                continue

            exit_time = pos.get("exit_time", None)
            if exit_time and now >= exit_time:
                symbol = pos["symbol"]
                direction = pos["direction"]
                try:
                    price = get_latest_price(symbol)
                    if price:
                        logger.log_info(f"⏰ {symbol} TIME EXIT reached.")
                        full_exit(symbol, direction, price, reason="TIME_EXIT")
                except Exception as e:
                    logger.log_critical(f"❌ Time exit error for {symbol}: {e}")

        time.sleep(30)

def watchdog_loop() -> None:
    """Monitors heartbeat activity. If heartbeat stale, triggers REST fallback checks."""
    while True:
        now = time.time()
        heartbeat_age = now - _last_heartbeat

        if heartbeat_age > get_heartbeat_timeout_sec():
            logger.log_critical(f"❌ Heartbeat timeout detected ({heartbeat_age:.2f}s). Initiating REST fallback.")

            for key, pos in position_manager.get_all_positions().items():
                if not all(k in pos for k in ["symbol", "direction", "stop_loss", "take_profit"]):
                    logger.log_error(f"Invalid position {key}: Missing required fields {pos}")
                    position_manager.remove_position(key)
                    continue

                symbol = pos["symbol"]
                direction = pos["direction"]
                sl = pos["stop_loss"]
                tp = pos["take_profit"]
                trail_active = pos.get("trail_active", False)
                trail_sl = pos.get("trailing_sl", sl)

                try:
                    price = get_latest_price(symbol)
                    if not price:
                        continue

                    if not trail_active:
                        if (direction == "long" and price <= sl) or (direction == "short" and price >= sl):
                            if order_tracker.is_exit_pending(symbol, direction):
                                logger.log_debug(f"{symbol}-{direction} ⏳ Exit already pending (watchdog SL).")
                                continue
                            full_exit(symbol, direction, price, reason="REST_EXIT_SL")
                            continue
                    else:
                        if (direction == "long" and price <= trail_sl) or (direction == "short" and price >= trail_sl):
                            if order_tracker.is_exit_pending(symbol, direction):
                                logger.log_debug(f"{symbol}-{direction} ⏳ Exit already pending (watchdog trail).")
                                continue
                            full_exit(symbol, direction, price, reason="REST_EXIT_TRAILING")
                            continue

                    if (direction == "long" and price >= tp) or (direction == "short" and price <= tp):
                        if order_tracker.is_exit_pending(symbol, direction):
                            logger.log_debug(f"{symbol}-{direction} ⏳ Exit already pending (watchdog TP).")
                            continue
                        full_exit(symbol, direction, price, reason="REST_EXIT_TP")
                        continue

                except Exception as e:
                    logger.log_critical(f"❌ Watchdog fallback error for {symbol}: {e}")

        time.sleep(get_watchdog_poll_interval_sec())