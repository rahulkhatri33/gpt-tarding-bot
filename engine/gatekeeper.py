# engine/gatekeeper.py

from typing import List, Dict

from core.logger import global_logger as logger
from core.position_manager import position_manager
from core.config import get_max_concurrent_trades_by_source
from utils.trade_cooldown import is_in_cooldown
from engine.trade_executor import execute_trade
from core import order_tracker

# === Gatekeeper for Trade Submissions ===

def submit_trade_requests(requests: List[Dict]) -> None:
    """
    Receives a list of trade request dicts.
    Filters, sorts, and submits the best eligible trade for execution.
    """
    if not requests:
        return

    valid_requests = []

    for req in requests:
        try:
            symbol = req["symbol"]
            direction = req["direction"]
            confidence = req["confidence"]
            entry_price = req["entry_price"]
            quantity = req["quantity"]
            source = req.get("source", "unknown")
            override = req.get("override", False)
            label = req.get("label", None)
            timestamp = req.get("timestamp", None)
        except KeyError as e:
            logger.log_error(f"⚠️ Malformed trade request skipped: missing {e}")
            continue

        if position_manager.is_active(symbol, direction):
            logger.log_debug(f"⛔ REJECTED {symbol}-{direction} | Source: {source} | Reason: ACTIVE | Conf: {confidence}")
            continue

        if is_in_cooldown(symbol, direction, source):         
            logger.log_debug(f"⏳ REJECTED {symbol}-{direction} | Source: {source} | Reason: COOLDOWN | Conf: {confidence}")
            continue

        valid_requests.append(req)

    if not valid_requests:
        return

    logger.log_info(f"📋 Lined-up trades for execution ({len(valid_requests)}):")
    for req in valid_requests:
        logger.log_info(
            f"  ↪ {req['symbol']}-{req['direction']} | Conf: {req['confidence']:.2f} | Entry: {req['entry_price']:.4f} | Qty: {req['quantity']:.2f} | Source: {req.get('source', 'N/A')}"
        )

    valid_requests.sort(key=lambda x: (-x["confidence"], x.get("timestamp") or 0))

    source = req.get("source", "unknown")
    max_trades = get_max_concurrent_trades_by_source(source)

    open_trades = len([
        p for p in position_manager.get_all_positions().values()
        if p.get("source", "").lower() == source.lower()
    ])

    for request in valid_requests:
        if open_trades >= max_trades:
            logger.log_info(f"🎯 Executing 0 of {len(valid_requests)} trades — max concurrent trades reached.")
            return

        symbol = request["symbol"]
        direction = request["direction"]
        confidence = request["confidence"]
        entry_price = request["entry_price"]
        quantity = request["quantity"]
        source = request.get("source", "unknown")
        override = request.get("override", False)
        label = request.get("label", None)
        timestamp = request.get("timestamp", None)

        lifecycle_state = order_tracker.get_lifecycle_state(symbol, direction)
        if lifecycle_state in ["ENTRY_PENDING", "EXIT_PENDING"]:
            logger.log_debug(f"{symbol}-{direction} 🚫 Blocked by lifecycle state: {lifecycle_state}")
            continue

        try:
            result = execute_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                confidence=confidence,
                label=label,
                sentiment="N/A",
                source=source,
                timestamp=timestamp
            )

            if result:
                logger.log_info(f"✅ Trade executed for {symbol}-{direction} ({source}).")
                return  # Only one trade per cycle

            else:
                logger.log_warning(f"⚠️ Trade execution failed for {symbol}-{direction}.")

        except Exception as e:
            logger.log_critical(f"❌ Exception during trade execution: {e}")
            return
