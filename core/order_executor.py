# core/order_executor.py
import os
from core.logger import global_logger as logger
from binance_utils import BinanceUtils
from core.config import CONFIG
from core.position_manager import position_manager

binance_utils = BinanceUtils(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

def execute_order(signal):
    """
    Execute a trade order with optional partial TP support.
    Signal dict must include:
      symbol, side, entry, sl, tp
    """
    try:
        symbol = signal['symbol']
        side = signal['side'].upper()
        entry = float(signal['entry'])
        sl = float(signal.get('sl', 0.0))
        tp = float(signal.get('tp', 0.0))

        # --- Position sizing ---
        qty = binance_utils.calculate_quantity(
            symbol,
            CONFIG['usd_allocation_scalper'].get(symbol, 50),
            entry
        )

        if qty <= 0:
            logger.log_error(f"{symbol} ❌ Invalid qty={qty}. Skipping {side}.")
            return

        # --- Partial TP config ---
        ptp_cfg = CONFIG.get("scalper_settings", {}).get("partial_tp", {})
        partial_enabled = ptp_cfg.get("enabled", False)
        rr_first = float(ptp_cfg.get("first_rr", 1.0))
        first_size_pct = float(ptp_cfg.get("first_size_pct", 0.5))

        # --- Calculate partial TP if enabled ---
        partial_tp_price = None
        if partial_enabled and sl > 0.0:
            risk = abs(entry - sl)
            if risk > 0:
                if side == "LONG":
                    partial_tp_price = entry + risk * rr_first
                else:  # SHORT
                    partial_tp_price = entry - risk * rr_first

        # --- Log execution plan ---
        if partial_enabled and partial_tp_price:
            logger.info(
                f"{symbol} 🚀 Executing {side} with Partial TP: "
                f"{first_size_pct*100:.0f}% at {partial_tp_price}, "
                f"SL={sl}, Final TP={tp}, Trail rest"
            )
        else:
            logger.info(
                f"{symbol} 🚀 Executing {side} full position: qty={qty}, entry={entry}, SL={sl}, TP={tp}"
            )

        # --- Place order (entry only for now) ---
        if binance_utils.validate_order(symbol, entry, qty):
            logger.info(f"📡 MARKET {side} order placed for {symbol}: Qty={qty}")

            # Save into position manager (with partial TP metadata if enabled)
            position_data = {
                "symbol": symbol,
                "direction": side.lower(),
                "entry_price": entry,
                "size": qty,
                "stop_loss": sl,
                "take_profit": tp,
                "partial_tp_price": partial_tp_price if partial_enabled else None,
                "partial_tp_size": qty * first_size_pct if partial_enabled else None,
                "trail_remaining": partial_enabled,
                "confidence": 1.0,
                "label": "scalper",
                "source": "order_executor",
            }
            position_manager.add_position(symbol, side.lower(), position_data)

        else:
            logger.log_error(f"{symbol} ❌ Validation failed for {side} order.")
    except Exception as e:
        logger.log_error(f"❌ Order execution failed for {signal.get('symbol','?')}: {str(e)[:200]}")
