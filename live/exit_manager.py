# exit_manager.py (patched minimal)
"""
Exit manager: SL / TP / TP1 partial / trailing logic.

Defensive features included:
 - Avoid immediate local deletion when Binance shows no position; mark binance_missing_since and wait a grace period.
 - If API credentials missing or dry-run mode enabled, do not call private endpoints and act conservatively.
 - Defensive trailing stop calculation: coerce numeric types and skip if invalid.
 - When TP1 remainder trims to zero, query Binance for remaining position and handle gracefully.
 - Add debug stack trace when close_position is invoked to quickly identify caller.
 - Ensure lifecycle & notifier have fallbacks.
"""
import os
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

from binance.client import Client
from binance.exceptions import BinanceAPIException

from core.logger import global_logger as logger
from core.position_manager import position_manager
from core import order_tracker
from core.symbol_precision import get_trimmed_quantity
from core.config import is_dry_run_enabled
from engine.sl_tp_engine import calculate_scalper_trailing_stop, calculate_trailing_stop_ml
from utils.price_fetcher import get_latest_price
from utils.notifier import notifier
from core.analytics.trade_recorder import append_lifecycle, snapshot_equity
from scalper.sl_tracker import record_scalper_sl_hit
from dotenv import load_dotenv

# fallback discord logger (used when notifier fails)
from utils.discord_logger import send_discord_log

load_dotenv()

# ENV credentials detection
ENV_API_KEY = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY_LOCAL")
ENV_API_SECRET = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET_LOCAL")
HAS_API_CREDENTIALS = bool(ENV_API_KEY and ENV_API_SECRET)

client = None
if HAS_API_CREDENTIALS:
    try:
        client = Client(api_key=ENV_API_KEY, api_secret=ENV_API_SECRET)
    except Exception as e:
        logger.log_warning(f"Failed to init Binance client: {e}")
        client = None
        HAS_API_CREDENTIALS = False
else:
    logger.log_info("No Binance API credentials found — exit_manager running in safe/dry-run mode (no private API calls).")

BINANCE_MISSING_GRACE_SECONDS = int(os.getenv("BINANCE_MISSING_GRACE_SECONDS", "30"))


# ---------- helpers ----------
def _to_float_safe(v):
    """Coerce numeric-ish values to float or return None."""
    try:
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        if hasattr(v, "item"):
            try:
                return float(v.item())
            except Exception:
                pass
        return float(v)
    except Exception:
        return None


def _pos_is_sane(pos: Dict[str, Any]) -> bool:
    """
    Local sanity check to avoid evaluating positions with invalid numeric fields.
    Returns True when entry, sl, tp and size are numeric and ordered properly.
    Uses configured min_sl_distance_pct so very small rounding differences don't reject positions.
    """
    try:
        if not isinstance(pos, dict):
            return False
        direction = pos.get("direction")
        entry = _to_float_safe(pos.get("entry_price"))
        sl = _to_float_safe(pos.get("stop_loss"))
        tp = _to_float_safe(pos.get("take_profit"))
        size = _to_float_safe(pos.get("size"))
        if entry is None or sl is None or tp is None or size is None:
            return False
        if entry <= 0 or size <= 0:
            return False

        # use config min distance
        try:
            from core.config import get_config

            cfg = get_config()
            scalper_settings = cfg.get("scalper_settings", {}) if isinstance(cfg, dict) else {}
            min_sl_pct = float(scalper_settings.get("min_sl_distance_pct", 0.0005))
        except Exception:
            min_sl_pct = 0.0005

        min_sl_abs = abs(entry) * min_sl_pct

        if direction == "long":
            if not (sl < entry):
                return False
            if not (entry < tp):
                return False
            if (entry - sl) < min_sl_abs:
                return False
            return True
        elif direction == "short":
            if not (sl > entry):
                return False
            if not (entry > tp):
                return False
            if (sl - entry) < min_sl_abs:
                return False
            return True
        else:
            return False
    except Exception:
        logger.log_debug(f"_pos_is_sane exception for pos: {pos}")
        return False


# ---------- Binance helpers ----------
def _is_position_live_on_binance(symbol: str, direction: str) -> bool:
    """Return True if Binance reports a live position for symbol/direction.
    Conservative on API failure or dry-run.
    """
    if is_dry_run_enabled() or not HAS_API_CREDENTIALS or client is None:
        logger.log_debug(f"{symbol}-{direction} _is_position_live_on_binance: dry/safe mode -> assume live")
        return True
    try:
        positions = client.futures_position_information(symbol=symbol)
        for p in positions:
            try:
                amt = float(p.get("positionAmt", 0.0))
            except Exception:
                amt = 0.0
            if abs(amt) <= 0.0000001:
                continue
            pos_side = "long" if amt > 0 else "short"
            if pos_side == direction:
                return True
        return False
    except Exception as e:
        logger.log_error(f"{symbol}-{direction} ⚠️ Failed to fetch positions from Binance: {e}")
        logger.log_debug(traceback.format_exc())
        # Be conservative and assume live to avoid accidental removal
        return True


def _is_account_hedge_mode() -> bool:
    if not HAS_API_CREDENTIALS or client is None:
        logger.log_debug("Hedge mode unknown: no API credentials -> assume one-way")
        return False
    try:
        resp = client.futures_get_position_mode()
        return bool(resp.get("dualSidePosition", False))
    except Exception as e:
        logger.log_warning(f"Could not determine futures position mode (assuming one-way): {e}")
        logger.log_debug(traceback.format_exc())
        return False


def _send_market_exit(symbol: str, direction: str, qty: float) -> Optional[Dict[str, Any]]:
    """Place market exit order. If dry-run or no credentials, simulate response (executedQty)."""
    if qty <= 0:
        logger.log_error(f"{symbol} ❌ Invalid exit qty: {qty}")
        return None

    # If no creds or dry-run: do not call API
    if is_dry_run_enabled() or not HAS_API_CREDENTIALS or client is None:
        logger.log_info(f"{symbol} 🧪 DRY RUN / no API creds: would exit {direction} qty={qty}")
        return {"status": "dry_run", "executedQty": float(qty)}

    close_side = Client.SIDE_SELL if direction == "long" else Client.SIDE_BUY
    trimmed_qty = get_trimmed_quantity(symbol, float(qty))
    if trimmed_qty <= 0:
        logger.log_error(f"{symbol} ❌ Trimmed exit qty is zero after precision: {trimmed_qty}")
        return None

    try:
        is_hedge = _is_account_hedge_mode()
        payload = {"symbol": symbol, "side": close_side, "type": Client.ORDER_TYPE_MARKET, "quantity": trimmed_qty}
        if is_hedge:
            payload["positionSide"] = "LONG" if direction == "long" else "SHORT"
        else:
            payload["reduceOnly"] = True

        logger.log_info(f"{symbol} 🔁 Sending market exit payload: {payload}")
        resp = client.futures_create_order(**payload)
        logger.log_info(f"{symbol} ✅ Exit order response: {resp}")
        return resp
    except AssertionError as e:
        logger.log_critical(f"{symbol} ❌ Assertion while exiting (probably missing creds): {e}")
        logger.log_debug(traceback.format_exc())
        return None
    except BinanceAPIException as e:
        # Catch Binance API errors and return None so caller can handle
        logger.log_critical(f"{symbol} ❌ BinanceAPIException while exiting: {getattr(e, 'message', str(e))}")
        logger.log_debug(traceback.format_exc())
        return None
    except Exception as e:
        logger.log_critical(f"{symbol} ❌ Unexpected error while exiting: {e}")
        logger.log_debug(traceback.format_exc())
        return None


# ---------- core exit flows ----------
def full_exit(symbol: str, direction: str, price: float, reason: str) -> None:
    """Fully exit the given position and record lifecycle."""
    # snapshot for debugging
    binance_snapshot = None
    if HAS_API_CREDENTIALS and client is not None:
        try:
            binance_snapshot = client.futures_position_information(symbol=symbol)
        except Exception as e:
            binance_snapshot = f"snapshot_fetch_failed: {e}"

    if not _is_position_live_on_binance(symbol, direction):
        logger.log_warning(
            f"{symbol}-{direction} ⛔ Binance reports no live position. Preserving local state for reconciliation. Snapshot: {binance_snapshot}"
        )
        try:
            pos = position_manager.get_position(symbol, direction)
            if isinstance(pos, dict) and not pos.get("binance_missing_since"):
                pos["binance_missing_since"] = int(time.time())
                position_manager.update_position(symbol, direction, {"binance_missing_since": pos["binance_missing_since"]})
        except Exception as e:
            logger.log_warning(f"{symbol}-{direction} ⚠️ Could not mark binance_missing_since: {e}")
        try:
            notifier.send_info(f"{symbol}-{direction} ⛔ Binance shows no position; local state preserved for reconciliation.")
        except Exception:
            try:
                send_discord_log(f"{symbol}-{direction} ⛔ Binance shows no position; local state preserved for reconciliation.")
            except Exception:
                logger.log_debug("Notifier & discord both failed while reporting Binance-missing.")
        return

    if not order_tracker.mark_exit_pending(symbol, direction):
        logger.log_debug(f"{symbol}-{direction} 🚫 Exit already pending. Skipping duplicate.")
        return

    pos = position_manager.get_position(symbol, direction)
    if not pos:
        logger.log_warning(f"{symbol}-{direction} ❗ Local position record missing at exit time.")
        return

    qty = _to_float_safe(pos.get("size") or 0.0) or 0.0
    entry_price = _to_float_safe(pos.get("entry_price")) or 0.0
    sl = _to_float_safe(pos.get("stop_loss")) or 0.0
    tp = _to_float_safe(pos.get("take_profit")) or 0.0

    if qty <= 0:
        logger.log_warning(f"{symbol}-{direction} ⚠️ Attempt to close empty or zero-size position.")
        position_manager.close_position(symbol, direction)
        return

    response = _send_market_exit(symbol, direction, qty)
    if response is None:
        logger.log_critical(f"{symbol} ❌ EXIT FAILED — Market exit not placed.")
        try:
            notifier.send_critical(f"{symbol} ❌ EXIT FAILED — Market exit not placed. Will retry.")
        except Exception:
            try:
                send_discord_log(f"{symbol} ❌ EXIT FAILED — Market exit not placed. Will retry.")
            except Exception:
                logger.log_debug("Notifier & discord failed sending critical exit failure.")
        return

    # compute approximate pnl if entry_price valid
    pnl = None
    try:
        if entry_price > 0:
            pnl = (float(price) - entry_price) * qty if pos.get("direction") == "long" else (entry_price - float(price)) * qty
    except Exception:
        pnl = None

    # lifecycle append with fallback to file
    try:
        append_lifecycle(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "symbol": symbol,
                "direction": direction,
                "event_type": f"{reason}_EXIT",
                "price": float(price),
                "qty": qty,
                "entry_price": entry_price,
                "pnl": pnl,
                "stop_loss": sl,
                "take_profit": tp,
                "reason": reason,
                "source": pos.get("source", "unknown"),
            }
        )
    except Exception:
        try:
            # fallback CSV write
            with open("trade_exit_fallback.csv", "a") as f:
                f.write(f"{datetime.utcnow().isoformat()},{symbol},{direction},{reason}_EXIT,{price},{qty},{entry_price},{pnl}\n")
            logger.log_info(f"{symbol} fallback exit row written.")
        except Exception:
            logger.log_debug("Failed to write fallback exit CSV.")

    snapshot_equity(tag=f"{reason}_EXIT")

    logger.log_info(f"{symbol}-{direction} CLOSED — Reason: {reason} | Exit @ {price:.6f} | PnL: {pnl}")
    try:
        notifier.send_exit_alert(symbol, reason, float(price), qty, direction, pnl)
    except Exception:
        try:
            send_discord_log(f"{symbol} CLOSED — {reason} | {price} qty={qty} pnl={pnl}")
        except Exception:
            logger.log_debug("Notifier & discord both failed to send exit alert.")

    position_manager.close_position(symbol, direction)
    order_tracker.clear(symbol, direction)

    if reason.upper() == "SL":
        try:
            if pos.get("source") == "5M_SCALPER" and not pos.get("tp1_triggered", False):
                record_scalper_sl_hit()
        except Exception:
            logger.log_debug("record_scalper_sl_hit failed.")


def handle_tp1(symbol: str, direction: str, price: float) -> None:
    """Handle TP1: close partial size, move SL to entry and schedule trailing."""
    pos = position_manager.get_position(symbol, direction)
    if not pos:
        logger.log_debug(f"{symbol}-{direction} TP1 handler: local position not found.")
        return

    # Sanity gate — if pos not sane, skip TP1 and preserve local state
    if not _pos_is_sane(pos):
        logger.log_warning(f"{symbol}-{direction} TP1 skipped: position not sane (will not attempt partial close).")
        # mark missing for manual reconciliation if entry_price invalid
        try:
            if not _to_float_safe(pos.get("entry_price")):
                pos["binance_missing_since"] = int(time.time())
                position_manager.update_position(symbol, direction, {"binance_missing_since": pos["binance_missing_since"]})
        except Exception:
            logger.log_debug("Failed to mark binance_missing_since in TP1 sanity fallback.")
        return

    total_size = _to_float_safe(pos.get("size") or 0.0)
    if total_size <= 0:
        logger.log_warning(f"{symbol}-{direction} TP1: size <= 0; skipping.")
        return

    remainder_size = total_size / 2.0
    trimmed_remainder = get_trimmed_quantity(symbol, remainder_size)

    if trimmed_remainder <= 0:
        logger.log_warning(f"{symbol}-{direction} ⚠️ TP1 remainder trimmed to <=0 ({trimmed_remainder}).")
        if HAS_API_CREDENTIALS and client is not None:
            try:
                positions = client.futures_position_information(symbol=symbol)
                remaining_amt = 0.0
                for p in positions:
                    try:
                        amt = float(p.get("positionAmt", 0.0))
                    except Exception:
                        amt = 0.0
                    if abs(amt) > 0:
                        remaining_amt = abs(amt)
                        break

                if remaining_amt > 0:
                    logger.log_info(f"{symbol} Closing remaining amount from Binance: {remaining_amt}")
                    resp = _send_market_exit(symbol, direction, remaining_amt)
                    if resp is None:
                        logger.log_critical(f"{symbol} ❌ Failed to close remaining after TP1.")
                        try:
                            notifier.send_critical(f"{symbol} ❌ Failed to close remaining after TP1: {remaining_amt}")
                        except Exception:
                            try:
                                send_discord_log(f"{symbol} ❌ Failed to close remaining after TP1: {remaining_amt}")
                            except Exception:
                                logger.log_debug("Notifier & discord failed for TP1 remaining-close.")
                        return

                    executed_qty = 0.0
                    try:
                        executed_qty = float(resp.get("executedQty") or resp.get("filledQty") or resp.get("origQty") or remaining_amt)
                    except Exception:
                        executed_qty = remaining_amt

                    try:
                        current_size = _to_float_safe(pos.get("size") or 0.0)
                        new_size = max(0.0, current_size - executed_qty)
                        position_manager.update_position(symbol, direction, {"size": new_size, "tp1_triggered": True})
                    except Exception:
                        logger.log_debug("Failed to persist updated size after remaining-close.")

                    try:
                        append_lifecycle(
                            {
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "symbol": symbol,
                                "direction": direction,
                                "event_type": "TP1_EXIT",
                                "price": float(price),
                                "qty": executed_qty,
                                "entry_price": float(pos.get("entry_price", 0.0)),
                                "pnl": (float(price) - float(pos.get("entry_price", 0.0))) * (executed_qty if direction == "long" else -executed_qty),
                                "stop_loss": float(pos.get("entry_price", 0.0)),
                                "take_profit": float(pos.get("take_profit", 0.0)),
                                "reason": "TP1 partial hit (remaining-close)",
                                "source": pos.get("source", "N/A"),
                            }
                        )
                    except Exception:
                        try:
                            with open("trade_exit_fallback.csv", "a") as f:
                                f.write(f"{datetime.utcnow().isoformat()},{symbol},TP1_EXIT,{price},{executed_qty},{pos.get('entry_price')}\n")
                        except Exception:
                            logger.log_debug("Fallback write failed for TP1 remaining-close.")

                    snapshot_equity(tag="TP1_EXIT")
                    try:
                        notifier.send_info(f"{symbol} 🎯 Partial TP filled (remaining-close): closed {executed_qty} @ {price:.6f}")
                    except Exception:
                        try:
                            send_discord_log(f"{symbol} 🎯 Partial TP filled (remaining-close): closed {executed_qty} @ {price:.6f}")
                        except Exception:
                            logger.log_debug("Notifier & discord failed for TP1 remaining-close.")

                    if new_size <= 0:
                        position_manager.close_position(symbol, direction)
                        order_tracker.clear(symbol, direction)
                    return
                else:
                    logger.log_info(f"{symbol} No remaining position on Binance after TP1 rounding. Marking missing.")
                    try:
                        pos_local = position_manager.get_position(symbol, direction)
                        if isinstance(pos_local, dict):
                            pos_local["binance_missing_since"] = int(time.time())
                            position_manager.update_position(symbol, direction, {"binance_missing_since": pos_local["binance_missing_since"]})
                    except Exception:
                        logger.log_debug("Failed to persist binance_missing_since after TP1 rounding.")
                    return
            except Exception as e:
                logger.log_error(f"{symbol} ❌ Error checking remaining size after TP1: {e}")
                logger.log_debug(traceback.format_exc())
                return
        else:
            try:
                pos_local = position_manager.get_position(symbol, direction)
                if isinstance(pos_local, dict):
                    pos_local["binance_missing_since"] = int(time.time())
                    position_manager.update_position(symbol, direction, {"binance_missing_since": pos_local["binance_missing_since"]})
            except Exception:
                logger.log_debug("Failed to persist binance_missing_since (no API).")
            try:
                notifier.send_info(f"{symbol}-{direction} ⛔ TP1: no API creds to verify remaining; preserving local state.")
            except Exception:
                try:
                    send_discord_log(f"{symbol}-{direction} ⛔ TP1: no API creds to verify remaining; preserving local state.")
                except Exception:
                    logger.log_debug("Notifier & discord failed for TP1 missing-state.")
            return

    # Normal partial close flow when trimmed_remainder > 0
    if not _is_position_live_on_binance(symbol, direction):
        logger.log_warning(f"{symbol}-{direction} ⛔ No live position found on Binance during TP1. Marking missing and preserving local state.")
        try:
            pos["binance_missing_since"] = int(time.time())
            position_manager.update_position(symbol, direction, {"binance_missing_since": pos["binance_missing_since"]})
        except Exception:
            logger.log_debug("Failed to persist binance_missing_since for TP1.")
        try:
            notifier.send_info(f"{symbol}-{direction} ⛔ TP1: Binance shows no live position; local state preserved.")
        except Exception:
            try:
                send_discord_log(f"{symbol}-{direction} ⛔ TP1: Binance shows no live position; local state preserved.")
            except Exception:
                logger.log_debug("Notifier & discord failed for TP1 missing-state.")
        return

    resp = _send_market_exit(symbol, direction, trimmed_remainder)
    if resp is None:
        logger.log_critical(f"{symbol} ❌ TP1 partial exit failed to place.")
        try:
            notifier.send_critical(f"{symbol} ❌ TP1 partial exit failed.")
        except Exception:
            try:
                send_discord_log(f"{symbol} ❌ TP1 partial exit failed.")
            except Exception:
                logger.log_debug("Notifier & discord failed for TP1 critical.")
        return

    executed_qty = 0.0
    try:
        executed_qty = float(resp.get("executedQty") or resp.get("filledQty") or resp.get("origQty") or trimmed_remainder)
    except Exception:
        executed_qty = trimmed_remainder

    try:
        current_size = _to_float_safe(pos.get("size") or 0.0)
        new_size = max(0.0, current_size - executed_qty)
        # Only set stop_loss to entry_price if entry_price is numeric and > 0
        entry_val = _to_float_safe(pos.get("entry_price")) or 0.0
        sl_to_set = entry_val if entry_val > 0 else pos.get("stop_loss", pos.get("entry_price", price))
        position_manager.update_position(symbol, direction, {"size": new_size, "tp1_triggered": True, "awaiting_trail_activation": True, "stop_loss": sl_to_set})
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to update local position after TP1: {e}")
        logger.log_debug(traceback.format_exc())

    try:
        append_lifecycle(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "symbol": symbol,
                "direction": direction,
                "event_type": "TP1_EXIT",
                "price": float(price),
                "qty": executed_qty,
                "entry_price": float(pos.get("entry_price", 0.0)),
                "pnl": (float(price) - float(pos.get("entry_price", 0.0))) * (executed_qty if direction == "long" else -executed_qty),
                "stop_loss": float(pos.get("entry_price", 0.0)) if _to_float_safe(pos.get("entry_price")) else pos.get("stop_loss"),
                "take_profit": float(pos.get("take_profit", 0.0)),
                "reason": "TP1 partial hit",
                "source": pos.get("source", "N/A"),
            }
        )
    except Exception:
        try:
            with open("trade_exit_fallback.csv", "a") as f:
                f.write(f"{datetime.utcnow().isoformat()},{symbol},TP1_EXIT,{price},{executed_qty},{pos.get('entry_price')}\n")
        except Exception:
            logger.log_debug("Fallback write failed for TP1_exit.")

    snapshot_equity(tag="TP1_EXIT")
    logger.log_info(f"{symbol}-{direction} 🏁 TP1 Partial Exit: closed {executed_qty} @ {price:.6f}, SL moved to Entry.")
    try:
        notifier.send_info(f"{symbol} 🎯 Partial TP triggered: closed {executed_qty} @ {price:.6f}, SL -> BE")
    except Exception:
        try:
            send_discord_log(f"{symbol} 🎯 Partial TP triggered: closed {executed_qty} @ {price:.6f}, SL -> BE")
        except Exception:
            logger.log_warning(f"{symbol} ⚠️ Notifier & discord failed to send TP1 alert.")


def price_poll_exit_loop() -> None:
    """
    Poll live prices and evaluate exits:
      - Stop Loss
      - TP1 (compare with stored partial_tp_price if present)
      - Trailing stop activation & trailing exit
    """
    poll_interval = 0.5
    while True:
        try:
            try:
                position_manager.sync_with_binance()
            except Exception as e:
                logger.log_debug(f"Position manager sync failed (non-blocking): {e}")

            positions_snapshot = list(position_manager.get_all_positions().items())

            for key, pos in positions_snapshot:
                try:
                    symbol = pos.get("symbol")
                    direction = pos.get("direction")
                    # Basic present checks
                    if not symbol or not direction:
                        continue

                    # Skip exit evaluation for unsane positions (prevents spurious immediate exits)
                    if not _pos_is_sane(pos):
                        logger.log_debug(f"{key} skipped: position not sane for exit checks (entry/sl/tp/size invalid).")
                        continue

                    sl = _to_float_safe(pos.get("stop_loss", 0.0)) or 0.0
                    tp = _to_float_safe(pos.get("take_profit", 0.0)) or 0.0
                    peak_price = pos.get("peak_price", pos.get("entry_price", 0.0))
                    # coerce entry_price if needed
                    entry_price = _to_float_safe(pos.get("entry_price")) or 0.0
                    exit_pending = pos.get("exit_pending", False)

                    if exit_pending:
                        continue

                    price = get_latest_price(symbol)
                    if price is None:
                        continue
                    price = float(price)

                    # STOP LOSS
                    if not pos.get("trail_active", False) and (
                        (direction == "long" and price <= sl)
                        or (direction == "short" and price >= sl)
                    ):
                        logger.log_info(f"{symbol} {direction} SL condition met: price={price} sl={sl}")
                        full_exit(symbol, direction, price, reason="SL")
                        continue

                    # TP1 (use stored partial_tp_price if available; fallback to final tp)
                    partial_tp = pos.get("partial_tp_price", None)
                    if partial_tp is None:
                        partial_tp = tp
                    try:
                        partial_tp = float(partial_tp)
                    except Exception:
                        partial_tp = tp

                    if not pos.get("tp1_triggered", False) and (
                        (direction == "long" and price >= partial_tp)
                        or (direction == "short" and price <= partial_tp)
                    ):
                        logger.log_info(f"{symbol} 🎯 TP1 candidate triggered in price poll: price={price} partial_tp={partial_tp}")
                        handle_tp1(symbol, direction, price)
                        continue

                    # Trailing activation check
                    if pos.get("awaiting_trail_activation", False):
                        buffer_triggered = False
                        if direction == "long" and price >= partial_tp * 1.002:
                            buffer_triggered = True
                        elif direction == "short" and price <= partial_tp * 0.998:
                            buffer_triggered = True

                        if buffer_triggered:
                            pos["awaiting_trail_activation"] = False
                            pos["trail_active"] = True
                            pos["stop_loss"] = partial_tp
                            try:
                                position_manager.update_position(symbol, direction, {"awaiting_trail_activation": False, "trail_active": True, "stop_loss": pos["stop_loss"]})
                            except Exception:
                                logger.log_debug("Failed to persist trail activation changes.")
                            try:
                                notifier.send_info(f"{symbol}-{direction} 🚀 Trailing activated; SL set to TP1 ({partial_tp:.6f})")
                            except Exception:
                                try:
                                    send_discord_log(f"{symbol}-{direction} 🚀 Trailing activated; SL set to TP1 ({partial_tp:.6f})")
                                except Exception:
                                    logger.log_debug("Notifier & discord failed for trail activation.")

                    # Trailing stop active logic (defensive)
                    if pos.get("trail_active", False):
                        # coerce peak_price safely
                        def _to_float_safe_local(x):
                            try:
                                if isinstance(x, (list, tuple)) and len(x) > 0:
                                    x = x[0]
                                if hasattr(x, "item"):
                                    return float(x.item())
                                return float(x)
                            except Exception:
                                return None

                        peak_price_val = _to_float_safe_local(peak_price)
                        if peak_price_val is None:
                            logger.log_error(f"{symbol}_{direction} ✖ invalid peak_price for trailing computation: {peak_price!r}. Skipping trailing calculation this cycle.")
                            continue

                        # update peak if price extends
                        if direction == "long" and price > peak_price_val:
                            position_manager.set_peak_price(symbol, direction, price)
                            peak_price_val = price
                        elif direction == "short" and price < peak_price_val:
                            position_manager.set_peak_price(symbol, direction, price)
                            peak_price_val = price

                        # compute trailing SL defensively using the engine function which supports symbol-based lookup too
                        try:
                            trailing_sl = calculate_scalper_trailing_stop(symbol, peak_price_val, direction)
                        except Exception as e:
                            # This guards against unexpected type errors coming from bad data types
                            logger.log_error(f"❌ Trailing calculation error for {symbol}_{direction}: {e}; skipping trailing this cycle.")
                            logger.log_debug(traceback.format_exc())
                            continue

                        # ensure trailing_sl is numeric
                        trailing_sl_val = _to_float_safe(trailing_sl)
                        if trailing_sl_val is None:
                            logger.log_debug(f"{symbol}_{direction} trailing_sl could not be computed (None or invalid). Skipping trailing exit check.")
                            continue

                        pos["trailing_sl"] = float(trailing_sl_val)
                        try:
                            position_manager.update_position(symbol, direction, {"trailing_sl": pos["trailing_sl"]})
                        except Exception:
                            logger.log_debug("Failed to persist trailing SL.")

                        if (direction == "long" and price <= trailing_sl_val) or (direction == "short" and price >= trailing_sl_val):
                            logger.log_info(f"{symbol} {direction} trailing SL hit: price={price} trailing_sl={trailing_sl_val}")
                            full_exit(symbol, direction, price, reason="TRAILING")
                            continue

                except Exception as e:
                    logger.log_critical(f"❌ Price polling exit error for {key}: {e}")
                    logger.log_debug(traceback.format_exc())

            time.sleep(poll_interval)
        except Exception as e:
            logger.log_critical(f"❌ price_poll_exit_loop top-level error: {e}")
            logger.log_debug(traceback.format_exc())
            time.sleep(1.0)
