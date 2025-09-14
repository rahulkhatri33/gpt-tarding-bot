import time
import math
import pandas as pd
from threading import Event
from datetime import timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
from core.logger import global_logger as logger
from core.config import CONFIG, get_usd_allocation
from core.position_manager import position_manager
from binance_utils import BinanceClient
from scalper.scalper_strategy import (
    calculate_ut_signals,
    _calculate_sl_tp,
    calculate_quantity,
    evaluate_scalper_entry,
)
from scalper.scalper_rolling_engine import scalper_rolling
from scalper.scalper_candle_listener import fetch_5m_data, convert_klines_to_dataframe
from utils.discord_logger import send_discord_log
import os

# Binance client - keep same env behavior you had
client = Client(
    api_key=os.getenv("BINANCE_API_KEY"),
    api_secret=os.getenv("BINANCE_API_SECRET"),
)

binance_utils = BinanceClient()
shutdown_flag = Event()


def run_scalper():
    """Main scalper loop."""
    base_pairs = CONFIG.get("base_pairs", [])
    scalper_settings = CONFIG.get("scalper_settings", {})
    min_candles = scalper_settings.get("min_candles", 300)
    timeframe = scalper_settings.get("timeframe", "5m")

    while not shutdown_flag.is_set():
        try:
            logger.log_info("[SCALPER] Starting new scalper cycle...")
            logger.log_debug(f"Full config: {CONFIG}")
            open_positions = position_manager.get_all_positions()
            logger.log_info(f"Open positions: {list(open_positions.keys())}")

            for symbol in base_pairs:
                try:
                    logger.log_debug(f"Processing symbol: {symbol}")
                    # Sync local <> exchange state for this symbol (safe sync implemented in position_manager)
                    position_manager.sync_with_binance(symbol=symbol)

                    balance = binance_utils.get_futures_balance()
                    logger.log_info(f"{symbol} 💰 Futures wallet balance: {balance} USDT")
                    current_price = binance_utils.get_price(symbol)
                    logger.log_info(f"{symbol} ✅ Current price: {current_price}")

                    # Let position manager check for partial TP hits
                    for direction in ["long", "short"]:
                        position_manager.check_partial_tp(symbol, direction, current_price)

                    logger.log_info(f"{symbol} 🧊 Fetching 5m candles...")
                    logger.log_debug(f"{symbol} Fetching {min_candles} klines for timeframe {timeframe}")
                    klines = fetch_5m_data(symbol, min_candles)
                    logger.log_debug(f"{symbol} Fetched {len(klines)} klines")
                    df = convert_klines_to_dataframe(klines)
                    if df.empty:
                        logger.log_warning(f"{symbol} 📉 Empty DataFrame, skipping...")
                        continue

                    logger.log_info(f"{symbol} ✅ 5m candles loaded: {len(df)}")
                    scalper_rolling.update_candles(symbol, df)

                    latest_candle_time = df['timestamp'].iloc[-1]
                    current_time = pd.Timestamp.now(tz=timezone.utc)
                    time_diff = (current_time - latest_candle_time).total_seconds()
                    logger.log_info(f"{symbol} ✅ Latest candle: {latest_candle_time} UTC, current: {current_time} UTC, diff: {time_diff}s")

                    side, sl_tp = evaluate_scalper_entry(
                        df,
                        scalper_settings | {"symbol": symbol, "max_concurrent_trades": CONFIG.get("max_concurrent_trades", {})},
                    )
                    if side is None or sl_tp is None:
                        if CONFIG.get("verbose_no_signal", False):
                            logger.log_info(f"{symbol} 📴 No trade signal.")
                        continue

                    qty = calculate_quantity(symbol, current_price, scalper_settings)
                    if qty == 0.0:
                        logger.log_error(f"{symbol} ❌ Skipping {side} trade: Invalid quantity")
                        continue

                    # Ensure sl/tp values are present and sane
                    if getattr(sl_tp, 'sl', 0.0) == 0.0 or getattr(sl_tp, 'tp', 0.0) == 0.0:
                        logger.log_error(f"{symbol} ❌ Skipping {side} trade: Invalid SL/TP")
                        continue

                    # SL/TP sanity check + RR
                    try:
                        risk = abs(current_price - sl_tp.sl)
                        reward = abs(sl_tp.tp - current_price)
                        rr = round(reward / risk, 4) if risk > 0 else None
                        logger.log_info(
                            f"{symbol} 🧮 SL/TP sanity | entry={current_price}, SL={sl_tp.sl}, TP={sl_tp.tp}, risk={risk:.8f}, reward={reward:.8f}, RR={rr}"
                        )
                        if risk <= 0 or reward <= 0:
                            logger.log_error(f"{symbol} ❌ Skipping {side} trade: Non-positive risk/reward.")
                            continue
                    except Exception as e:
                        logger.log_warning(f"{symbol} ⚠️ Failed RR precheck (non-blocking): {e}")

                    logger.log_info(
                        f"{symbol} 🚀 Executing {side} trade (filters+SL/TP validated): qty={qty}, price={current_price}, sl={sl_tp.sl}, tp={sl_tp.tp}, trailing_stop={sl_tp.trailing_stop}"
                    )
                    execute_trade(symbol, qty, side, current_price, sl_tp.sl, sl_tp.tp, sl_tp.trailing_stop)

                except Exception as e:
                    logger.log_error(f"{symbol} ❌ Scalper error: {str(e)}")
                    if CONFIG.get("alerts", {}).get("enabled", False):
                        try:
                            send_discord_log(f"{symbol} ❌ Scalper error: {str(e)[:200]}")
                        except Exception as discord_err:
                            logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
                    continue

            time.sleep(15)
        except Exception as e:
            logger.log_error(f"[SCALPER] Cycle error: {str(e)}")
            time.sleep(15)


def execute_trade(symbol: str, qty: float, side: str, price: float, sl: float, tp: float, trailing_stop: float):
    """Execute a trade on Binance Futures with safe preflight and robust entry-price persisting."""

    config = CONFIG
    scalper_settings = config.get("scalper_settings", {})
    symbol_precisions = scalper_settings.get("symbol_precisions", {}).get(symbol, {}) or {}
    price_precision = int(symbol_precisions.get("pricePrecision", 8))
    quantity_precision = int(symbol_precisions.get("quantityPrecision", 2))
    leverage = int(symbol_precisions.get("leverage", 20))

    # Normalize side/direction (accepts 'long'/'short' or 'LONG'/'SHORT')
    side_upper = side.upper() if isinstance(side, str) else str(side).upper()
    direction = "long" if side_upper == "LONG" else "short"

    # --- Minimal robust same-direction guard ---
    try:
        existing_pos = None
        if hasattr(position_manager, "get_position"):
            try:
                existing_pos = position_manager.get_position(symbol, direction)
            except TypeError:
                existing_pos = position_manager.get_position(symbol)
        elif hasattr(position_manager, "get_all_positions"):
            position_key = f"{symbol}_{direction}"
            existing_pos = position_manager.get_all_positions().get(position_key)
        else:
            existing_pos = None
    except Exception as _e:
        logger.log_warning(f"{symbol} ⚠️ Could not query existing position safely: {_e}")
        existing_pos = None

    if existing_pos:
        logger.log_info(f"{symbol} 📴 Skipping {side} trade: Position already exists")
        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} 📴 Skipped {side} trade: Position already exists")
            except Exception as discord_err:
                logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
        return
    # --- end minimal guard ---

    # --- Validate order parameters against Binance filters (preflight step 1) ---
    try:
        exchange_info = client.get_symbol_info(symbol)
        if not exchange_info:
            logger.log_error(f"{symbol} ❌ Failed to fetch exchange info")
            return

        lot_size_filter = next((f for f in exchange_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), {})
        min_qty = float(lot_size_filter.get("minQty", 0))
        max_qty = float(lot_size_filter.get("maxQty", float("inf")))
        step_size = float(lot_size_filter.get("stepSize", 0))

        notional_filter = next((f for f in exchange_info.get("filters", []) if f.get("filterType") == "MIN_NOTIONAL"), {})
        min_notional = float(notional_filter.get("minNotional", 0))

        # Trim qty to exchange step/precision safely
        if step_size > 0:
            try:
                qty = math.floor(qty / step_size) * step_size
            except Exception:
                qty = round(qty, quantity_precision)
        qty = round(qty, quantity_precision)

        if qty < min_qty or qty > max_qty:
            logger.log_error(f"{symbol} ❌ Quantity {qty} outside allowed range [{min_qty}, {max_qty}]")
            return

        notional = qty * price
        if notional < min_notional:
            logger.log_error(f"{symbol} ❌ Notional value {notional:.2f} below minimum {min_notional:.2f}")
            return

        logger.log_debug(f"{symbol} Order params preflight: qty={qty}, price={price:.{price_precision}f}, sl={sl:.{price_precision}f}, tp={tp:.{price_precision}f}, notional={notional:.2f}")
    except BinanceAPIException as e:
        logger.log_error(f"{symbol} ❌ Failed to validate order parameters (Binance error): {e}")
        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} ❌ Failed to validate order parameters: {str(e)[:200]}")
            except Exception as discord_err:
                logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
        return
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to validate order parameters: {str(e)}")
        return

    # --- Ensure we have correct quantity precision (preflight step 2) ---
    try:
        if "quantityPrecision" not in symbol_precisions:
            # Try to read from exchange-info top-level (some libs expose it)
            quantity_precision = int(exchange_info.get("quantityPrecision", quantity_precision))
            logger.log_info(f"{symbol} Using Binance quantityPrecision: {quantity_precision}")
        qty = round(qty, quantity_precision)
        if qty <= 0:
            logger.log_error(f"{symbol} ❌ Invalid quantity after trimming: {qty}")
            if config.get("alerts", {}).get("enabled", False):
                try:
                    send_discord_log(f"{symbol} ❌ Invalid quantity after trimming: {qty}")
                except Exception as discord_err:
                    logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
            return
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to fetch quantity precision: {str(e)}")
        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} ❌ Failed to fetch quantity precision: {str(e)[:200]}")
            except Exception as discord_err:
                logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
        return

    # --- Margin & balance precheck (preflight step 3) ---
    try:
        balance = binance_utils.get_futures_balance()
        notional_value = qty * price
        margin_required = notional_value / leverage
        maintenance_margin = notional_value * 0.01
        total_margin_needed = margin_required + maintenance_margin
        logger.log_info(
            f"{symbol} 🧮 Margin precheck | notional={notional_value:.4f}, lev={leverage}, req={margin_required:.4f}, maint~1%={maintenance_margin:.4f}, total_needed={total_margin_needed:.4f}, avail={balance:.4f}"
        )
        if total_margin_needed > balance:
            logger.log_error(f"{symbol} ❌ Insufficient margin: required={total_margin_needed:.2f} USDT, available={balance:.2f} USDT")
            if config.get("alerts", {}).get("enabled", False):
                try:
                    send_discord_log(f"{symbol} ❌ Insufficient margin: required={total_margin_needed:.2f} USDT, available={balance:.2f} USDT")
                except Exception as discord_err:
                    logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
            return
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Margin precheck failed: {e}")
        return

    # --- SAFE REVERSAL: close opposite only AFTER all preflight checks passed ---
    try:
        opposite_dir = "short" if direction == "long" else "long"
        opposite_key = f"{symbol}_{opposite_dir}"
        all_pos = position_manager.get_all_positions()
        if opposite_key in all_pos:
            prev = all_pos.get(opposite_key, {})
            prev_qty = float(prev.get("size", 0.0))
            if prev_qty > 0:
                close_side_for_opposite = "SELL" if opposite_dir == "long" else "BUY"
                position_side_opposite = "LONG" if opposite_dir == "long" else "SHORT"

                logger.log_info(f"{symbol} 🔁 SAFE REVERSAL: closing {opposite_dir.upper()} qty={prev_qty} before opening {side}")
                if not config.get("dry_run", False):
                    try:
                        pos_mode = binance_utils.client.futures_get_position_mode()
                        is_hedge = bool(pos_mode.get("dualSidePosition", False))
                        rounded_prev_qty = round(prev_qty, quantity_precision)

                        if is_hedge:
                            binance_utils.client.futures_create_order(
                                symbol=symbol,
                                side=close_side_for_opposite,
                                type="MARKET",
                                quantity=rounded_prev_qty,
                                positionSide=position_side_opposite,
                            )
                        else:
                            binance_utils.client.futures_create_order(
                                symbol=symbol,
                                side=close_side_for_opposite,
                                type="MARKET",
                                quantity=rounded_prev_qty,
                                reduceOnly=True,
                            )

                        logger.log_info(f"{symbol} ✅ Closed {opposite_dir.upper()} via market order (mode-aware).")
                    except BinanceAPIException as e:
                        logger.log_error(f"{symbol} ❌ Failed to close opposite position: {e}")
                        return
                    except Exception as e:
                        logger.log_error(f"{symbol} ❌ Unexpected error closing opposite position: {e}")
                        return

                position_manager.close_position(symbol, opposite_dir)

                if config.get("alerts", {}).get("enabled", False):
                    try:
                        send_discord_log(f"{symbol} 🔁 Reversal: closed {opposite_dir.upper()} (qty={prev_qty}) before opening {side}")
                    except Exception as discord_err:
                        logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Reversal handling error: {str(e)}")
        return

    # Round price/SL/TP for final logging + storage (order is market so rounding is for consistency)
    price = round(price, price_precision)
    sl = round(sl, price_precision)
    tp = round(tp, price_precision)

    if config.get("dry_run", False):
        logger.log_info(
            f"{symbol} 🧪 Dry run: {side} trade would be executed with qty={qty}, price={price}, sl={sl}, tp={tp} (no SL/TP orders placed)"
        )
        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} 🧪 Dry run: {side} trade would be executed with qty={qty}, price={price}, sl={sl}, tp={tp} (no SL/TP orders placed)")
            except Exception as discord_err:
                logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")
        return

    market_side = "SELL" if side_upper == "SHORT" else "BUY"
    position_side = "SHORT" if side_upper == "SHORT" else "LONG"

    try:
        # ensure leverage and mode
        position_mode = binance_utils.client.futures_get_position_mode()
        logger.log_debug(f"{symbol} Position mode: {'Hedge' if position_mode.get('dualSidePosition') else 'One-way'}")
        binance_utils.client.futures_change_leverage(symbol=symbol, leverage=leverage)

        order = binance_utils.client.futures_create_order(
            symbol=symbol,
            side=market_side,
            type="MARKET",
            quantity=qty,
            positionSide=position_side,
        )

        # --- Robust entry price discovery: ---
        entry_price = None
        entry_price_estimated = False
        try:
            if order is None:
                raise Exception("No order response from futures_create_order")
            # common fields used historically
            if order.get('avgPrice'):
                entry_price = float(order.get('avgPrice'))
            elif isinstance(order.get('fills'), list) and len(order.get('fills')) > 0 and order['fills'][0].get('price'):
                entry_price = float(order['fills'][0]['price'])
            elif order.get('price'):
                entry_price = float(order.get('price'))
        except Exception:
            entry_price = None

        # fallback: use fresh ticker if response lacked fill price or price is zero
        if not entry_price or entry_price <= 0:
            try:
                ticker_price = float(binance_utils.get_price(symbol))
                if ticker_price and ticker_price > 0:
                    entry_price = ticker_price
                    entry_price_estimated = True
                    logger.log_warning(f"{symbol} ⚠️ Using ticker as estimated entry price: {entry_price} (order response lacked fill price).")
                else:
                    entry_price = float(price) if price and price > 0 else 0.0
                    entry_price_estimated = entry_price > 0
                    if entry_price_estimated:
                        logger.log_warning(f"{symbol} ⚠️ Using provided strategy price as estimated entry price: {entry_price}")
            except Exception as e:
                logger.log_error(f"{symbol} ❌ Failed to determine entry price from order or ticker: {e}")
                entry_price = float(price) if price and price > 0 else 0.0
                entry_price_estimated = entry_price > 0

        # Safety: do not persist a position with entry_price == 0.0
        if not entry_price or entry_price <= 0:
            logger.log_error(f"{symbol} ❌ Entry price resolved to 0 — aborting position persist to avoid bad partial TP math.")
            return

        logger.log_info(
            f"{symbol} 🚀 Executed {side} trade: qty={qty}, entry_price={entry_price}, sl={sl}, tp={tp}, orderId={order.get('orderId')} (no SL/TP orders placed)"
        )

        # Persist a position record so exits can operate reliably
        try:
            pos_payload = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": float(entry_price),
                "size": float(qty),
                "stop_loss": float(sl),
                "take_profit": float(tp),
                "peak_price": float(entry_price),
                "source": "5M_SCALPER",
                "entry_time": pd.Timestamp.now(tz=timezone.utc).isoformat(),
            }
            if entry_price_estimated:
                pos_payload["entry_price_estimated"] = True

            # add or update depending on position_manager API
            try:
                position_manager.add_position(symbol, direction, pos_payload)
            except Exception:
                position_manager.update_position(symbol, direction, pos_payload)
        except Exception as e:
            logger.log_warning(f"{symbol} ⚠️ Failed to persist position after entry: {e}")

        # Partial TP configuration and logging: only when entry_price > 0
        ptp_cfg = config.get("scalper_settings", {}).get("partial_tp", {})
        if ptp_cfg.get("enabled", False):
            rr_first = ptp_cfg.get("first_rr", 1.0)
            first_size_pct = ptp_cfg.get("first_size_pct", 0.5)
            try:
                # Use direction-correct formula
                if direction == "long":
                    first_tp = entry_price + (entry_price - sl) * rr_first
                else:
                    first_tp = entry_price - (sl - entry_price) * rr_first

                # sanity checks: positive and not equal to SL
                if first_tp <= 0 or abs(first_tp - sl) < 1e-9:
                    logger.log_warning(f"{symbol} ⚠️ Computed TP1 invalid (<=0 or identical to SL). Skipping partial TP.")
                else:
                    position_manager.update_position(
                        symbol,
                        direction,
                        {
                            "partial_tp_price": round(first_tp, price_precision),
                            "partial_tp_size": round(qty * first_size_pct, quantity_precision),
                            "trail_remaining": ptp_cfg.get("trail_remaining", True),
                        },
                    )
                    logger.log_info(f"{symbol} 📊 Partial TP set: {round(first_tp, price_precision)} size={round(qty * first_size_pct, quantity_precision)}")
                    if config.get('alerts', {}).get('enabled', False):
                        try:
                            send_discord_log(f"{symbol} 📊 Partial TP set: {first_size_pct*100:.0f}% at {first_tp}, SL->BE, trail rest")
                        except Exception as discord_err:
                            logger.log_error(f"{symbol} ❌ Failed to send Partial TP alert: {discord_err}")
            except Exception as e:
                logger.log_warning(f"{symbol} ⚠️ Failed to compute partial TP: {e}")
        else:
            logger.log_debug(f"{symbol} Partial TP disabled in config.")

        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} 🚀 {side} trade executed: qty={qty}, price={entry_price}, sl={sl}, tp={tp}, orderId={order.get('orderId')} (no SL/TP orders)")
            except Exception as discord_err:
                logger.log_error(f"{symbol} ❌ Failed to send Discord alert: {str(discord_err)}")

    except BinanceAPIException as e:
        logger.log_error(f"{symbol} ❌ Trade execution error: {str(e)}")
        if 'precision' in str(e).lower() or 'quantity' in str(e).lower():
            logger.log_error(f"{symbol} ❌ Looks like a precision/stepSize error — verify symbol_precisions in config.json and stepSize from exchange info.")
        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} ❌ Trade execution error: {str(e)[:200]}")
            except Exception:
                pass
        return
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Unexpected trade execution error: {e}")
        if config.get("alerts", {}).get("enabled", False):
            try:
                send_discord_log(f"{symbol} ❌ Unexpected trade execution error: {str(e)[:200]}")
            except Exception:
                pass
        return
