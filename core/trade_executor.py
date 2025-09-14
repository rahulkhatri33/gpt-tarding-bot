from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from core.logger import global_logger as logger
from core.position_manager import position_manager
from core.config import get_config
from utils.discord_logger import send_discord_log  # ✅ Added for SL/TP alerts
from utils.exchange import client, get_qty_step_size, round_to_step
import math

def execute_trade(symbol: str, direction: str, price: float, qty: float, dry_run: bool, sl: float, tp: float) -> dict:
    logger.log_info(f"{symbol} 🚀 Executing {direction.upper()} trade | Qty: {qty} | Price: {price} | SL: {sl} | TP: {tp}")
    config = get_config()
    price_precision = config.get("scalper_settings", {}).get("symbol_precisions", {}).get(symbol, {}).get("pricePrecision", 8)
    qty_precision = config.get("scalper_settings", {}).get("symbol_precisions", {}).get(symbol, {}).get("quantityPrecision", 2)
    qty_step = get_qty_step_size(symbol)
    min_qty = config.get("scalper_settings", {}).get("symbol_precisions", {}).get(symbol, {}).get("minQuantity", 0.001)
    
    # Adjust quantity to meet notional value >= 5 USDT
    qty = round_to_step(qty, qty_step, qty_precision)
    notional = qty * price
    if notional < 5:
        logger.log_warning(f"{symbol} notional {notional} < 5 USDT, adjusting qty")
        qty = math.ceil(5 / price / (10 ** -qty_precision)) * (10 ** -qty_precision)
        qty = max(min_qty, round_to_step(qty, qty_step, qty_precision))
        notional = qty * price
        logger.log_info(f"{symbol} Adjusted qty: {qty}, notional: {notional}")

    # Check minimum quantity
    try:
        # --- SL/TP float casting and validation ---
        try:
            sl = float(sl)
            tp = float(tp)
        except Exception:
            logger.log_error(f"{symbol} ❌ Invalid SL/TP values sl={sl}, tp={tp}")
            return None

        from core.config import get_config
        from core.symbol_precision import get_precise_price
        config = get_config()
        scalper_settings = config.get("scalper_settings", {})
        min_sl_pct = scalper_settings.get("min_sl_distance_pct", 0.002)
        rr_ratio = scalper_settings.get("risk_reward_ratio", 1.4)

        if direction == "long":
            if sl >= price * (1 - min_sl_pct):
                sl = price * (1 - min_sl_pct)
            if tp <= price * (1 + min_sl_pct * rr_ratio):
                tp = price * (1 + min_sl_pct * rr_ratio)
        else:
            if sl <= price * (1 + min_sl_pct):
                sl = price * (1 + min_sl_pct)
            if tp >= price * (1 - min_sl_pct * rr_ratio):
                tp = price * (1 - min_sl_pct * rr_ratio)

        sl = get_precise_price(symbol, sl)
        tp = get_precise_price(symbol, tp)
        logger.log_info(f"{symbol} 🎯 Finalized SL/TP | Entry: {price:.4f} | SL: {sl:.4f} | TP: {tp:.4f}")

        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.log_error(f"{symbol} ❌ No LOT_SIZE filter found in symbol info")
            return {"status": "failed", "error": "No LOT_SIZE filter found in symbol info"}
        min_qty = float(lot_size_filter['minQty'])
        if qty < min_qty:
            logger.log_error(f"{symbol} ❌ Quantity {qty} is less than minimum {min_qty}")
            return {"status": "failed", "error": f"Quantity {qty} is less than minimum {min_qty}"}
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to fetch symbol info: {str(e)}")
        return {"status": "failed", "error": f"Failed to fetch symbol info: {str(e)}"}

    # Check wallet balance
    try:
        balances = client.futures_account_balance()
        wallet_balance = None
        for balance in balances:
            if balance['asset'] == 'USDT':
                wallet_balance = float(balance['balance'])
                break
        if wallet_balance is None:
            logger.log_error(f"{symbol} ❌ No USDT balance found in futures account")
            return {"status": "failed", "error": "No USDT balance found in futures account"}
        required_margin = (qty * price) / 20  # Assuming 20x leverage
        logger.log_info(f"{symbol} 💰 Wallet balance: {wallet_balance} USDT | Required margin: {required_margin} USDT")
        if wallet_balance < required_margin:
            logger.log_error(f"{symbol} ❌ Insufficient balance: {wallet_balance} USDT < {required_margin} USDT")
            return {"status": "failed", "error": f"Insufficient balance: {wallet_balance} USDT < {required_margin} USDT"}
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Failed to fetch wallet balance: {str(e)}")
        return {"status": "failed", "error": f"Failed to fetch wallet balance: {str(e)}"}

    if dry_run:
        logger.log_info(f"{symbol} 🧪 Dry run: {direction.upper()} | Qty: {qty} | Price: {price} | SL: {sl} | TP: {tp}")
        return {"status": "success", "order_id": "dry_run", "sl_order_id": "dry_run", "tp_order_id": "dry_run"}

    try:
        # Set leverage
        leverage = 20
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        logger.log_info(f"{symbol} ✅ Set leverage to {leverage}x")

        # Determine position side for Hedge Mode
        position_side = "LONG" if direction == "long" else "SHORT"
        side = SIDE_BUY if direction == "long" else SIDE_SELL

        # Place market order (no reduceOnly)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=qty,
            positionSide=position_side
        )
        logger.log_info(f"{symbol} ✅ Market order placed: {order}")

        # Place stop-loss order
        sl_side = SIDE_SELL if direction == "long" else SIDE_BUY
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type='STOP_MARKET',
            quantity=qty,
            stopPrice=round(sl, price_precision),
            positionSide=position_side,
            reduceOnly=True
        )
        logger.log_info(f"{symbol} ✅ SL order placed: {sl_order}")
        if sl <= 0:
            if get_config().get('alerts', {}).get('enabled', False):
                send_discord_log(f'{symbol} ⚠️ SL auto-adjusted before order placement')

        # Place take-profit order
        tp_side = SIDE_SELL if direction == "long" else SIDE_BUY
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TAKE_PROFIT_MARKET',
            quantity=qty,
            stopPrice=round(tp, price_precision),
            positionSide=position_side,
            reduceOnly=True
        )
        logger.log_info(f"{symbol} ✅ TP order placed: {tp_order}")
        if tp <= 0:
            if get_config().get('alerts', {}).get('enabled', False):
                send_discord_log(f'{symbol} ⚠️ TP auto-adjusted before order placement')

        # Update position manager
        try:
            position_manager.add_position(symbol, position_side.lower(), {
                "symbol": symbol,
                "direction": position_side.lower(),
                "entry_price": price,
                "size": qty,
                "stop_loss": sl,
                "take_profit": tp,
                "sl_order_id": sl_order.get("orderId"),
                "tp_order_id": tp_order.get("orderId"),
                "confidence": 1.0,
                "label": "scalper",
                "source": "trade_executor",
                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            logger.log_info(f"{symbol} ✅ Updated position manager")
        except Exception as e:
            logger.log_error(f"{symbol} ❌ Failed to update position manager: {str(e)}")

        return {
            "status": "success",
            "order_id": order.get("orderId"),
            "sl_order_id": sl_order.get("orderId"),
            "tp_order_id": tp_order.get("orderId")
        }
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Trade execution failed: {str(e)}")
        return {"status": "failed", "error": str(e)}