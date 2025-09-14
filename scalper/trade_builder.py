# scalper/trade_builder.py

from datetime import datetime
from typing import Optional, Dict
from core.config import get_scalper_usd_allocation
from engine.sl_tp_engine import calculate_scalper_sl_tp
from core.symbol_precision import get_trimmed_quantity
from core.logger import global_logger as logger

def build_trade_request(
    pair: str,
    direction: str,
    score: float,
    entry_price: float,
    df,
    label: Optional[int],
    override: bool
) -> Optional[Dict]:
    try:
        usd_allocation = get_scalper_usd_allocation(pair)
        sl, tp = calculate_scalper_sl_tp(pair, entry_price, direction)

        if not isinstance(sl, (float, int)) or not isinstance(tp, (float, int)):
            logger.log_critical(f"❌ SL/TP returned invalid types for {pair}-{direction}: SL={type(sl)}, TP={type(tp)} | SL={sl} | TP={tp}")
            return None

        if not sl or not tp:
            logger.log_error(f"❌ SL/TP calculation failed for {pair}-{direction}.")
            return None

        stop_distance = abs(entry_price - sl)
        tp_distance = abs(tp - entry_price)

        min_sl_distance = entry_price * 0.001
        min_tp_distance = entry_price * 0.0012

        if stop_distance < min_sl_distance:
            logger.log_error(f"❌ SL distance too small for {pair}-{direction}: {stop_distance:.8f} < {min_sl_distance:.8f}")
            return None

        if tp_distance < min_tp_distance:
            logger.log_error(f"❌ TP distance too small for {pair}-{direction}: {tp_distance:.8f} < {min_tp_distance:.8f}")
            return None

        quantity = usd_allocation / entry_price
        quantity = get_trimmed_quantity(pair, quantity)

        if quantity <= 0:
            logger.log_error(f"❌ Quantity too small after trimming for {pair}-{direction}.")
            return None

        confidence = min(0.95, max(0.0, score / 11.0))

        return {
            "symbol": pair.upper(),
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry_price,
            "quantity": quantity,
            "source": "5M_SCALPER",
            "label": label,
            "override": override,
            "timestamp": datetime.now().astimezone().isoformat()
        }

    except Exception as e:
        logger.log_error(f"❌ Trade request build error for {pair}-{direction}: {e}")
        return None
