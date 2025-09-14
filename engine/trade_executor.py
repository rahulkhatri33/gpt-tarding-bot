import traceback
from core.logger import global_logger as logger
from core.trade_executor import execute_trade as core_execute_trade

def execute_trade(symbol, direction, price, qty, sl, tp, source="unknown", confidence=1.0, label="engine"):
    """
    Engine-level executor wrapper.
    Sanitizes inputs before delegating to core.trade_executor.execute_trade.
    """

    try:
        sl = float(sl)
        tp = float(tp)
    except Exception as e:
        logger.log_error(f"{symbol} ❌ Engine received invalid SL/TP: sl={sl}, tp={tp} | {e}")
        return None

    logger.log_info(
        f"{symbol} ⚙️ Engine forwarding trade "
        f"| Dir: {direction.upper()} | Entry: {price:.4f} | Qty: {qty} | SL: {sl:.4f} | TP: {tp:.4f} "
        f"| Source: {source} | Conf: {confidence:.2f}"
    )

    try:
        return core_execute_trade(
            symbol, direction, price, qty, sl, tp,
            source=source, confidence=confidence, label=label
        )
    except Exception as e:
        logger.log_critical(f"{symbol} ❌ Engine failed to execute trade: {e}\n{traceback.format_exc()}")
        return None
