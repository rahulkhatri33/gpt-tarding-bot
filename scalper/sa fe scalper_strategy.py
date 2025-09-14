# scalper/scalper_strategy.py (VVCS Strategy)

import pandas as pd
from typing import Optional
from datetime import datetime

from core.logger import global_logger as logger
from engine.sl_tp_engine import calculate_scalper_sl_tp
from core.config import get_scalper_usd_allocation
from core.symbol_precision import get_trimmed_quantity
from ml_engine.ml_inference import ml_inference_cache as ml_cache


def evaluate_scalper_entry(symbol: str,
                           df: pd.DataFrame,
                           params: dict | None = None) -> Optional[dict]:
    # ---------- guards ----------
    if df is None or len(df) < 50:
        logger.log_scalper_signal(f"{symbol} ❌ Not enough data rows ({len(df)})")
        return None

    params = params or {}
    mfi_upper      = params.get("mfi_upper", 65)   # for longs (overbought filter)
    mfi_lower      = params.get("mfi_lower",35)   # for shorts (oversold filter)
    volume_mult    = params.get("volume_multiplier", 2.2)

    last           = df.iloc[-1]
    close          = last["close"]
    volume         = last["volume"]

    # ---------- OBV + Vortex trend ----------
    obv            = df["OBV"]
    vortex_pos     = df["VORTEX_POS"].iloc[-1]
    vortex_neg     = df["VORTEX_NEG"].iloc[-1]

    obv_up   = obv.iloc[-1] > obv.iloc[-3]
    obv_down = obv.iloc[-1] < obv.iloc[-3]

    trend = (
        "long"  if obv_up   and vortex_pos > vortex_neg else
        "short" if obv_down and vortex_neg > vortex_pos else
        None
    )
    if trend is None:
        logger.log_scalper_signal(f"{symbol} ❌ OBV/Vortex trend misaligned.")
        return None

    # ---------- directional checks ----------
    if trend == "long":
        recent_high = df["high"].iloc[-5:-1].max()
        if close <= recent_high:
            logger.log_scalper_signal(
                f"{symbol}-long ❌ Breakout failed. Close {close:.4f} ≤ High {recent_high:.4f}"
            )
            return None

        mfi = df["MFI"].iloc[-1]
        if mfi >= mfi_upper:
            logger.log_scalper_signal(f"{symbol}-long ❌ MFI too high ({mfi:.2f})")
            return None

    else:  # trend == "short"
        recent_low = df["low"].iloc[-5:-1].min()
        if close >= recent_low:
            logger.log_scalper_signal(
                f"{symbol}-short ❌ Breakdown failed. Close {close:.4f} ≥ Low {recent_low:.4f}"
            )
            return None

        mfi = df["MFI"].iloc[-1]
        if mfi <= mfi_lower:
            logger.log_scalper_signal(f"{symbol}-short ❌ MFI too low ({mfi:.2f})")
            return None

    # ---------- common volume filter ----------
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    if volume < volume_mult * vol_avg:
        logger.log_scalper_signal(
            f"{symbol}-{trend} ❌ Volume weak ({volume:.2f} < {volume_mult:.2f}×{vol_avg:.2f})"
        )
        return None

    logger.log_scalper_signal(
        f"✅ {symbol}-{trend.upper()} VVCS Entry | OBV+Vortex OK | "
        f"{'Breakout' if trend=='long' else 'Breakdown'} ✅ | "
        f"Volume OK ({volume:.2f}) | MFI={mfi:.2f}"
    )

    return {
        "symbol": symbol,
        "direction": trend,
        "score": 4,
        "confidence": 0.95,
        "entry_price": close,
        "type": "vvcs_obv_vortex_mfi"
    }

def build_trade_request(
    pair: str,
    direction: str,
    score: float,
    entry_price: float,
    label: Optional[int] = None,
    override: bool = False
) -> Optional[dict]:
    from engine.sl_tp_engine import calculate_scalper_sl_tp
    from core.symbol_precision import get_trimmed_quantity
    from core.config import get_scalper_usd_allocation

    # ✅ Enforce long-only Spot logic
    if direction != "long":
        return None

    _, tp = calculate_scalper_sl_tp(pair, entry_price, direction)

    # ✅ TP is mandatory; SL is always None in Spot mode
    if tp is None:
        from core.logger import global_logger as logger
        logger.log_error(f"{pair}-{direction} ❌ TP calculation failed. Skipping trade.")
        return None

    usd_alloc = get_scalper_usd_allocation(pair)
    qty = get_trimmed_quantity(pair, usd_alloc / entry_price)

    if qty <= 0:
        from core.logger import global_logger as logger
        logger.log_error(f"{pair} ❌ Invalid trade quantity after trimming: {qty}")
        return None

    confidence = 0.8  # Placeholder until ML score used
    request = {
        "symbol": pair,
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "label": label,
        "entry_price": entry_price,
        "quantity": qty,
        "source": "5M_SCALPER",
        "timestamp": datetime.now().astimezone().isoformat(),
        "override": override
    }

    return request
