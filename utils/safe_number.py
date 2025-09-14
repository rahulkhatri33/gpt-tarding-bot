# utils/safe_number.py
"""
Small, defensive numeric helpers.
Use these to coerce incoming values (possibly dicts/strings) into floats or None.
"""

from decimal import Decimal, InvalidOperation
from typing import Any, Optional

COMMON_NUMERIC_KEYS = ("value", "qty", "quantity", "size", "amount", "price", "minQty", "stepSize", "min_notional", "minNotional")

def to_float_or_none(x: Any) -> Optional[float]:
    """Try to coerce x to float. If x is a dict, try common numeric keys inside it.
    Return None when not parseable.
    """
    try:
        if x is None:
            return None
        # If dict, look for numeric-like keys
        if isinstance(x, dict):
            for k in COMMON_NUMERIC_KEYS:
                if k in x:
                    return to_float_or_none(x[k])
            # nothing useful inside dict
            return None
        # single-element list/tuple -> use first element
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return to_float_or_none(x[0])
        # numpy/pandas scalars
        if hasattr(x, "item"):
            try:
                return float(x.item())
            except Exception:
                pass
        # Decimal/str/int/float -> convert via Decimal for stability
        try:
            return float(Decimal(str(x)))
        except (InvalidOperation, ValueError, TypeError):
            # final fallback try float()
            return float(x)
    except Exception:
        return None
