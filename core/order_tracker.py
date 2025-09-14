# core/order_tracker.py
import threading
import time
from typing import Optional, Dict

_ORDER_TRACKER: Dict[str, dict] = {}
_TRACKER_LOCK = threading.Lock()


def _key(symbol: str, direction: str) -> str:
    return f"{symbol.upper()}|{direction.lower()}"


def track_entry(symbol: str, direction: str, order_id: str, source: str) -> None:
    with _TRACKER_LOCK:
        _ORDER_TRACKER[_key(symbol, direction)] = {
            "state": "ENTRY_PENDING",
            "order_id": order_id,
            "timestamp": time.time(),
            "source": source
        }


def mark_open(symbol: str, direction: str) -> None:
    with _TRACKER_LOCK:
        key = _key(symbol, direction)
        if key in _ORDER_TRACKER:
            _ORDER_TRACKER[key]["state"] = "OPEN"
            _ORDER_TRACKER[key]["timestamp"] = time.time()


def mark_exit_pending(symbol: str, direction: str) -> bool:
    with _TRACKER_LOCK:
        key = _key(symbol, direction)
        state = _ORDER_TRACKER.get(key, {}).get("state")
        if state == "EXIT_PENDING":
            return False
        _ORDER_TRACKER[key] = {
            "state": "EXIT_PENDING",
            "order_id": None,
            "timestamp": time.time()
        }
        return True


def is_exit_pending(symbol: str, direction: str) -> bool:
    with _TRACKER_LOCK:
        key = _key(symbol, direction)
        return _ORDER_TRACKER.get(key, {}).get("state") == "EXIT_PENDING"


def clear(symbol: str, direction: str) -> None:
    with _TRACKER_LOCK:
        key = _key(symbol, direction)
        _ORDER_TRACKER.pop(key, None)


def get_lifecycle_state(symbol: str, direction: str) -> Optional[str]:
    with _TRACKER_LOCK:
        return _ORDER_TRACKER.get(_key(symbol, direction), {}).get("state")


def get_all() -> Dict[str, dict]:
    with _TRACKER_LOCK:
        return dict(_ORDER_TRACKER)
