# core/analytics/trade_recorder.py
import os, csv, json, time
from datetime import datetime
from typing import Dict, List
from core.position_manager import position_manager
from utils.price_fetcher import get_latest_price
from core.logger import global_logger as logger

ROOT = "logs/trades_archive"
os.makedirs(ROOT, exist_ok=True)

FILES = {
    "trades":         os.path.join(ROOT, "trade_history.csv"),
    "lifecycle":      os.path.join(ROOT, "trade_lifecycle.csv"),
    "equity":         os.path.join(ROOT, "equity_curve.csv"),
    "diagnostics":    os.path.join(ROOT, "diagnostics.jsonl")
}

# ---------- CSV helpers ----------
def _append_csv(path: str, data: Dict):
    try:
        first = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data.keys())
            if first: w.writeheader()
            w.writerow(data)
    except Exception as e:
        logger.log_error(f"❌ Recorder write failed: {e}")

def append_trade(row: Dict):      _append_csv(FILES["trades"], row)
def append_lifecycle(row: Dict):  _append_csv(FILES["lifecycle"], row)

# ---------- Equity curve ----------
def _calc_equity() -> float:
    eq = 0.0
    for pos in position_manager.get_all_positions().values():
        price = get_latest_price(pos["symbol"]) or pos["entry_price"]
        size  = pos["size"]
        if pos["direction"] == "long":
            eq += (price - pos["entry_price"]) * size
        else:
            eq += (pos["entry_price"] - price) * size
    return eq

_equity_peak = 0
def snapshot_equity(tag: str = ""):
    global _equity_peak
    eq = _calc_equity()
    _equity_peak = max(_equity_peak, eq)
    dd = 0 if _equity_peak == 0 else (eq - _equity_peak) / _equity_peak * 100
    _append_csv(FILES["equity"], {
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "equity_usdt": round(eq, 4),
        "drawdown_pct": round(dd, 2)
    })

# ---------- Diagnostics ----------
def log_reject(symbol: str, filt: str, **features):
    line = {
        "ts": datetime.now().isoformat(),
        "symbol": symbol,
        "filter": filt,
        **features
    }
    try:
        with open(FILES["diagnostics"], "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception as e:
        logger.log_error(f"❌ Diagnostics write failed: {e}")
