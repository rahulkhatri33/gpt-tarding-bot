# core/config.py
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from core.logger import global_logger as logger

# Global config cache
_CONFIG: Dict[str, Any] = {}

def _load_config() -> Dict[str, Any]:
    """Load config.json with proper error handling"""
    global _CONFIG
    if not _CONFIG:
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.json"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")

            with open(config_path) as f:
                _CONFIG = json.load(f)
            logger.log_info(f"✅ Config loaded from {config_path}")
            
        except json.JSONDecodeError as e:
            logger.log_error(f"❌ Invalid JSON in config: {e}")
            _CONFIG = {}
        except Exception as e:
            logger.log_error(f"❌ Failed to load config: {e}")
            _CONFIG = {}
    return _CONFIG

# Legacy CONFIG variable for backward compatibility
CONFIG = _load_config()

# ========================
# Core Configuration
# ========================

def is_dry_run_enabled() -> bool:
    return _load_config().get("dry_run", True)

def is_live_mode() -> bool:
    return _load_config().get("live_mode", False)

# ========================
# ML Configuration
# ========================

def get_confidence_thresholds() -> Dict[str, Dict[str, float]]:
    """Get ML confidence thresholds for all pairs"""
    return _load_config().get("ml_settings", {}).get("confidence_thresholds", {
        "default": {"long": 0.6, "short": 0.6}
    })

def get_ml_sl_pct() -> float:
    return _load_config().get("ml_settings", {}).get("sl_pct", 0.006)

def get_ml_tp_pct() -> float:
    return _load_config().get("ml_settings", {}).get("tp_pct", 0.012)

def get_triple_barrier_config() -> Dict[str, Any]:
    return _load_config().get("ml_settings", {}).get("triple_barrier_config", {
        "tp": 0.05,
        "sl": 0.03,
        "horizon_bars": 4
    })

# ========================
# Trading Configuration
# ========================

def get_max_concurrent_trades_by_source(source: str) -> int:
    return _load_config().get("max_concurrent_trades", {}).get(source.lower(), 1)

def get_cooldown_minutes_by_source(source: str) -> int:
    try:
        alias_map = {
            "5m_scalper": "scalper",
            "scalper": "scalper",
            "ml": "ml",
            "ML": "ml"
        }
        key = alias_map.get(source.lower(), source.lower())
        return _load_config().get("cooldown_minutes", {}).get(key, 10)
    except Exception as e:
        logger.log_error(f"❌ get_cooldown_minutes_by_source failed: {e}")
        return 10

def get_hold_limit_hours() -> int:
    return _load_config().get("hold_limit_hours", 36)

# ========================
# Allocation Functions
# ========================

def get_usd_allocation(pair: str, source: str = "ML") -> float:
    """Get USD allocation for a trading pair"""
    pair = pair.upper()
    if source.upper() == "ML":
        return _load_config().get("usd_allocation_ml", {}).get(pair, 200.0)
    return get_scalper_usd_allocation(pair)

def get_scalper_usd_allocation(pair: str) -> float:
    """Get scalper-specific allocation"""
    return _load_config().get("usd_allocation_scalper", {}).get(pair.upper(), 100.0)

# ========================
# SL/TP Configuration
# ========================

def get_scalper_fixed_sl_tp_pct(pair: str) -> Tuple[float, float]:
    """Get fixed SL/TP percentages"""
    group = _load_config().get("scalper_sl_tp_pct", {})
    pair_config = group.get(pair.upper(), group.get("default", {}))
    return (
        pair_config.get("sl", 0.02),
        pair_config.get("tp", 0.022)
    )

def get_scalper_config() -> Dict[str, Any]:
    """Get dynamic SL/TP settings"""
    return _load_config().get("scalper_settings", {
        "use_dynamic_sl_tp": True,
        "swing_sl_lookback": 5,
        "min_sl_distance_pct": 0.001,
        "risk_reward_ratio": 2,
        "fallback_sl_pct": 0.02,
        "fallback_tp_pct": 0.04,
        "enable_stc_confirmation": True
    })

# ========================
# Alert Configuration
# ========================

def get_discord_webhook() -> Optional[str]:
    return _load_config().get("alerts", {}).get("discord_webhook")

def get_discord_log_webhook() -> Optional[str]:
    return _load_config().get("alerts", {}).get("discord_log_webhook")

# ========================
# Watchdog Configuration
# ========================

def get_heartbeat_timeout_sec() -> int:
    return _load_config().get("watchdog", {}).get("heartbeat_timeout_sec", 120)

def get_watchdog_poll_interval_sec() -> int:
    return _load_config().get("watchdog", {}).get("poll_interval_sec", 30)

def get_sl_tp_buffer_pct() -> float:
    return _load_config().get("watchdog", {}).get("sl_tp_buffer_pct", 0.0003)

def get_config() -> Dict[str, Any]:
    """Get the full config dictionary"""
    return _load_config()

CONFIG = get_config()