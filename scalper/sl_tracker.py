import time
from core.logger import global_logger as logger
from core.config import CONFIG

_scalper_sl_streak = 0
_scalper_hibernating_until = 0

def is_scalper_hibernating() -> bool:
    is_hibernating = time.time() < _scalper_hibernating_until
    if is_hibernating:
        logger.log_info(f"🌙 Scalper hibernating until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_scalper_hibernating_until))}")
    else:
        logger.log_debug("✅ Scalper not in hibernation")
    return is_hibernating

def trigger_scalper_hibernation():
    global _scalper_hibernating_until
    cooldown = CONFIG.get("scalper_hibernation_minutes", 30)
    _scalper_hibernating_until = time.time() + cooldown * 60
    logger.log_critical(f"🌙 Scalper entered hibernation for {cooldown} minutes due to SL streak.")

def reset_scalper_sl_streak():
    global _scalper_sl_streak
    _scalper_sl_streak = 0
    logger.log_info("🔄 Scalper SL streak reset")

def record_scalper_sl_hit():
    global _scalper_sl_streak
    _scalper_sl_streak += 1
    threshold = CONFIG.get("scalper_max_sl_streak", 3)
    logger.log_info(f"📉 Scalper SL hit recorded. Streak: {_scalper_sl_streak}/{threshold}")
    if _scalper_sl_streak >= threshold:
        trigger_scalper_hibernation()
        _scalper_sl_streak = 0