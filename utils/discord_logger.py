import requests
import time

# ❌ Avoid top-level logger/config import to prevent circular dependency
# ✅ Use lazy imports inside the function

def send_discord_log(message: str, tag: str = "📣", retry: int = 3):
    """
    Sends a message to the Discord logging channel (separate from trade alerts).
    """
    try:
        from core.config import get_discord_log_webhook
        LOG_WEBHOOK_URL = get_discord_log_webhook()
    except Exception as e:
        try:
            from core.logger import global_logger as logger
            logger.log_error(f"⚠️ Failed to fetch log webhook URL: {e}")
        except:
            pass
        return

    if not LOG_WEBHOOK_URL:
        try:
            from core.logger import global_logger as logger
            logger.log_once("⚠️ DISCORD_LOG_WEBHOOK not set in config.json.")
        except:
            pass
        return

    payload = {"content": f"{tag} {message}"}

    for i in range(retry):
        try:
            resp = requests.post(LOG_WEBHOOK_URL, json=payload)
            if resp.status_code != 204:
                from core.logger import global_logger as logger
                logger.log_error(f"⚠️ Discord log hook failed ({resp.status_code}): {resp.text}")
            return
        except Exception as e:
            if i == retry - 1:
                from core.logger import global_logger as logger
                logger.log_error(f"❌ Failed to send Discord log after retries: {e}")
            time.sleep(2 ** i)
