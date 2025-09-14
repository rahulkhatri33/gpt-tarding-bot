# utils/notifier.py

import json
import os
import time
import requests
from typing import Optional, Dict

from core.config import CONFIG, get_discord_log_webhook

class Notifier:
    def __init__(self):
        alerts_config = CONFIG.get("alerts", {})
        self.webhook_url = alerts_config.get("discord_webhook", None)
        self.log_webhook_url = alerts_config.get("discord_log_webhook", None)
        self.alert_enabled = alerts_config.get("enabled", False)
        self.exits_csv = os.path.join("logs", "trade_exits.csv")

    def send_trade_alert(
        self,
        symbol: str,
        direction: str,
        price: float,
        qty: float,
        confidence: float,
        dry_run: bool,
        source: str,
        label: Optional[int] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        trailing_active: Optional[bool] = None
    ) -> None:
        if not self.alert_enabled or not self.webhook_url:
            from core.logger import global_logger as logger
            logger.log_info(f"📬 Trade alert skipped: {symbol}-{direction}")
            return

        emoji = "🟢" if direction == "long" else "🔴"
        live_status = "LIVE" if not dry_run else "DRY RUN"

        message = (
            f"**{emoji} {live_status} TRADE OPENED**\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Entry Price: {price:.4f}\n"
            f"Quantity: {qty:.4f}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Source: {source}\n"
        )

        if label is not None:
            message += f"Label: {label}\n"
        if sl_price and tp_price:
            message += f"SL: {sl_price:.5f} | TP: {tp_price:.5f}\n"
        if trailing_active:
            message += f"Trailing: Enabled\n"

        self._send(message)

    def send_exit_alert(
        self,
        symbol: str,
        exit_type: str,
        price: float,
        qty: float,
        reason: Optional[str] = "",
        direction: Optional[str] = "",
        pnl: Optional[float] = None,
        time_str: Optional[str] = None
    ) -> None:
        if not self.alert_enabled or not self.webhook_url:
            from core.logger import global_logger as logger
            logger.log_info(f"📬 Exit alert skipped: {symbol}-{direction}")
            return

        emoji = "🏁"
        timestamp = time_str if time_str else time.strftime("%Y-%m-%d %H:%M", time.localtime())

        message = (
            f"**{emoji} TRADE CLOSED**\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper() if direction else 'N/A'}\n"
            f"Exit Type: {exit_type}\n"
            f"Exit Price: {price:.4f}\n"
            f"Quantity: {qty:.4f}\n"
            f"PnL: {pnl:.2f}\n"
            f"Time: {timestamp}\n"
            f"Reason: {reason or exit_type}"
        )

        self._send(message)
        self._log_exit_to_csv({
            "symbol": symbol,
            "direction": direction,
            "exit_type": exit_type,
            "exit_price": price,
            "qty": qty,
            "pnl": pnl,
            "timestamp": timestamp,
            "reason": reason
        })

    def send_error(self, error_message: str, priority: str = "warning") -> None:
        if not self.alert_enabled or not self.webhook_url:
            return

        prefix = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }.get(priority, "⚠️")

        message = f"{prefix} {error_message}"
        self._send(message)

    def send_info(self, msg: str) -> None:
        self.send_error(msg, priority="info")

    def send_critical(self, critical_message: str) -> None:
        """Send to critical log webhook if available, else fallback."""
        webhook = self.log_webhook_url or self.webhook_url
        if not webhook:
            from core.logger import global_logger as logger
            logger.log_error(f"❌ Critical alert lost (no webhook): {critical_message}")
            return

        try:
            payload = {"content": f"🚨 **CRITICAL** 🚨\n{critical_message}"}
            requests.post(webhook, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
        except Exception as e:
            from core.logger import global_logger as logger
            logger.log_error(f"❌ Failed to send critical alert: {e}")

    def _send(self, message: str, webhook_override: Optional[str] = None) -> None:
        webhook = webhook_override or self.webhook_url
        if not webhook:
            return

        payload = {"content": message}
        for attempt in range(3):
            try:
                response = requests.post(webhook, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
                if response.status_code == 204:
                    break  # success
            except Exception as e:
                from core.logger import global_logger as logger
                logger.log_warning(f"⚠️ Webhook send attempt {attempt+1} failed: {e}")
            time.sleep(1)

    def _log_exit_to_csv(self, row: Dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.exits_csv), exist_ok=True)
            file_exists = os.path.isfile(self.exits_csv)

            with open(self.exits_csv, "a") as f:
                if not file_exists:
                    header = "symbol,direction,exit_type,exit_price,qty,pnl,timestamp,reason\n"
                    f.write(header)
                line = f"{row['symbol']},{row['direction']},{row['exit_type']},{row['exit_price']},{row['qty']},{row['pnl']},{row['timestamp']},{row['reason']}\n"
                f.write(line)
        except Exception as e:
            from core.logger import global_logger as logger
            logger.log_warning(f"⚠️ Failed to write trade exit to CSV: {e}")

# Singleton instance
notifier = Notifier()
