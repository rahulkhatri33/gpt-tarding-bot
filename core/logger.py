import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
import sys

class YogiLogger:
    def __init__(self, name, log_dir="logs", max_bytes=10*1024*1024, backup_count=5):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        os.makedirs(log_dir, exist_ok=True)

        self._configure_handlers(name, log_dir, max_bytes, backup_count)

        self._logged_messages = set()

    def _configure_handlers(self, name, log_dir, max_bytes, backup_count):
        self.file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        self.file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] - %(message)s'
        ))

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    # Standard logging methods
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def log_info(self, msg, *args, **kwargs):
        self.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def log_warning(self, msg, *args, **kwargs):
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args, exc_info=False, **kwargs):
        self.logger.error(msg, *args, exc_info=exc_info, **kwargs)

    def log_error(self, msg, *args, exc_info=False, **kwargs):
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def log_debug(self, msg, *args, **kwargs):
        self.debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def log_critical(self, msg, *args, **kwargs):
        self.critical(msg, *args, **kwargs)

    # One-time logger
    def log_once(self, msg, level="info"):
        if msg not in self._logged_messages:
            self._logged_messages.add(msg)
            level = level.lower()
            if hasattr(self, level):
                getattr(self, level)(msg)
            else:
                self.info(msg)

    def force_rollover(self):
        self.file_handler.doRollover()
        self.info("Log files rotated")

    def log_live_feed(self, message: str):
        try:
            encoded_msg = message.encode('utf-8', errors='replace').decode('utf-8')
            self.info(f"[LIVE] {encoded_msg}")
        except Exception as e:
            self.error(f"Failed to log live feed: {str(e)}")

    # 🆕 Scalper signal logging
    def log_scalper_signal(self, msg):
        self.info(f"[SCALPER] {msg}")

    # 🆕 ML signal logging
    def log_ml_signal(self, msg):
        self.info(f"[ML] {msg}")

    # 🆕 Trade execution logging
    def log_trade(self, msg: str, level: str = "info"):
        tag = "📊 TRADE"
        formatted = f"{tag} {msg}"
        level = level.lower()

        if hasattr(self, level):
            getattr(self, level)(formatted)
        else:
            self.info(formatted)

# Global logger instance
global_logger = YogiLogger("trading_bot")
