"""
Structured logging utility for the adaptive fraud detection system.

Provides consistent, color-coded console output and optional file logging
for tracking pipeline progress, drift events, and RL training metrics.
"""

import logging
import os
import sys
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for console output."""

    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = None,
    log_to_file: bool = False,
) -> logging.Logger:
    """
    Create a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    log_dir : str, optional
        Directory for log files. Created if it doesn't exist.
    log_to_file : bool
        If True, also write logs to a timestamped file in log_dir.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger
