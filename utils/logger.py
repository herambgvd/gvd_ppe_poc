"""
# ======================================
# LOGGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module creates consistent application logging for all workers, Flask routes, database operations, and CV pipelines.
- It solves the common production problem where background camera threads fail silently.
- Enterprise systems use centralized structured logging so incidents can be diagnosed after deployment.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import CONFIG


_LOGGER_CACHE = {}


def get_logger(name: str = "ppe", level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    CONFIG.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    system_handler = RotatingFileHandler(
        CONFIG.LOG_DIR / "system.log", maxBytes=5_000_000, backupCount=5
    )
    system_handler.setFormatter(formatter)
    system_handler.setLevel(logging.INFO)

    error_handler = RotatingFileHandler(
        CONFIG.LOG_DIR / "errors.log", maxBytes=5_000_000, backupCount=5
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(system_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    _LOGGER_CACHE[name] = logger
    return logger
