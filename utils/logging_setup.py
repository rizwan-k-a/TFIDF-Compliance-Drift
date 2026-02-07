"""Centralized logging configuration for TF-IDF Compliance Drift System.

This module provides structured logging setup that works across backend,
frontend, and utility modules without requiring Streamlit in the backend.

Usage:
    # In any module:
    from utils.logging_setup import get_logger
    logger = get_logger(__name__)
    
    logger.info("Processing document: %s", filename)
    logger.warning("Vectorization degraded for %d docs", len(docs))
    logger.error("Classification failed: %s", e)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# Global flag to track if logging has been initialized
_LOGGING_INITIALIZED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Initialize structured logging for the entire application.

    This function sets up logging once at application startup. Subsequent calls
    are safely ignored to prevent duplicate handlers.

    Args:
        level: Logging level (default: INFO). Use logging.DEBUG for verbose output.
        log_file: Optional file path to write logs. If None, logs to stdout only.
        format_string: Custom log format. Default includes timestamp, logger name,
                      level, and message.

    Example:
        >>> setup_logging(level=logging.INFO)
        >>> logger = get_logger("backend.tfidf_engine")
        >>> logger.info("TF-IDF vectorization started")
    """
    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED:
        return

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Optionally add file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            # If we can't write to file, log warning but continue
            root_logger.warning("Could not create log file %s: %s", log_file, e)

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.

    This function combines logging.getLogger() with automatic initialization.
    It's safe to call from any module at any time.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        logging.Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
        >>> logger.critical("Critical message")
    """
    # Ensure logging is initialized at least once
    if not _LOGGING_INITIALIZED:
        setup_logging()

    return logging.getLogger(name)
