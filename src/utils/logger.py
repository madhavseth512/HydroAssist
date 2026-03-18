"""Structured logging setup for HydroAssist."""

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: Optional[str] = None, log_format: Optional[str] = None) -> logging.Logger:
    """
    Set up a structured logger with consistent formatting.

    Args:
        name: Logger name (usually module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Set level
    log_level = level or "INFO"
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Set format
    default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format or default_format)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger
