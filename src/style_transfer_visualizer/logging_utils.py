"""
logging_utils.py

Shared application-wide logger instance and setup function.

Usage:
    from style_transfer_visualizer.logging_utils import logger

Why this exists:
    Centralizing logger creation in its own isolated module prevents
    circular imports.

All log messages throughout the application should use this logger for
consistency in formatting, level control, and future extensibility
(e.g., file logging or verbosity flags).
"""
import logging


def setup_logger(
        name: str = __name__,
        level: int = logging.INFO,
        formatter: logging.Formatter = None,
        handler: logging.Handler = None
) -> logging.Logger:
    """Configure and return a module-level logger with optional custom
    settings.

    Args:
        name: Name of the module
        level: Level of the logger
        formatter: Format for log messages
        handler: Custom handler for log messages

    Returns:
        Logger: A module-level logger with optional custom settings.
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    if not logger_instance.handlers:
        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s")
        if handler is None:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.propagate = False
    return logger_instance


# Shared logger used across modules
logger = setup_logger("style_transfer")
