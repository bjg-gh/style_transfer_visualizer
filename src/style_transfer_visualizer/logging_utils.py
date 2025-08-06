"""
Centralized logging utilities for the style transfer application.

Defines a shared logger instance and setup function to ensure consistent
logging configuration across the codebase. Centralization also avoids
circular imports and simplifies future extensions like file logging
or verbosity control.
"""

import logging


def setup_logger(
        name: str = __name__,
        level: int = logging.INFO,
        formatter: logging.Formatter | None = None,
        handler: logging.Handler | None = None,
) -> logging.Logger:
    """
    Configure a logger with optional custom formatting and handler.

    Creates a module-level logger with sensible defaults for level and
    formatting. Custom handlers and formatters can be supplied if
    needed.

    Args:
        name: Logger name, typically set to __name__.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        formatter: Optional custom formatter.
        handler: Optional custom handler.

    Returns:
        A configured logger instance.

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
