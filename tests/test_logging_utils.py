"""Tests logging functions in style_transfer_visualizer."""
import logging

import style_transfer_visualizer.logging_utils as stv_logging_utils


class TestLoggingUtils:
    def test_logger_singleton_behavior(self) -> None:
        """Test that logger instances are singleton per name."""
        logger1 = stv_logging_utils.setup_logger("test_logger")
        logger2 = stv_logging_utils.setup_logger("test_logger")
        assert logger1 is logger2
        assert len(logger1.handlers) == 1

    def test_logger_custom_formatter_and_handler(self) -> None:
        """Test custom formatter and handler are applied."""
        formatter = logging.Formatter("[CUSTOM] %(message)s")
        handler = logging.StreamHandler()
        logger = stv_logging_utils.setup_logger(
            "custom_logger",
            formatter=formatter,
            handler=handler
        )
        assert logger.name == "custom_logger"
        assert len(logger.handlers) == 1
        assert logger.handlers[0].formatter._fmt.startswith("[CUSTOM]")
