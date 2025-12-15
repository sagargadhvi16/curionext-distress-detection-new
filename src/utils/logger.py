"""Logging utilities."""
import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Configured logger

    TODO: Implement logger setup
    """
    pass  # To be implemented
