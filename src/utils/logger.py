"""Logging utilities with structured logging support."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_format: Optional[str] = None,
    file_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Creates a logger with:
    - Console handler with colored output (DEBUG, INFO, WARNING, ERROR)
    - File handler with rotation (if log_file specified)
    - Structured formatting with timestamps, levels, and messages

    Args:
        name: Logger name (typically __name__ of the module)
        log_file: Name of log file (optional). If None, no file logging.
                 If relative path, will be placed in log_dir.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/ in project root)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        console_format: Custom format for console output
        file_format: Custom format for file output

    Returns:
        Configured logger instance

    Example:
        logger = setup_logger(__name__, log_file='api.log', level=logging.DEBUG)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Default formats
    if console_format is None:
        console_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
    if file_format is None:
        file_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (if log_file specified)
    if log_file:
        # Determine log directory
        if log_dir is None:
            # Use project root/logs/
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine full log file path
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = log_dir / log_path
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.
    
    This is a convenience function that uses default settings.
    For custom configuration, use setup_logger() directly.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Setup with defaults if not already configured
        return setup_logger(name, log_file=f"{name.split('.')[-1]}.log")
    return logger
