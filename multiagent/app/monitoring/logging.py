"""
Centralized logging configuration for the application.

This module sets up a comprehensive logging system with:
- Configurable log levels
- Console and file logging
- Structured log formatting
- Environment-specific logging configurations
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logging(
    log_level: Optional[str] = None, 
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file (optional)
    """
    # Determine log level
    log_level = (log_level or os.getenv('LOG_LEVEL', 'INFO')).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers to prevent duplicate logging
    root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Create a rotating file handler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(numeric_level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Could not set up file logging: {e}")

    # Disable propagation to prevent duplicate log messages
    root_logger.propagate = False

    # Optional: Set logging for specific libraries
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger (optional, defaults to root logger)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)