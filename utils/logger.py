import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Global logger cache, to avoid duplicate initialization
_logger_cache: dict[str, logging.Logger] = {}


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get logger, supporting file logging and log rotation
    
    Args:
        name: logger name, usually using __name__
        log_file: optional log file path, if provided, also output to file
        
    Returns:
        logging.Logger instance
    """
    # If logger already exists and is configured, return directly
    if name in _logger_cache:
        return _logger_cache[name]
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate adding handler
    if logger.handlers:
        _logger_cache[name] = logger
        return logger
    
    # Get log level from environment variables
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    # Improved log format: includes module name, function name, and line number
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simplified console format (more readable)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified)
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler to implement log rotation
        # maxBytes=10MB, backupCount=5
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Cache logger
    _logger_cache[name] = logger
    
    return logger


def setup_root_logger(log_file: Optional[str] = None):
    """
    Setup root logger configuration
    
    Args:
        log_file: optional log file path
    """
    root_logger = logging.getLogger()
    
    # Get log level from environment variables
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    root_logger.setLevel(log_level)
    
    # If root logger already has handler, do not add again
    if root_logger.handlers:
        return
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
