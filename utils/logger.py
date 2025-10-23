"""
Logging utilities for the ingestion pipeline.
Provides structured logging with consistent formatting.
"""
import logging
import sys
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Format
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


class IngestionLogger:
    """
    Specialized logger for ingestion operations.
    Tracks metrics and provides structured logging.
    """
    
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = setup_logger(name, level)
        self.start_time: Optional[datetime] = None
        self.metrics = {
            'fetched': 0,
            'validated': 0,
            'written': 0,
            'dropped': 0,
            'errors': 0
        }
    
    def start(self, operation: str):
        """Start an ingestion operation."""
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting {operation}")
        self.logger.info(f"   Timestamp: {self.start_time.isoformat()}")
    
    def complete(self, operation: str):
        """Complete an ingestion operation with summary."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"✅ Completed {operation}")
            self.logger.info(f"   Duration: {duration:.2f}s")
        else:
            self.logger.info(f"✅ Completed {operation}")
        
        # Print metrics
        self.logger.info(f"   Metrics:")
        for key, value in self.metrics.items():
            if value > 0:
                self.logger.info(f"     - {key.capitalize()}: {value:,}")
    
    def increment(self, metric: str, count: int = 1):
        """Increment a metric counter."""
        if metric in self.metrics:
            self.metrics[metric] += count
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.metrics['errors'] += 1
    
    def error(self, msg: str, exc_info=False):
        """Log error message."""
        self.logger.error(msg, exc_info=exc_info)
        self.metrics['errors'] += 1
    
    def critical(self, msg: str, exc_info=False):
        """Log critical message."""
        self.logger.critical(msg, exc_info=exc_info)
        self.metrics['errors'] += 1
