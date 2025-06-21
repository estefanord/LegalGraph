
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .config import settings


class StructuredFormatter(logging.Formatter):
    
    def format(self, record: logging.LogRecord) -> str:
        # Create the base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    enable_console: bool = True,
    enable_file: bool = True,
    structured: bool = True
) -> None:
    
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Create data directory if it doesn't exist
    Path(settings.data_directory).mkdir(exist_ok=True)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        log_path = Path(settings.data_directory) / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class LogContext:
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_context = {}
    
    def __enter__(self):
        # Store old context and add new context
        for key, value in self.context.items():
            if hasattr(self.logger, key):
                self.old_context[key] = getattr(self.logger, key)
            setattr(self.logger, key, value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        for key in self.context:
            if key in self.old_context:
                setattr(self.logger, key, self.old_context[key])
            else:
                delattr(self.logger, key)


def log_execution_time(logger: logging.Logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.info(
                    f"Function {func.__name__} completed",
                    extra={
                        "function": func.__name__,
                        "execution_time_seconds": execution_time,
                        "success": True
                    }
                )
                return result
            except Exception as e:
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        "function": func.__name__,
                        "execution_time_seconds": execution_time,
                        "success": False,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


# Initialize logging when module is imported
setup_logging() 