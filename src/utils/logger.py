"""
Logging utility setup
"""

import logging
import os
from datetime import datetime


def setup_logger(config: dict) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        logger: Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    save_logs = log_config.get('save_logs', True)
    log_file = log_config.get('log_file', 'logs/system.log')
    
    # Create logs directory if it doesn't exist
    if save_logs:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # File handler
    if save_logs:
        # Add timestamp to log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_with_timestamp = log_file.replace('.log', f'_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file_with_timestamp)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=handlers
    )
    
    logger = logging.getLogger('Zaytrics')
    logger.info("=" * 80)
    logger.info("Zaytrics Smart Crowd Monitoring System - Version 1.0")
    logger.info("Organization: SEECS, NUST")
    logger.info("=" * 80)
    
    return logger
