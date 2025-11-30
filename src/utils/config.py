"""
Configuration loader utility
"""

import yaml
import os
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        config: Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_config(config: dict) -> bool:
    """
    Validate configuration values
    
    Args:
        config: Configuration dictionary
    
    Returns:
        valid: True if configuration is valid
    """
    required_keys = ['model', 'video', 'crowd', 'heatmap', 'dashboard']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration section: {key}")
            return False
    
    # Validate threshold values
    if not 0 <= config['model']['confidence_threshold'] <= 1:
        logger.error("Invalid confidence threshold (must be 0-1)")
        return False
    
    if not 0 <= config['heatmap']['alpha'] <= 1:
        logger.error("Invalid heatmap alpha (must be 0-1)")
        return False
    
    logger.info("✓ Configuration validation passed")
    return True
