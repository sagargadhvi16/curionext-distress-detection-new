"""Configuration management utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file (relative to project root or absolute)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    # If relative path, try to resolve from project root
    if not config_path.is_absolute():
        # Try project root (where setup.py is)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_path
        
        # Also try configs/ directory
        if not config_path.exists():
            configs_dir = project_root / "configs" / config_path.name
            if configs_dir.exists():
                config_path = configs_dir
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        return {}
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones for overlapping keys.
    Nested dictionaries are merged recursively.

    Args:
        *configs: Variable number of config dicts to merge

    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}
    
    merged = {}
    
    for config in configs:
        if not isinstance(config, dict):
            continue
            
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_configs(merged[key], value)
            else:
                # Overwrite with new value
                merged[key] = value
    
    return merged


class Config:
    """
    Configuration manager class for handling model hyperparameters, paths, and settings.
    
    Usage:
        config = Config.from_files('configs/model_config.yaml', 'configs/training_config.yaml')
        audio_dim = config.get('audio_encoder.embedding_dim', default=1024)
        learning_rate = config.training.learning_rate
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        # Convert nested dicts to Config objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_files(cls, *config_paths: str) -> 'Config':
        """
        Load configuration from one or more YAML files.
        
        Args:
            *config_paths: Paths to YAML config files (later files override earlier ones)
            
        Returns:
            Config instance with merged configurations
        """
        configs = []
        for path in config_paths:
            configs.append(load_config(path))
        
        merged_config = merge_configs(*configs)
        return cls(merged_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(config_dict)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot-notation path.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'audio_encoder.embedding_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        result = {}
        for key, value in self._config.items():
            if isinstance(value, dict):
                # Convert nested Config objects back to dicts
                nested_config = getattr(self, key, None)
                if isinstance(nested_config, Config):
                    result[key] = nested_config.to_dict()
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._config = merge_configs(self._config, updates)
        # Update attributes
        for key, value in updates.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of Config."""
        return f"Config({self._config})"
