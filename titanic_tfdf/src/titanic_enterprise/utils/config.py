"""Configuration management utilities using Hydra."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_name: str = "config",
    config_dir: Optional[str] = None,
    overrides: Optional[list] = None,
) -> DictConfig:
    """
    Load configuration using Hydra.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        config_dir: Directory containing config files. If None, uses default.
        overrides: List of config overrides in Hydra format
        
    Returns:
        DictConfig: Loaded configuration
        
    Example:
        >>> cfg = load_config("config", overrides=["model.num_trees=500"])
        >>> print(cfg.model.num_trees)
        500
    """
    if overrides is None:
        overrides = []
        
    if config_dir is None:
        # Get the project root directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        config_dir = str(project_root / "configs")
    
    # Ensure config directory exists
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    
    return cfg


def save_config(cfg: DictConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        cfg: Configuration to save
        output_path: Path where to save the config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        OmegaConf.save(cfg, f)


def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_cfg: Base configuration
        override_cfg: Override configuration
        
    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base_cfg, override_cfg)


def validate_config(cfg: DictConfig, required_keys: list) -> None:
    """
    Validate that required keys exist in configuration.
    
    Args:
        cfg: Configuration to validate
        required_keys: List of required keys in dot notation (e.g., "model.num_trees")
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key in required_keys:
        try:
            OmegaConf.select(cfg, key)
        except Exception:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_config_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get a configuration value with optional default.
    
    Args:
        cfg: Configuration object
        key: Key in dot notation (e.g., "model.num_trees")
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value or default
    """
    try:
        return OmegaConf.select(cfg, key)
    except Exception:
        return default


def update_config(cfg: DictConfig, updates: Dict[str, Any]) -> DictConfig:
    """
    Update configuration with new values.
    
    Args:
        cfg: Configuration to update
        updates: Dictionary of updates in dot notation
        
    Returns:
        DictConfig: Updated configuration
    """
    cfg_copy = OmegaConf.create(cfg)
    
    for key, value in updates.items():
        OmegaConf.update(cfg_copy, key, value)
    
    return cfg_copy