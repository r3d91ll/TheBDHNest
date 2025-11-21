"""Configuration management for TheBDHNest

Phase 0: Simple YAML loading
Phase 1: Will expand for TheNest platform
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Default config location
DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"
EXAMPLE_CONFIG = Path(__file__).parent.parent / "config.yaml.example"

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file

    Args:
        config_path: Path to config file (default: ./config.yaml)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy {EXAMPLE_CONFIG} to {config_path} and customize"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)

def get_experiments_root(config: Optional[Dict[str, Any]] = None) -> Path:
    """Get experiments root directory"""
    if config is None:
        config = load_config()
    return Path(config['paths']['experiments_root']).expanduser().resolve()

def get_monitoring_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get monitoring settings"""
    if config is None:
        config = load_config()
    return config['monitoring']

def get_gpu_devices(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Get GPU device assignments"""
    if config is None:
        config = load_config()
    return config['gpu']
