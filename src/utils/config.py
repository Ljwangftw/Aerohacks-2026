"""
config.py — Loads YAML configuration and provides easy access to parameters.

Usage:
    cfg = load_config()           # loads configs/default.yaml
    cfg = load_config("custom")   # loads configs/custom.yaml
    value = cfg["control"]["pid_x"]["kp"]
"""

import os
import yaml


def load_config(name: str = "default") -> dict:
    """Load a YAML config file from the configs/ directory.

    Args:
        name: config file name without extension (default: "default")

    Returns:
        Dictionary of configuration values.
    """
    # Resolve path relative to project root (two levels up from this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, "configs", f"{name}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_nested(cfg: dict, *keys, default=None):
    """Safely access nested config values.

    Usage:
        kp = get_nested(cfg, "control", "pid_x", "kp", default=1.0)
    """
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
