"""Configuration loader for the rehabilitation exercise grading system."""
import os
import yaml


def load_config(config_path=None):
    """Load YAML config. Default: config/config.yaml relative to project root."""
    if config_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
