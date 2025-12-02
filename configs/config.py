import os
from pathlib import Path

_DEFAULT_CFG = Path(__file__).resolve().parent / "config.yaml"
YAML_PATH = os.getenv("CONFIG_PATH", str(_DEFAULT_CFG))
