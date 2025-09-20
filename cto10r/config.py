\
import yaml, os
from dataclasses import dataclass
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
