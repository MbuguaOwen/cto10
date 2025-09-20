\
from pathlib import Path
import os, json
from typing import Dict, Any

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def skip_if_exists(path: Path, force: bool) -> bool:
    return path.exists() and not force
