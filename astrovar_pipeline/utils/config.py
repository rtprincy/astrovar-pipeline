
from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Expand env vars like ${GAIA_USER}
    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k,v in obj.items()}
        if isinstance(obj, list):
            return [_expand(x) for x in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj
    return _expand(cfg)
