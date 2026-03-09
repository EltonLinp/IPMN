from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_CONFIG_CACHE: Dict[str, Any] | None = None


def _load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    path = Path(__file__).resolve().parent / "runtime_config.json"
    if not path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        _CONFIG_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_setting(key: str, default: Any = None) -> Any:
    cfg = _load_config()
    return cfg.get(key, default)
