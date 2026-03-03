from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_path(base: Path, namespace: str, key: str, suffix: str = "json") -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    path = base / namespace / f"{digest}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_cache(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
