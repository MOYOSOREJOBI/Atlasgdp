from __future__ import annotations

import json
import secrets
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas_gdp.config import Settings


def utc_now() -> datetime:
    return datetime.now(UTC)


def make_run_id(prefix: str = "run") -> str:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
    return f"{stamp}_{suffix}"


def ensure_run_dirs(settings: Settings, run_id: str) -> tuple[Path, Path]:
    runs_root = settings.paths.artifacts / "runs"
    run_dir = runs_root / run_id
    latest_dir = settings.paths.artifacts / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, latest_dir


def clear_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_tree_contents(src: Path, dst: Path) -> None:
    clear_directory(dst)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sqlite_path_from_db_url(db_url: str) -> Path:
    if db_url.startswith("sqlite:///"):
        return Path(db_url.removeprefix("sqlite:///"))
    if db_url.startswith("sqlite://"):
        return Path(db_url.removeprefix("sqlite://"))
    return Path(db_url)


def _open_db(settings: Settings) -> sqlite3.Connection:
    db_path = _sqlite_path_from_db_url(settings.db_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    existing = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
    if existing:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        required = {"run_id", "created_at", "as_of", "horizon_q", "scenario_json", "manifest_path"}
        if not required.issubset(columns):
            conn.execute("DROP TABLE IF EXISTS runs")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            as_of TEXT NOT NULL,
            horizon_q INTEGER NOT NULL,
            scenario_json TEXT NOT NULL,
            manifest_path TEXT NOT NULL
        )
        """
    )
    return conn


def record_run(
    settings: Settings,
    manifest: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
    manifest_path: str | None = None,
    created_at: str | None = None,
    as_of: str | None = None,
    horizon_q: int | None = None,
    scenario: dict[str, Any] | None = None,
) -> None:
    if manifest is not None:
        run_id = str(manifest["run_id"])
        manifest_path = str(manifest["manifest_path"])
        created_at = str(manifest.get("created_at_utc") or manifest.get("created_at"))
        as_of = str(manifest["as_of"])
        horizons = list(manifest.get("horizons", []))
        horizon_q = int(max(horizons) if horizons else manifest.get("horizon_q", 0))
        scenario = dict(manifest.get("scenario", {}))
    if run_id is None or manifest_path is None or created_at is None or as_of is None or horizon_q is None:
        raise ValueError("record_run requires run_id, manifest_path, created_at, as_of, and horizon_q")

    conn = _open_db(settings)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id,
                created_at,
                as_of,
                horizon_q,
                scenario_json,
                manifest_path
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                as_of,
                int(horizon_q),
                json.dumps(scenario or {}),
                manifest_path,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def latest_manifest_pointer(settings: Settings) -> Path:
    return settings.paths.artifacts / "latest_manifest.json"


def latest_run_id_pointer(settings: Settings) -> Path:
    return settings.paths.artifacts / "latest_run_id.txt"


def write_latest_pointer(settings: Settings, manifest: dict[str, Any]) -> None:
    pointer = {
        "run_id": manifest["run_id"],
        "created_at_utc": manifest["created_at_utc"],
        "manifest_path": manifest["manifest_path"],
    }
    write_json(latest_manifest_pointer(settings), pointer)
    latest_run_id_pointer(settings).write_text(f"{manifest['run_id']}\n", encoding="utf-8")


def read_latest_pointer(settings: Settings) -> dict[str, Any] | None:
    path = latest_manifest_pointer(settings)
    if not path.exists():
        return None
    return read_json(path)


def read_latest_run_id(settings: Settings) -> str | None:
    path = latest_run_id_pointer(settings)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None
