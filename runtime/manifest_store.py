from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def new_run_id() -> str:
    """Create a timestamped run identifier used in artifact paths."""

    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def now_iso() -> str:
    """Return current local timestamp in stable ISO format."""

    return datetime.now().isoformat(timespec="seconds")


def manifest_path(run_root: Path) -> Path:
    """Return canonical manifest location for a run root."""

    return run_root / "manifest.json"


def read_manifest(path: Path) -> Dict[str, Any]:
    """Read manifest JSON; return empty object when missing or invalid."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Write manifest JSON and fail fast on non-serializable values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    """Append one timestamped line to the run log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
