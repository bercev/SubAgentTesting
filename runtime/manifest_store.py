from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def manifest_path(run_root: Path) -> Path:
    return run_root / "manifest.json"


def read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return repr(value)


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
