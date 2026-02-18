from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def new_run_id() -> str:
    """Create a timestamped run identifier used in artifact paths."""

    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def now_iso() -> str:
    """Return current local timestamp in stable ISO format."""

    return datetime.now().isoformat(timespec="seconds")


def now_human() -> str:
    """Return local timestamp in a compact log-friendly format."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def _infer_log_source() -> str:
    """Best-effort caller source in file:line format."""

    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is None:
            return "unknown:0"
        return f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}"
    finally:
        del frame


def append_log(
    path: Path,
    message: str,
    *,
    level: str = "INFO",
    source: Optional[str] = None,
) -> None:
    """Append one formatted line to the run log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_level = (level or "INFO").upper()
    normalized_source = source or _infer_log_source()
    line = f"{now_human()} | {normalized_level:<8} | {normalized_source:<24} | {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
