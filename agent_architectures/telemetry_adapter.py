from __future__ import annotations

import json
from typing import Any, Dict, Mapping

MAX_TOOL_MESSAGE_CHARS = 12000


def json_size_bytes(payload: Any) -> int:
    """Approximate payload size as UTF-8 JSON bytes."""

    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(payload)
    return len(serialized.encode("utf-8", errors="ignore"))


def truncate_text(text: str, limit: int) -> str:
    """Truncate large payloads before appending them to model context."""

    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def serialize_tool_message(payload: Any) -> str:
    """Serialize tool payload using the legacy runtime truncation policy."""

    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(payload)
    return truncate_text(serialized, MAX_TOOL_MESSAGE_CHARS)


def build_runtime_payload(
    *,
    mode_name: str,
    loop_exit_reason: str,
    budget_exhausted: bool,
    wall_time_exhausted: bool,
    termination_ack: bool,
    events: list[dict[str, Any]],
) -> Dict[str, Any]:
    """Build tool-quality runtime payload matching the legacy schema."""

    return {
        "mode": mode_name,
        "loop_exit_reason": loop_exit_reason,
        "budget_exhausted": budget_exhausted,
        "wall_time_exhausted": wall_time_exhausted,
        "termination_ack": termination_ack,
        "events": list(events),
    }


def pick_repo_metadata(resources: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return optional metadata fields expected by downstream adapters."""

    if not isinstance(resources, Mapping):
        return {}
    repo = resources.get("repo")
    if isinstance(repo, str) and repo:
        return {"repo": repo}
    return {}
