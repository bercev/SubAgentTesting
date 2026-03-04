from __future__ import annotations

from typing import Optional

ARCHITECTURE_NONE = "none"
ARCHITECTURE_MINI_SWE_AGENT = "mini-swe-agent"

VALID_AGENT_ARCHITECTURES = {
    ARCHITECTURE_NONE,
    ARCHITECTURE_MINI_SWE_AGENT,
}


def normalize_architecture_id(value: Optional[str], *, default: str = ARCHITECTURE_NONE) -> str:
    """Normalize and validate one architecture identifier."""

    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("agent architecture id must be a string")

    normalized = value.strip() or default
    if normalized not in VALID_AGENT_ARCHITECTURES:
        supported = ", ".join(sorted(VALID_AGENT_ARCHITECTURES))
        raise ValueError(
            f"Unsupported agent architecture '{normalized}'. Supported values: {supported}."
        )
    return normalized
