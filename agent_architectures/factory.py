from __future__ import annotations

from typing import Optional

from agent_architectures.base import AgentArchitecture
from agent_architectures.constants import (
    ARCHITECTURE_MINI_SWE_AGENT,
    ARCHITECTURE_NONE,
    normalize_architecture_id,
)
from agent_architectures.legacy import LegacyArchitecture
from agent_architectures.mini_swe_agent import MiniSweAgentArchitecture


def resolve_agent_architecture(
    *,
    cli_override: Optional[str],
    run_override: Optional[str],
    profile_architecture: Optional[str],
) -> str:
    """Resolve architecture by precedence: CLI > run-config > profile > default."""

    for candidate in (cli_override, run_override, profile_architecture):
        if candidate is None:
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        return normalize_architecture_id(candidate)
    return ARCHITECTURE_NONE


def get_agent_architecture(architecture_id: str) -> AgentArchitecture:
    """Construct an architecture implementation from one normalized id."""

    normalized = normalize_architecture_id(architecture_id)
    if normalized == ARCHITECTURE_NONE:
        return LegacyArchitecture()
    if normalized == ARCHITECTURE_MINI_SWE_AGENT:
        return MiniSweAgentArchitecture()
    raise ValueError(f"Unsupported agent architecture: {normalized}")
