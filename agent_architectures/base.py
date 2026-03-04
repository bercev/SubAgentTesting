from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Protocol, Set

from runtime.schemas import AgentResult, BenchmarkTask
from runtime.tools import ToolRegistry


@dataclass
class ArchitectureRunRequest:
    """Task-scoped payload passed to one architecture runtime implementation."""

    task: BenchmarkTask
    system_prompt: str
    initial_user_message: str
    mode_name: Literal["patch_only", "tools_enabled"]
    backend_config: Mapping[str, Any]
    decoding_defaults: Mapping[str, Any]
    tool_registry: ToolRegistry
    allowed_tools: Set[str]
    max_tool_calls: int
    max_wall_time_s: int
    termination_tool: str
    full_log_previews: bool
    api_log: Optional[Callable[[str], None]] = None
    architecture_config: Optional[Mapping[str, Any]] = None


class AgentArchitecture(Protocol):
    """Protocol for pluggable agent architectures used by run_service."""

    architecture_id: str

    def run_task(self, request: ArchitectureRunRequest) -> AgentResult:
        """Execute one task and return a terminal artifact + metadata payload."""
        raise NotImplementedError


def filter_tool_schemas(registry: ToolRegistry, allowed: Set[str]) -> list[dict[str, Any]]:
    """Keep only tool schemas explicitly allowed by the active policy."""

    schemas: list[dict[str, Any]] = []
    for schema in registry.schemas:
        name = schema.get("function", {}).get("name")
        if name in allowed:
            schemas.append(schema)
    return schemas
