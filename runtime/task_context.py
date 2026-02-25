from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TaskWorkspaceContext:
    """Structured adapter-provided workspace/tool readiness context for one task."""

    workspace_root: Path
    workspace_exists: bool
    tools_ready: bool
    workspace_kind: str
    reason: Optional[str] = None
    repo: Optional[str] = None
    dataset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
