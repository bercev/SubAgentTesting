from __future__ import annotations

from runtime.schemas import BenchmarkTask
from runtime.task_context import TaskWorkspaceContext


def build_initial_user_message(
    task: BenchmarkTask,
    workspace: TaskWorkspaceContext,
    mode_name: str,
) -> str:
    """Build the first user message, appending workspace context for tools runs."""

    base = task.instruction or ""
    if mode_name != "tools_enabled":
        return base

    lines = [
        "<workspace_context>",
        f"repo: {workspace.repo or ''}",
        f"workspace_root: {workspace.workspace_root.resolve()}",
        f"workspace_kind: {workspace.workspace_kind}",
        f"workspace_exists: {str(workspace.workspace_exists).lower()}",
        f"tools_ready: {str(workspace.tools_ready).lower()}",
    ]
    if workspace.reason:
        lines.append(f"reason: {workspace.reason}")
    lines.append("</workspace_context>")
    suffix = "\n".join(lines)
    if not base:
        return suffix
    return f"{base}\n\n{suffix}"
