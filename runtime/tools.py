import inspect
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from runtime.artifact_policy import apply_artifact_policy

DEFAULT_OPEN_MAX_LINES = 250
MAX_OPEN_RANGE_LINES = 400


@dataclass
class ToolContext:
    """Execution context shared by all tool handlers."""

    workspace_root: Path
    submit_callback: Optional[Callable[[str], None]] = None
    expected_output_type: str = "patch"
    patch_submit_policy: str = "allow"
    max_invalid_submit_attempts: int = 3
    invalid_submit_attempts: int = 0
    last_invalid_submit_reason: Optional[str] = None
    bash_timeout_s: int = 60
    output_truncate: int = 4000
    search_exclude_dirnames: Tuple[str, ...] = (
        ".git",
        ".venv",
        "artifacts",
        "logs",
        "summaries",
        "__pycache__",
        ".pytest_cache",
        "portable_agent_runner.egg-info",
    )


class ToolRegistry:
    """Collection of workspace and submission tools callable by the agent."""

    def __init__(self, ctx: ToolContext) -> None:
        """Register available tool handlers for the current task context."""

        self.ctx = ctx
        self._tools = {
            "workspace_list": self.workspace_list,
            "workspace_open": self.workspace_open,
            "workspace_search": self.workspace_search,
            "workspace_apply_patch": self.workspace_apply_patch,
            "workspace_write": self.workspace_write,
            "bash": self.bash,
            "submit": self.submit,
        }

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function tool schemas."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "workspace_list",
                    "description": "List files under a path relative to the workspace root",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_open",
                    "description": (
                        "Open a file and return selected lines. Prefer line-bounded reads. "
                        "If end_line is omitted, returns a capped page (default 250 lines) "
                        "starting at start_line. Page through large files with start_line/end_line."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_search",
                    "description": (
                        "Search for a regex pattern in files. Supports only query and optional glob "
                        "(for example glob='**/*.py' or '**/tests/*.py'). "
                        "Do not pass start_line/end_line to this tool."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "glob": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_apply_patch",
                    "description": "Apply a unified diff patch relative to workspace root",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unified_diff": {"type": "string"},
                        },
                        "required": ["unified_diff"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_write",
                    "description": "Write full content to a file (overwrite)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a bash command inside the workspace root",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "timeout_s": {"type": "integer"},
                        },
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit",
                    "description": (
                        "Submit final artifact and terminate the task. For patch tasks, final_artifact "
                        "must be either one raw unified diff starting with 'diff --git' or an empty string."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"final_artifact": {"type": "string"}},
                        "required": ["final_artifact"],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one tool call by name with dict arguments."""

        if name not in self._tools:
            return {"error": f"unknown tool {name}"}
        if not isinstance(arguments, dict):
            return {"error": f"invalid arguments for {name}: expected object payload"}
        normalized_arguments = dict(arguments)
        if name == "bash" and "command" in normalized_arguments:
            if "cmd" not in normalized_arguments and isinstance(normalized_arguments.get("command"), str):
                normalized_arguments["cmd"] = normalized_arguments["command"]
            normalized_arguments.pop("command", None)
        try:
            return self._tools[name](**normalized_arguments)
        except TypeError as exc:
            expected_keys = sorted(inspect.signature(self._tools[name]).parameters.keys())
            return {
                "error": f"invalid arguments for {name}: {exc}",
                "tool_name": name,
                "provided_keys": sorted(arguments.keys()),
                "expected_keys": expected_keys,
            }

    def _workspace_root(self) -> Path:
        """Return normalized absolute workspace root path."""

        return self.ctx.workspace_root.resolve()

    def _resolve_target(self, path: str) -> tuple[Path, Path]:
        """Resolve one workspace-relative target path and return (root, target)."""

        root = self._workspace_root()
        target = (root / path).resolve()
        return root, target

    # Tool implementations

    def workspace_list(self, path: str) -> Dict[str, Any]:
        """List files/directories under a workspace-relative path."""

        root, target = self._resolve_target(path)
        if root not in target.parents and target != root:
            return {"error": "path escapes workspace"}
        if not target.exists():
            return {"error": "path not found", "path": path, "workspace_root": str(root)}
        entries = []
        for entry in sorted(target.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
            })
        return {"entries": entries}

    def workspace_open(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> Dict[str, Any]:
        """Read a line range from a workspace file with escape checks."""

        root, target = self._resolve_target(path)
        if root not in target.parents and target != root:
            return {"error": "path escapes workspace"}
        if not target.is_file():
            return {"error": "file not found", "path": path, "workspace_root": str(root)}
        with target.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total_lines = len(lines)
        start = max(1, int(start_line))
        clamped = False

        if end_line is None:
            requested_end = start + DEFAULT_OPEN_MAX_LINES - 1
        else:
            requested_end = max(start, int(end_line))
            max_end = start + MAX_OPEN_RANGE_LINES - 1
            if requested_end > max_end:
                requested_end = max_end
                clamped = True

        end = min(requested_end, total_lines) if total_lines > 0 else 0
        snippet = "".join(lines[start - 1 : end]) if total_lines > 0 else ""
        truncated = end < total_lines
        next_start_line = (end + 1) if truncated else None
        return {
            "content": snippet,
            "start_line": start,
            "end_line": end,
            "total_lines": total_lines,
            "truncated": truncated,
            "clamped": clamped,
            "next_start_line": next_start_line,
        }

    def workspace_search(self, query: str, glob: str = "**/*") -> Dict[str, Any]:
        """Regex-search files under workspace and return capped matches."""

        root = self._workspace_root()
        pattern = re.compile(query)
        matches = []
        excluded = set(self.ctx.search_exclude_dirnames)
        for path in root.glob(glob):
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if any(part in excluded for part in rel_parts):
                continue
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for m in pattern.finditer(content):
                    line_no = content[: m.start()].count("\n") + 1
                    matches.append({"file": str(path.relative_to(root)), "line": line_no, "match": m.group(0)})
                    if len(matches) >= 50:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches}

    def workspace_apply_patch(self, unified_diff: str) -> Dict[str, Any]:
        """Run `patch` in workspace root and return success + truncated output."""

        root = self._workspace_root()
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(unified_diff)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["patch", "-p0", "-d", str(root)],
                stdin=open(tmp_path, "r"),
                capture_output=True,
                text=True,
                timeout=self.ctx.bash_timeout_s,
            )
            success = proc.returncode == 0
            output = (proc.stdout + proc.stderr)[: self.ctx.output_truncate]
            return {"success": success, "output": output}
        finally:
            os.unlink(tmp_path)

    def workspace_write(self, path: str, content: str) -> Dict[str, Any]:
        """Overwrite a workspace file after path safety checks."""

        root, target = self._resolve_target(path)
        if root not in target.parents and target != root:
            return {"error": "path escapes workspace"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"written": str(target.relative_to(root))}

    def bash(self, cmd: str, timeout_s: Optional[int] = None) -> Dict[str, Any]:
        """Execute one shell command in workspace with timeout and truncation."""

        root = self._workspace_root()
        timeout = timeout_s or self.ctx.bash_timeout_s
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr)[: self.ctx.output_truncate]
        return {"returncode": proc.returncode, "output": output}

    def submit(self, final_artifact: str) -> Dict[str, Any]:
        """Signal task completion and forward final artifact to runtime callback."""

        artifact = final_artifact if isinstance(final_artifact, str) else str(final_artifact)
        output_type = (self.ctx.expected_output_type or "text").strip().lower()
        submit_policy = (self.ctx.patch_submit_policy or "allow").strip().lower()

        if output_type == "patch":
            policy_result = apply_artifact_policy(artifact, "patch")
            if policy_result.valid:
                normalized = policy_result.artifact
                if self.ctx.submit_callback:
                    self.ctx.submit_callback(normalized)
                return {
                    "submitted": True,
                    "artifact_preview": normalized[:2000],
                    "artifact_bytes": len(normalized.encode("utf-8", errors="ignore")),
                }

            self.ctx.last_invalid_submit_reason = policy_result.reason

            if submit_policy == "allow":
                if self.ctx.submit_callback:
                    self.ctx.submit_callback(artifact)
                return {
                    "submitted": True,
                    "artifact_preview": artifact[:2000],
                    "artifact_bytes": len(artifact.encode("utf-8", errors="ignore")),
                    "submission_warning": f"invalid_patch:{policy_result.reason}",
                    "invalid_submission_reason": policy_result.reason,
                }

            self.ctx.invalid_submit_attempts += 1
            max_attempts = max(1, int(self.ctx.max_invalid_submit_attempts))
            exhausted = self.ctx.invalid_submit_attempts >= max_attempts
            terminal_reason: Optional[str] = None
            if submit_policy == "reject_fast":
                terminal_reason = "invalid_submission_fast_fail"
            elif submit_policy == "reject_retry" and exhausted:
                terminal_reason = "invalid_submission_retries_exhausted"

            result: Dict[str, Any] = {
                "error": (
                    "invalid patch submission "
                    f"(reason={policy_result.reason}); submit a unified diff starting with 'diff --git'"
                ),
                "invalid_submission": True,
                "invalid_submission_reason": policy_result.reason,
                "invalid_submit_attempts": self.ctx.invalid_submit_attempts,
                "max_invalid_submit_attempts": max_attempts,
                "retryable": terminal_reason is None,
            }
            if terminal_reason is not None:
                result["invalid_submission_terminal_reason"] = terminal_reason
            return result

        if self.ctx.submit_callback:
            self.ctx.submit_callback(artifact)
        return {
            "submitted": True,
            "artifact_preview": artifact[:2000],
            "artifact_bytes": len(artifact.encode("utf-8", errors="ignore")),
        }
