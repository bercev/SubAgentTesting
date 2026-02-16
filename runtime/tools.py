import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolContext:
    """Execution context shared by all tool handlers."""
    workspace_root: Path
    submit_callback: Optional[callable] = None
    bash_timeout_s: int = 60
    output_truncate: int = 4000


class ToolRegistry:
    """Collection of workspace and submission tools callable by the agent."""

    def __init__(self, ctx: ToolContext) -> None:
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

    # ============================================================
    # SAFE EXECUTION WRAPPER (Prevents Runtime Crashes)
    # ============================================================

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool safely without allowing exceptions to crash runtime."""

        if name not in self._tools:
            return {"error": f"unknown tool '{name}'"}

        try:
            arguments = arguments or {}
            return self._tools[name](**arguments)

        except TypeError as e:
            return {
                "error": f"invalid arguments for tool '{name}'",
                "details": str(e),
                "received_arguments": arguments,
            }

        except Exception as e:
            return {
                "error": f"tool '{name}' execution failed",
                "details": str(e),
            }

    # ============================================================
    # TOOL SCHEMAS (OpenAI-Compatible)
    # ============================================================

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "workspace_list",
                    "description": "List files under a workspace-relative path",
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
                    "description": "Open a file and return selected lines",
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
                    "description": "Search for a regex pattern in files",
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
                    "description": "Write full content to a file",
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
                    "description": "Execute a shell command inside workspace",
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
                    "description": "Submit final artifact",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "final_artifact": {"type": "string"}
                        },
                        "required": ["final_artifact"],
                    },
                },
            },
        ]

    # ============================================================
    # TOOL IMPLEMENTATIONS
    # ============================================================

    def _safe_resolve(self, path: str) -> Optional[Path]:
        root = self.ctx.workspace_root
        target = (root / path).resolve()
        if root not in target.parents and target != root:
            return None
        return target

    def workspace_list(self, path: str, **kwargs) -> Dict[str, Any]:
        target = self._safe_resolve(path)
        if target is None:
            return {"error": "path escapes workspace"}
        if not target.exists():
            return {"error": "path not found"}

        entries = [
            {"name": p.name, "type": "dir" if p.is_dir() else "file"}
            for p in sorted(target.iterdir())
        ]
        return {"entries": entries}

    def workspace_open(
        self,
        path: str,
        start_line: Optional[int] = 1,
        end_line: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        target = self._safe_resolve(path)
        if target is None:
            return {"error": "path escapes workspace"}
        if not target.is_file():
            return {"error": "file not found"}

        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

        start = max(1, start_line or 1)
        end = end_line if end_line is not None else len(lines)

        snippet = "".join(lines[start - 1 : end])
        return {
            "content": snippet,
            "start_line": start,
            "end_line": end,
        }

    def workspace_search(self, query: str, glob: str = "**/*", **kwargs) -> Dict[str, Any]:
        root = self.ctx.workspace_root
        pattern = re.compile(query)
        matches = []

        for path in root.glob(glob):
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                for m in pattern.finditer(content):
                    line_no = content[: m.start()].count("\n") + 1
                    matches.append({
                        "file": str(path.relative_to(root)),
                        "line": line_no,
                        "match": m.group(0),
                    })
                    if len(matches) >= 50:
                        return {"matches": matches, "truncated": True}

        return {"matches": matches}

    def workspace_apply_patch(self, unified_diff: str, **kwargs) -> Dict[str, Any]:
        root = self.ctx.workspace_root

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

            output = (proc.stdout + proc.stderr)[: self.ctx.output_truncate]
            return {"success": proc.returncode == 0, "output": output}

        finally:
            os.unlink(tmp_path)

    def workspace_write(self, path: str, content: str, **kwargs) -> Dict[str, Any]:
        target = self._safe_resolve(path)
        if target is None:
            return {"error": "path escapes workspace"}

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"written": str(target.relative_to(self.ctx.workspace_root))}

    def bash(self, cmd: str, timeout_s: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        timeout = timeout_s or self.ctx.bash_timeout_s

        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=self.ctx.workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = (proc.stdout + proc.stderr)[: self.ctx.output_truncate]
        return {"returncode": proc.returncode, "output": output}

    def submit(self, final_artifact: str, **kwargs) -> Dict[str, Any]:
        if self.ctx.submit_callback:
            self.ctx.submit_callback(final_artifact)

        return {
            "submitted": True,
            "artifact_preview": final_artifact[:2000],
        }
