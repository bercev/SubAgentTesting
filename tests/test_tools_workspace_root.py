from __future__ import annotations

from pathlib import Path

from runtime.tools import ToolContext, ToolRegistry


def test_workspace_tools_allow_relative_root_with_dot_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("hello\nworld\n", encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    listed = registry.workspace_list(".")
    opened = registry.workspace_open("file.txt")

    assert "entries" in listed
    assert listed.get("error") is None
    assert opened["content"] == "hello\nworld\n"


def test_workspace_tools_block_paths_outside_workspace(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    outside_dir = tmp_path.parent

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_list(str(outside_dir))

    assert result == {"error": "path escapes workspace"}


def test_execute_returns_error_for_invalid_tool_argument_shape(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("hello\n", encoding="utf-8")
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    result = registry.execute("workspace_open", {"raw": "{\"path\": \"file.txt\"}"})

    assert "error" in result
    assert "invalid arguments for workspace_open" in result["error"]
    assert result.get("provided_keys") == ["raw"]
