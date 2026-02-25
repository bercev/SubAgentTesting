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


def test_bash_tool_accepts_command_alias(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    result = registry.execute("bash", {"command": "pwd"})

    assert result["returncode"] == 0
    assert str(tmp_path) in result["output"]


def test_bash_tool_prefers_cmd_over_command_alias(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    result = registry.execute("bash", {"cmd": "printf cmd", "command": "printf command"})

    assert result["returncode"] == 0
    assert result["output"] == "cmd"


def test_workspace_list_missing_path_includes_context(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    result = registry.workspace_list("missing-dir")

    assert result["error"] == "path not found"
    assert result["path"] == "missing-dir"
    assert result["workspace_root"] == str(tmp_path.resolve())


def test_workspace_open_missing_file_includes_context(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    result = registry.workspace_open("missing.txt")

    assert result["error"] == "file not found"
    assert result["path"] == "missing.txt"
    assert result["workspace_root"] == str(tmp_path.resolve())


def test_workspace_search_skips_default_excluded_directories(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "src" / "good.py").write_text("TOKEN_MATCH = 1\n", encoding="utf-8")
    (tmp_path / "artifacts" / "noise.txt").write_text("TOKEN_MATCH = 2\n", encoding="utf-8")
    (tmp_path / ".venv" / "noise.py").write_text("TOKEN_MATCH = 3\n", encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_search("TOKEN_MATCH")

    files = [row["file"] for row in result["matches"]]
    assert "src/good.py" in files
    assert "artifacts/noise.txt" not in files
    assert ".venv/noise.py" not in files
