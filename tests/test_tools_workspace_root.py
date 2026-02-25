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
    assert result.get("tool_name") == "workspace_open"
    assert result.get("provided_keys") == ["raw"]
    assert result.get("expected_keys") == ["end_line", "path", "start_line"]


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


def test_workspace_open_without_end_line_is_capped_and_paginates(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    content = "".join(f"line {i}\n" for i in range(1, 501))
    (tmp_path / "big.txt").write_text(content, encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_open("big.txt")

    assert result["start_line"] == 1
    assert result["end_line"] == 250
    assert result["total_lines"] == 500
    assert result["truncated"] is True
    assert result["clamped"] is False
    assert result["next_start_line"] == 251
    assert result["content"].startswith("line 1\n")
    assert "line 250\n" in result["content"]
    assert "line 251\n" not in result["content"]


def test_workspace_open_clamps_oversized_explicit_range(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    content = "".join(f"row {i}\n" for i in range(1, 1001))
    (tmp_path / "big.txt").write_text(content, encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_open("big.txt", start_line=10, end_line=999)

    assert result["start_line"] == 10
    assert result["end_line"] == 409
    assert result["total_lines"] == 1000
    assert result["clamped"] is True
    assert result["truncated"] is True
    assert result["next_start_line"] == 410
    assert result["content"].startswith("row 10\n")
    assert "row 409\n" in result["content"]
    assert "row 410\n" not in result["content"]
