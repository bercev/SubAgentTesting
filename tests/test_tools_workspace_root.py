from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    (tmp_path / "venv" / "lib").mkdir(parents=True)
    (tmp_path / "build").mkdir()
    (tmp_path / "site-packages").mkdir()
    (tmp_path / "src" / "good.py").write_text("TOKEN_MATCH = 1\n", encoding="utf-8")
    (tmp_path / "artifacts" / "noise.txt").write_text("TOKEN_MATCH = 2\n", encoding="utf-8")
    (tmp_path / ".venv" / "noise.py").write_text("TOKEN_MATCH = 3\n", encoding="utf-8")
    (tmp_path / "venv" / "lib" / "noise.py").write_text("TOKEN_MATCH = 4\n", encoding="utf-8")
    (tmp_path / "build" / "noise.py").write_text("TOKEN_MATCH = 5\n", encoding="utf-8")
    (tmp_path / "site-packages" / "noise.py").write_text("TOKEN_MATCH = 6\n", encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_search("TOKEN_MATCH")

    files = [row["file"] for row in result["matches"]]
    assert "src/good.py" in files
    assert "artifacts/noise.txt" not in files
    assert ".venv/noise.py" not in files
    assert "venv/lib/noise.py" not in files
    assert "build/noise.py" not in files
    assert "site-packages/noise.py" not in files


def test_workspace_list_rejects_blocked_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build").mkdir()

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_list("build")

    assert result["error"] == "path not allowed"
    assert result["path"] == "build"
    assert result["workspace_root"] == str(tmp_path.resolve())


def test_workspace_list_filters_blocked_children(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "src" / "ok.txt").write_text("ok\n", encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_list(".")

    names = [row["name"] for row in result["entries"]]
    assert "src" in names
    assert "build" not in names
    assert ".venv" not in names


def test_workspace_open_rejects_blocked_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "noise.txt").write_text("x\n", encoding="utf-8")

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_open("build/noise.txt")

    assert result["error"] == "path not allowed"
    assert result["path"] == "build/noise.txt"
    assert result["workspace_root"] == str(tmp_path.resolve())


def test_workspace_write_rejects_blocked_path(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dist").mkdir()

    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    result = registry.workspace_write("dist/out.txt", "content")

    assert result["error"] == "path not allowed"
    assert result["path"] == "dist/out.txt"
    assert result["workspace_root"] == str(tmp_path.resolve())
    assert not (tmp_path / "dist" / "out.txt").exists()


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


def test_workspace_apply_patch_uses_p1_when_it_succeeds(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    seen_cmds: list[list[str]] = []

    def _fake_run(cmd, capture_output, text, timeout):  # noqa: ANN001, ANN201
        del capture_output, text, timeout
        seen_cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="applied", stderr="")

    monkeypatch.setattr("runtime.tools.subprocess.run", _fake_run)

    result = registry.workspace_apply_patch("diff --git a/a b/a\n")

    assert result["success"] is True
    assert result["strip_level_used"] == 1
    assert len(seen_cmds) == 1
    assert "--batch" in seen_cmds[0]
    assert "--forward" in seen_cmds[0]
    assert "-p1" in seen_cmds[0]


def test_workspace_apply_patch_falls_back_to_p0_after_p1_failure(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))
    seen_cmds: list[list[str]] = []

    def _fake_run(cmd, capture_output, text, timeout):  # noqa: ANN001, ANN201
        del capture_output, text, timeout
        seen_cmds.append(list(cmd))
        if "-p1" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="p1 failed")
        return SimpleNamespace(returncode=0, stdout="p0 applied", stderr="")

    monkeypatch.setattr("runtime.tools.subprocess.run", _fake_run)

    result = registry.workspace_apply_patch("diff --git a/a b/a\n")

    assert result["success"] is True
    assert result["strip_level_used"] == 0
    assert len(seen_cmds) == 2
    assert "-p1" in seen_cmds[0]
    assert "-p0" in seen_cmds[1]
    assert "p1 failed" in result["output"]
    assert "p0 applied" in result["output"]


def test_workspace_apply_patch_reports_clean_failure_when_all_strip_levels_fail(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    def _fake_run(cmd, capture_output, text, timeout):  # noqa: ANN001, ANN201
        del cmd, capture_output, text, timeout
        return SimpleNamespace(returncode=1, stdout="", stderr="failed")

    monkeypatch.setattr("runtime.tools.subprocess.run", _fake_run)

    result = registry.workspace_apply_patch("diff --git a/a b/a\n")

    assert result["success"] is False
    assert "patch -p1: returncode=1" in result["output"]
    assert "patch -p0: returncode=1" in result["output"]


def test_workspace_apply_patch_rejects_blocked_target_without_subprocess(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry(ToolContext(workspace_root=Path(".")))

    called = {"value": False}

    def _fake_run(cmd, capture_output, text, timeout):  # noqa: ANN001, ANN201
        del cmd, capture_output, text, timeout
        called["value"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("runtime.tools.subprocess.run", _fake_run)

    patch_text = (
        "diff --git a/build/generated.py b/build/generated.py\n"
        "--- a/build/generated.py\n"
        "+++ b/build/generated.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    result = registry.workspace_apply_patch(patch_text)

    assert result["success"] is False
    assert result["error"] == "path not allowed"
    assert result["path"] == "build/generated.py"
    assert called["value"] is False
