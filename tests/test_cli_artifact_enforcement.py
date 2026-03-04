import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import typer

from agent_architectures.constants import ARCHITECTURE_NONE
import scripts.cli as cli
import runtime.eval_service as eval_service
import runtime.run_service as run_service
from agents.spec_loader import AgentSpec
from runtime.config_loader import normalize_run_config
from runtime.schemas import AgentResult, BenchmarkTask
from runtime.task_context import TaskWorkspaceContext


class _FakeAdapter:
    benchmark_name = "swebench_verified"
    _workspace_root = Path(".")
    _workspace_tools_ready = True
    _workspace_kind = "repo_checkout"
    _workspace_reason = None

    def __init__(self, task: BenchmarkTask):
        self._task = task

    @classmethod
    def from_config(cls, config):
        task = BenchmarkTask(
            task_id="astropy__astropy-12907",
            instruction="Fix issue",
            resources={"repo": "astropy/astropy"},
            expected_output_type="patch",
        )
        return cls(task=task)

    def load_tasks(self, split: str, selector: int | None = None):
        return [self._task]

    def workspace_context_for_task(self, task: BenchmarkTask) -> TaskWorkspaceContext:
        repo = task.resources.get("repo") if task.resources else None
        repo_name = repo if isinstance(repo, str) else None
        return TaskWorkspaceContext(
            workspace_root=self._workspace_root,
            workspace_exists=True,
            tools_ready=self._workspace_tools_ready,
            workspace_kind=self._workspace_kind,
            reason=self._workspace_reason,
            repo=repo_name,
            dataset_name="SWE-bench/SWE-bench_Verified",
        )

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: dict | None = None,
    ) -> dict:
        return {
            "instance_id": task.task_id,
            "model_patch": artifact,
            "model_name_or_path": model_name_or_path,
            "model_name": model_name,
            "repo": (metadata or {}).get("repo"),
        }

    def get_evaluator(self, config):
        raise NotImplementedError


class _FakeRegistry:
    def get_adapter(self, name: str):
        assert name == "swebench_verified"
        return _FakeAdapter


_UNSET = object()


def _run_once(
    monkeypatch,
    tmp_path: Path,
    raw_artifact: str,
    *,
    verbose: bool = False,
    mode: str = "patch_only",
    runtime_tool_payload: dict | None = None,
    loader_allowed_tools=_UNSET,
    runtime_init_capture: dict | None = None,
    full_log_previews: bool = False,
    build_backend_capture: dict | None = None,
    agent_architecture: str | None = None,
    runtime_agent_architecture_override: str | None = None,
    profile_agent_architecture: str = ARCHITECTURE_NONE,
    profile_tools=_UNSET,
    patch_submit_policy: str = "allow",
    max_invalid_submit_attempts: int = 3,
    workspace_tools_ready: bool = True,
    workspace_kind: str = "repo_checkout",
    workspace_reason: str | None = None,
    terminated: bool = True,
    mini_turn_trace: list[str] | None = None,
):
    run_config = normalize_run_config(
        {
            "benchmark": {
                "name": "swebench_verified",
                "dataset_name": "SWE-bench/SWE-bench_Verified",
                "split": "test",
                "data_source": "hf",
                "data_root": None,
            },
            "evaluation": {
                "harness_cmd": "python -m swebench.harness.run_evaluation",
                "eval_root": "./external/SWE-bench",
                "workdir": ".",
            },
            "runtime": {
                "mode": mode,
                "selector": 1,
                "max_tool_calls": 1,
                "max_wall_time_s": 10,
                "patch_submit_policy": patch_submit_policy,
                "max_invalid_submit_attempts": max_invalid_submit_attempts,
                "agent_architecture_override": runtime_agent_architecture_override,
            },
            "output": {
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
        }
    )

    spec = AgentSpec(
        name="fake-agent",
        backend={"type": "openrouter", "model": "fake/model"},
        prompt_template="Prompt",
        tools=[] if profile_tools is _UNSET else profile_tools,
        skills=[],
        tool_to_skill_map={},
        termination={"tool": "submit", "output_type": "patch"},
        decoding_defaults={},
        agent_architecture=profile_agent_architecture,
        agent_architecture_config={},
    )

    _FakeAdapter._workspace_root = Path(".")
    _FakeAdapter._workspace_tools_ready = workspace_tools_ready
    _FakeAdapter._workspace_kind = workspace_kind
    _FakeAdapter._workspace_reason = workspace_reason

    class _FakeArchitecture:
        def run_task(self, request):
            if runtime_init_capture is not None:
                runtime_init_capture["request"] = request
            if build_backend_capture is not None:
                build_backend_capture["request"] = request
            metadata = {"terminated": terminated, "repo": "astropy/astropy"}
            if runtime_tool_payload is not None:
                metadata["tool_quality_runtime"] = runtime_tool_payload
            if mini_turn_trace is not None:
                metadata["mini_turn_trace"] = list(mini_turn_trace)
            return AgentResult(
                task_id=request.task.task_id,
                final_artifact=raw_artifact,
                metadata=metadata,
            )

    monkeypatch.setattr(run_service, "BenchmarkRegistry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        run_service.AgentSpecLoader,
        "load",
        lambda *_args, **_kwargs: (
            spec,
            "Prompt",
            set() if loader_allowed_tools is _UNSET else loader_allowed_tools,
        ),
    )
    def _fake_get_agent_architecture(name: str):
        if runtime_init_capture is not None:
            runtime_init_capture["resolved_architecture"] = name
        if build_backend_capture is not None:
            build_backend_capture["resolved_architecture"] = name
        return _FakeArchitecture()

    monkeypatch.setattr(run_service, "get_agent_architecture", _fake_get_agent_architecture)

    outcome = run_service.execute_run(
        agent_path="profiles/agents/qwen3_coder_free.yaml",
        config=run_config,
        agent_architecture=agent_architecture,
        verbose=verbose,
        full_log_previews=full_log_previews,
    )

    records = [
        json.loads(line)
        for line in outcome.predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 1
    return records[0], outcome


def test_run_service_writes_mini_trace_file_for_mini_architecture(monkeypatch, tmp_path: Path):
    _, outcome = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="CANNOT PRODUCE OUTPUT {reason}",
        mode="tools_enabled",
        profile_agent_architecture="mini-swe-agent",
        mini_turn_trace=["First model turn", "Second model turn"],
    )

    trace_path = outcome.predictions_path.parent / "mini_swe_agent_trace.txt"
    assert trace_path.exists()
    assert trace_path.read_text(encoding="utf-8") == (
        "Task: astropy__astropy-12907\n"
        "Turn 1:\n"
        "First model turn\n\n"
        "Turn 2:\n"
        "Second model turn\n\n"
    )


def test_run_service_does_not_write_mini_trace_for_non_mini_architecture(monkeypatch, tmp_path: Path):
    _, outcome = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="CANNOT PRODUCE OUTPUT {reason}",
        profile_agent_architecture="none",
        mini_turn_trace=["Trace should be ignored outside mini architecture"],
    )

    trace_path = outcome.predictions_path.parent / "mini_swe_agent_trace.txt"
    assert not trace_path.exists()


def test_run_service_tools_enabled_empty_allowlist_still_includes_submit(monkeypatch, tmp_path: Path):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools=set(),
        runtime_init_capture=captured,
    )

    assert captured["request"].allowed_tools == {"submit"}


def test_run_service_tools_enabled_uses_full_fallback_when_allowlist_is_none(monkeypatch, tmp_path: Path):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools=None,
        runtime_init_capture=captured,
    )

    allowed_tools = captured["request"].allowed_tools
    assert isinstance(allowed_tools, set)
    assert "submit" in allowed_tools
    assert "bash" in allowed_tools


def test_run_service_mini_tools_enabled_defaults_to_full_registry_when_profile_tools_missing(
    monkeypatch, tmp_path: Path
):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools={"submit"},
        profile_agent_architecture="mini-swe-agent",
        profile_tools=None,
        runtime_init_capture=captured,
    )

    allowed_tools = captured["request"].allowed_tools
    assert isinstance(allowed_tools, set)
    assert "submit" in allowed_tools
    assert "bash" in allowed_tools
    assert "workspace_open" in allowed_tools


def test_run_service_mini_tools_enabled_profile_alias_uses_full_registry(
    monkeypatch, tmp_path: Path
):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools={"submit"},
        profile_agent_architecture="mini-swe-agent",
        profile_tools=["mini-swe-agent"],
        runtime_init_capture=captured,
    )

    allowed_tools = captured["request"].allowed_tools
    assert isinstance(allowed_tools, set)
    assert "submit" in allowed_tools
    assert "bash" in allowed_tools
    assert "workspace_open" in allowed_tools
    assert "mini-swe-agent" not in allowed_tools


def test_run_service_mini_tools_enabled_profile_alias_overrides_explicit_subset(
    monkeypatch, tmp_path: Path
):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools={"submit"},
        profile_agent_architecture="mini-swe-agent",
        profile_tools=["mini-swe-agent", "submit"],
        runtime_init_capture=captured,
    )

    allowed_tools = captured["request"].allowed_tools
    assert "submit" in allowed_tools
    assert "bash" in allowed_tools
    assert "workspace_apply_patch" in allowed_tools


def test_run_service_mini_tools_enabled_prefers_explicit_profile_tools_over_skill_allowlist(
    monkeypatch, tmp_path: Path
):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools={"submit"},
        profile_agent_architecture="mini-swe-agent",
        profile_tools=["submit", "workspace_open"],
        runtime_init_capture=captured,
    )

    assert captured["request"].allowed_tools == {"submit", "workspace_open"}


def test_run_service_mini_tools_enabled_preserves_explicit_empty_profile_tools(
    monkeypatch, tmp_path: Path
):
    captured = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        loader_allowed_tools={"submit", "workspace_open"},
        profile_agent_architecture="mini-swe-agent",
        profile_tools=[],
        runtime_init_capture=captured,
    )

    assert captured["request"].allowed_tools == {"submit"}


def test_run_service_architecture_precedence_cli_override(monkeypatch, tmp_path: Path):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        runtime_init_capture=captured,
        agent_architecture="mini-swe-agent",
        runtime_agent_architecture_override="none",
        profile_agent_architecture="none",
    )
    assert captured["resolved_architecture"] == "mini-swe-agent"


def test_run_service_architecture_precedence_run_override(monkeypatch, tmp_path: Path):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        runtime_init_capture=captured,
        runtime_agent_architecture_override="mini-swe-agent",
        profile_agent_architecture="none",
    )
    assert captured["resolved_architecture"] == "mini-swe-agent"


def test_run_service_architecture_precedence_profile_default(monkeypatch, tmp_path: Path):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        runtime_init_capture=captured,
        profile_agent_architecture="mini-swe-agent",
    )
    assert captured["resolved_architecture"] == "mini-swe-agent"


def test_run_service_architecture_defaults_to_none(monkeypatch, tmp_path: Path):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        runtime_init_capture=captured,
    )
    assert captured["resolved_architecture"] == "none"


def test_run_service_preserves_invalid_patch_output(monkeypatch, tmp_path: Path):
    record, _ = _run_once(monkeypatch, tmp_path, raw_artifact="I'll inspect files first.")
    assert record["instance_id"] == "astropy__astropy-12907"
    assert record["model_patch"] == "I'll inspect files first."


def test_run_service_preserves_valid_patch_output(monkeypatch, tmp_path: Path):
    valid_patch = (
        "diff --git a/example.py b/example.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    record, _ = _run_once(monkeypatch, tmp_path, raw_artifact=valid_patch)
    assert record["model_patch"] == valid_patch


def test_run_service_normalizes_valid_patch_missing_trailing_newline(monkeypatch, tmp_path: Path):
    raw_patch = (
        "diff --git a/example.py b/example.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new"
    )
    record, _ = _run_once(monkeypatch, tmp_path, raw_artifact=raw_patch)
    assert record["model_patch"] == raw_patch + "\n"
    assert record["model_patch"].endswith("\n")


def test_run_service_creates_manifest(monkeypatch, tmp_path: Path):
    record, outcome = _run_once(monkeypatch, tmp_path, raw_artifact="")
    payload = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == outcome.run_id
    assert payload["predictions_path"] == str(outcome.predictions_path.resolve())
    assert payload["model_name"] == "fake-agent"
    assert payload["model_name_or_path"] == "fake/model"
    assert payload["evaluation"]["status"] == "not_run"
    assert payload["benchmark_name"] == "swebench_verified"
    assert payload["dataset_name"] == "SWE-bench/SWE-bench_Verified"
    assert payload["split"] == "test"
    assert payload["mode"] == "patch_only"
    assert record["instance_id"] == "astropy__astropy-12907"


def test_run_service_quiet_suppresses_per_task_logs(monkeypatch, tmp_path: Path, capsys):
    _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    out = capsys.readouterr().out
    assert "artifact_valid" not in out


def test_cli_run_verbose_option_defaults_to_quiet():
    verbose_option = cli.run.__defaults__[-1]
    assert getattr(verbose_option, "default", None) is False


def test_cli_run_prints_post_run_summary_by_default(monkeypatch, tmp_path: Path, capsys):
    run_root = tmp_path / "artifacts" / "2026-02-24_120000"
    run_log_path = run_root / "run.log"
    manifest_path = run_root / "manifest.json"

    fake_run_outcome = run_service.RunOutcome(
        run_id="2026-02-24_120000",
        benchmark_name="swebench_verified",
        split_name="test",
        mode_name="patch_only",
        tasks_total=1,
        valid_artifacts=1,
        invalid_artifacts=0,
        model_name_or_path="openrouter/free",
        predictions_path=run_root / "predictions.jsonl",
        manifest_path=manifest_path,
        run_log_path=run_log_path,
        manifest_payload={},
    )

    monkeypatch.setattr(cli, "load_run_config", lambda _: object())
    monkeypatch.setattr(cli, "execute_run", lambda **_kwargs: fake_run_outcome)
    monkeypatch.setattr(
        cli,
        "execute_run_log_summary",
        lambda **_kwargs: SimpleNamespace(
            terminal_lines=[
                "Post-run summary: run_id=2026-02-24_120000",
                "OpenRouter cost: total=$0.001000",
            ],
            manifest_path=manifest_path,
        ),
    )

    cli.run(agent="a.yaml", run_config="r.yaml")

    out = capsys.readouterr().out
    assert "Post-run summary: run_id=2026-02-24_120000" in out
    assert "OpenRouter cost: total=$0.001000" in out
    assert "Manifest updated with run_log_summary:" in out


def test_cli_run_can_skip_post_run_summary(monkeypatch, tmp_path: Path, capsys):
    run_root = tmp_path / "artifacts" / "2026-02-24_120000"
    fake_run_outcome = run_service.RunOutcome(
        run_id="2026-02-24_120000",
        benchmark_name="swebench_verified",
        split_name="test",
        mode_name="patch_only",
        tasks_total=1,
        valid_artifacts=1,
        invalid_artifacts=0,
        model_name_or_path="openrouter/free",
        predictions_path=run_root / "predictions.jsonl",
        manifest_path=run_root / "manifest.json",
        run_log_path=run_root / "run.log",
        manifest_payload={},
    )

    called = {"summary": 0}

    monkeypatch.setattr(cli, "load_run_config", lambda _: object())
    monkeypatch.setattr(cli, "execute_run", lambda **_kwargs: fake_run_outcome)
    monkeypatch.setattr(
        cli,
        "execute_run_log_summary",
        lambda **_kwargs: called.__setitem__("summary", called["summary"] + 1),
    )

    cli.run(agent="a.yaml", run_config="r.yaml", summary=False)

    out = capsys.readouterr().out
    assert "Post-run summary:" not in out
    assert called["summary"] == 0


def test_cli_run_passes_full_log_previews_flag_to_execute_run(monkeypatch, capsys):
    captured: dict = {}
    fake_run_outcome = run_service.RunOutcome(
        run_id="2026-02-24_120000",
        benchmark_name="swebench_verified",
        split_name="test",
        mode_name="patch_only",
        tasks_total=1,
        valid_artifacts=1,
        invalid_artifacts=0,
        model_name_or_path="openrouter/free",
        predictions_path=Path("artifacts/2026-02-24_120000/predictions.jsonl"),
        manifest_path=Path("artifacts/2026-02-24_120000/manifest.json"),
        run_log_path=Path("artifacts/2026-02-24_120000/run.log"),
        manifest_payload={},
    )

    monkeypatch.setattr(cli, "load_run_config", lambda _: object())

    def _fake_execute_run(**kwargs):
        captured.update(kwargs)
        return fake_run_outcome

    monkeypatch.setattr(cli, "execute_run", _fake_execute_run)

    cli.run(agent="a.yaml", run_config="r.yaml", summary=False, full_log_previews=True)

    capsys.readouterr()
    assert captured["full_log_previews"] is True


def test_cli_predict_forwards_full_log_previews_flag(monkeypatch):
    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run", _fake_run)

    cli.predict(agent="a.yaml", run_config="r.yaml", summary=False, full_log_previews=True)

    assert captured["full_log_previews"] is True


def test_cli_run_forwards_agent_architecture_override(monkeypatch, capsys):
    captured: dict = {}
    fake_run_outcome = run_service.RunOutcome(
        run_id="2026-02-24_120000",
        benchmark_name="swebench_verified",
        split_name="test",
        mode_name="patch_only",
        tasks_total=1,
        valid_artifacts=1,
        invalid_artifacts=0,
        model_name_or_path="openrouter/free",
        predictions_path=Path("artifacts/2026-02-24_120000/predictions.jsonl"),
        manifest_path=Path("artifacts/2026-02-24_120000/manifest.json"),
        run_log_path=Path("artifacts/2026-02-24_120000/run.log"),
        manifest_payload={},
    )

    monkeypatch.setattr(cli, "load_run_config", lambda _: object())
    monkeypatch.setattr(cli, "execute_run", lambda **kwargs: captured.update(kwargs) or fake_run_outcome)

    cli.run(
        agent="a.yaml",
        run_config="r.yaml",
        summary=False,
        agent_architecture="mini-swe-agent",
    )
    capsys.readouterr()

    assert captured["agent_architecture"] == "mini-swe-agent"


def test_cli_predict_forwards_agent_architecture_override(monkeypatch):
    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run", _fake_run)
    cli.predict(
        agent="a.yaml",
        run_config="r.yaml",
        summary=False,
        agent_architecture="mini-swe-agent",
    )

    assert captured["agent_architecture"] == "mini-swe-agent"


def test_cli_summarize_run_prints_summary_and_updates_manifest(monkeypatch, tmp_path: Path, capsys):
    run_log_path = tmp_path / "artifacts" / "2026-02-24_120000" / "run.log"
    run_log_path.parent.mkdir(parents=True)
    run_log_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "execute_run_log_summary",
        lambda **_kwargs: SimpleNamespace(
            terminal_lines=["Post-run summary: run_id=2026-02-24_120000"],
            manifest_path=run_log_path.parent / "manifest.json",
        ),
    )

    cli.summarize_run(run_log=str(run_log_path))

    out = capsys.readouterr().out
    assert "Post-run summary: run_id=2026-02-24_120000" in out
    assert "Manifest updated with run_log_summary:" in out


def test_cli_summarize_run_invalid_path_maps_to_bad_parameter(monkeypatch):
    monkeypatch.setattr(
        cli,
        "execute_run_log_summary",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad run log path")),
    )
    with pytest.raises(typer.BadParameter):
        cli.summarize_run(run_log="artifacts/invalid/run.log")


def test_run_service_verbose_prints_per_task_logs(monkeypatch, tmp_path: Path, capsys):
    _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=True)
    out = capsys.readouterr().out
    assert "artifact_valid=" in out


def test_run_service_writes_run_log_file(monkeypatch, tmp_path: Path):
    _, outcome = _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    content = outcome.run_log_path.read_text(encoding="utf-8")
    assert "Starting run:" in content
    assert "Run summary:" in content
    assert "artifact_valid=" in content
    assert "workspace_context" in content


def test_run_service_tools_enabled_fails_fast_when_workspace_not_tool_ready(monkeypatch, tmp_path: Path):
    backend_captured: dict = {}
    with pytest.raises(ValueError, match="benchmark.data_source=local"):
        _run_once(
            monkeypatch,
            tmp_path,
            raw_artifact="",
            mode="tools_enabled",
            workspace_tools_ready=False,
            workspace_kind="runner_root",
            workspace_reason="HF-backed tasks do not provide local repo workspaces",
            build_backend_capture=backend_captured,
        )

    assert backend_captured == {}


def test_run_service_patch_only_allows_non_tool_ready_workspace(monkeypatch, tmp_path: Path):
    record, _ = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="diff --git a/a b/a\n",
        mode="patch_only",
        workspace_tools_ready=False,
        workspace_kind="runner_root",
        workspace_reason="HF-backed tasks do not provide local repo workspaces",
    )
    assert record["instance_id"] == "astropy__astropy-12907"


def test_run_service_artifact_preview_truncates_by_default(monkeypatch, tmp_path: Path):
    long_artifact = "A" * 500 + "TAIL_MARKER"
    _, outcome = _run_once(monkeypatch, tmp_path, raw_artifact=long_artifact, verbose=False)
    run_log = outcome.run_log_path.read_text(encoding="utf-8")
    assert "artifact_preview=" in run_log
    assert "...[truncated]" in run_log


def test_run_service_artifact_preview_full_when_enabled(monkeypatch, tmp_path: Path):
    long_artifact = "A" * 500 + "TAIL_MARKER"
    _, outcome = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact=long_artifact,
        verbose=False,
        full_log_previews=True,
    )
    run_log = outcome.run_log_path.read_text(encoding="utf-8")
    assert "artifact_preview=" in run_log
    assert "TAIL_MARKER" in run_log
    assert "...[truncated]" not in run_log


def test_run_service_passes_full_log_previews_to_architecture_request(monkeypatch, tmp_path: Path):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        verbose=False,
        full_log_previews=True,
        build_backend_capture=captured,
    )
    assert captured["request"].full_log_previews is True


def test_run_service_builds_tools_mode_initial_user_message_with_workspace_context(
    monkeypatch, tmp_path: Path
):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        verbose=False,
        mode="tools_enabled",
        runtime_init_capture=captured,
    )
    initial_user_message = captured["request"].initial_user_message
    assert "Fix issue" in initial_user_message
    assert "<workspace_context>" in initial_user_message
    assert "tools_ready: true" in initial_user_message


def test_run_service_run_log_uses_structured_prefix(monkeypatch, tmp_path: Path):
    _, outcome = _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    first_line = outcome.run_log_path.read_text(encoding="utf-8").splitlines()[0]
    assert " | INFO" in first_line
    assert " | run_service.py:" in first_line
    assert " | Starting run:" in first_line


def test_run_service_passes_patch_submit_policy_into_tool_context(monkeypatch, tmp_path: Path):
    captured: dict = {}
    _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        patch_submit_policy="reject_retry",
        max_invalid_submit_attempts=5,
        runtime_init_capture=captured,
    )
    ctx = captured["request"].tool_registry.ctx
    assert ctx.expected_output_type == "patch"
    assert ctx.patch_submit_policy == "reject_retry"
    assert ctx.max_invalid_submit_attempts == 5


def test_run_service_tools_mode_no_submit_empty_patch_uses_cannot_produce_output_fallback(
    monkeypatch, tmp_path: Path
):
    runtime_tool_payload = {
        "mode": "tools_enabled",
        "loop_exit_reason": "no_tool_calls_without_submit",
        "budget_exhausted": False,
        "wall_time_exhausted": False,
        "termination_ack": False,
        "events": [],
    }
    record, outcome = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        mode="tools_enabled",
        terminated=False,
        runtime_tool_payload=runtime_tool_payload,
    )

    assert record["model_patch"].startswith(
        "CANNOT PRODUCE OUTPUT no_submit_without_termination:no_tool_calls_without_submit"
    )
    run_log = outcome.run_log_path.read_text(encoding="utf-8")
    assert "applied_no_submit_fallback=true" in run_log


def test_run_service_tools_mode_no_submit_invalid_text_uses_cannot_produce_output_fallback(
    monkeypatch, tmp_path: Path
):
    runtime_tool_payload = {
        "mode": "tools_enabled",
        "loop_exit_reason": "no_tool_calls_without_submit",
        "budget_exhausted": False,
        "wall_time_exhausted": False,
        "termination_ack": False,
        "events": [],
    }
    record, _ = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="I could not complete this task.",
        mode="tools_enabled",
        terminated=False,
        runtime_tool_payload=runtime_tool_payload,
    )

    assert record["model_patch"].startswith(
        "CANNOT PRODUCE OUTPUT no_submit_without_termination:no_tool_calls_without_submit"
    )


def test_run_service_tools_mode_preserves_explicit_cannot_produce_output_artifact(
    monkeypatch, tmp_path: Path
):
    runtime_tool_payload = {
        "mode": "tools_enabled",
        "loop_exit_reason": "no_tool_calls_without_submit",
        "budget_exhausted": False,
        "wall_time_exhausted": False,
        "termination_ack": False,
        "events": [],
    }
    explicit = "CANNOT PRODUCE OUTPUT no_tool_calls_without_submit:completion_cap_reached"
    record, outcome = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact=explicit,
        mode="tools_enabled",
        terminated=False,
        runtime_tool_payload=runtime_tool_payload,
    )

    assert record["model_patch"] == explicit
    run_log = outcome.run_log_path.read_text(encoding="utf-8")
    assert "applied_no_submit_fallback=true" not in run_log


def test_run_service_writes_tool_quality_artifacts_and_logs(monkeypatch, tmp_path: Path):
    runtime_tool_payload = {
        "mode": "tools_enabled",
        "loop_exit_reason": "submitted",
        "budget_exhausted": False,
        "wall_time_exhausted": False,
        "termination_ack": True,
        "events": [
            {
                "turn_index": 0,
                "call_index": 0,
                "tool_name": "bash",
                "is_termination_tool": False,
                "allowed": True,
                "executed": True,
                "success": True,
                "error_code": "none",
                "args_size_bytes": 12,
                "result_size_bytes": 24,
                "latency_ms": 3,
                "return_code": 0,
            }
        ],
    }
    _, outcome = _run_once(
        monkeypatch,
        tmp_path,
        raw_artifact="",
        verbose=False,
        mode="tools_enabled",
        runtime_tool_payload=runtime_tool_payload,
    )

    telemetry_path = outcome.predictions_path.parent / "tool_telemetry.jsonl"
    assert telemetry_path.exists()
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("row_type") == "tool_call" for row in rows)
    assert any(row.get("row_type") == "task_summary" for row in rows)

    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    tool_quality = manifest["tool_quality"]
    assert tool_quality["version"] == "v1"
    assert tool_quality["telemetry_path"] == str(telemetry_path.resolve())
    assert tool_quality["counts"]["tool_calls_total"] == 1
    assert tool_quality["counts"]["tasks_total"] == 1

    run_log = outcome.run_log_path.read_text(encoding="utf-8")
    assert "tool_quality task=" in run_log
    assert "tool_quality summary" in run_log


def test_derive_run_id_from_artifacts_path(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    p = artifacts_dir / run_id / "predictions.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    assert eval_service.derive_run_id_from_predictions(p, artifacts_dir) == run_id


def test_derive_run_id_rejects_non_canonical_layout(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    p = artifacts_dir / "2026-02-13_010203" / "predictions" / "model" / "test" / "patch_only" / "predictions.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        eval_service.derive_run_id_from_predictions(p, artifacts_dir)
