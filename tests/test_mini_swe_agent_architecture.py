from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_architectures.base import ArchitectureRunRequest
from agent_architectures import mini_swe_agent as mini_module
from agent_architectures.mini_swe_agent import MiniSweAgentArchitecture
from runtime.model_backend import GenerationResult, ToolCall
from runtime.schemas import BenchmarkTask
from runtime.tools import ToolContext, ToolRegistry


class _FakeSubmitted(Exception):
    def __init__(self, message: dict[str, Any]):
        super().__init__("submitted")
        self.message = message


class _FakeLimitsExceeded(Exception):
    def __init__(self, message: dict[str, Any]):
        super().__init__("limits")
        self.message = message


@dataclass
class _FakeAgentConfig:
    system_template: str
    instance_template: str


@dataclass
class _FakeRunConfig:
    step_limit: int


class _FakeBackend:
    def __init__(self, responses: list[GenerationResult]):
        self._responses = list(responses)

    def generate(self, messages, tools=None, decoding=None):  # noqa: ANN001, ANN201
        del messages, tools, decoding
        if not self._responses:
            raise AssertionError("Unexpected generate call")
        return self._responses.pop(0)


class _FakeDefaultAgent:
    def __init__(self, *, model, environment, config, run_config):  # noqa: ANN001
        del config, run_config
        self._model = model
        self._environment = environment

    def run(self, instance_args):  # noqa: ANN001, ANN201
        messages = [{"role": "user", "content": instance_args["instruction"]}]
        self._environment.setup()
        try:
            assistant = self._model.query(messages)
            actions = assistant.get("extra", {}).get("actions", [])
            outputs = []
            for action in actions:
                outputs.append(self._environment.execute(action))
            self._model.format_observation_messages(assistant, outputs, {})
            return {
                "role": "exit",
                "content": "done",
                "extra": {
                    "exit_status": "Finished",
                    "submission": "",
                },
            }
        except (_FakeSubmitted, _FakeLimitsExceeded) as exc:
            return exc.message
        finally:
            self._environment.teardown()


def _make_request(
    tmp_path: Path,
    *,
    allowed_tools: set[str],
    mode_name: str = "patch_only",
    patch_submit_policy: str = "allow",
    max_invalid_submit_attempts: int = 3,
):
    submitted_artifact: dict[str, str] = {}

    def _submit_cb(value: str) -> None:
        submitted_artifact["value"] = value

    registry = ToolRegistry(
        ToolContext(
            workspace_root=tmp_path,
            submit_callback=_submit_cb,
            expected_output_type="patch",
            patch_submit_policy=patch_submit_policy,
            max_invalid_submit_attempts=max_invalid_submit_attempts,
        )
    )
    task = BenchmarkTask(
        task_id="task-1",
        instruction="Fix bug",
        resources={"repo": "astropy/astropy"},
        expected_output_type="patch",
    )
    request = ArchitectureRunRequest(
        task=task,
        system_prompt="system",
        initial_user_message="user",
        mode_name=mode_name,
        backend_config={"type": "openrouter", "model": "fake/model"},
        decoding_defaults={},
        tool_registry=registry,
        allowed_tools=allowed_tools,
        max_tool_calls=3,
        max_wall_time_s=30,
        termination_tool="submit",
        full_log_previews=False,
        api_log=None,
        architecture_config={},
    )
    return request, submitted_artifact


def test_mini_architecture_patch_only_submit_returns_artifact(monkeypatch, tmp_path: Path):
    responses = [
        GenerationResult(
            assistant_text="",
            tool_calls=[ToolCall(name="submit", arguments={"final_artifact": "diff --git a/a b/a"})],
        )
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(tmp_path, allowed_tools={"submit"})

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _FakeDefaultAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.final_artifact.startswith("diff --git a/a b/a")
    assert result.metadata["terminated"] is True
    assert payload["loop_exit_reason"] == "submitted"
    assert payload["events"][0]["tool_name"] == "submit"
    assert payload["events"][0]["success"] is True


def test_mini_architecture_denies_disallowed_tool(monkeypatch, tmp_path: Path):
    responses = [
        GenerationResult(
            assistant_text="",
            tool_calls=[ToolCall(name="workspace_list", arguments={"path": "."})],
        )
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(tmp_path, allowed_tools={"submit"})

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _FakeDefaultAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.final_artifact == ""
    assert result.metadata["terminated"] is False
    assert payload["events"][0]["tool_name"] == "workspace_list"
    assert payload["events"][0]["error_code"] == "not_allowed"


def test_import_mini_components_supports_v2_module_layout(monkeypatch):
    class _V2AgentConfig:
        def __init__(self, **kwargs):  # noqa: ANN003
            self._data = dict(kwargs)

        def model_dump(self) -> dict[str, Any]:
            return dict(self._data)

    class _V2DefaultAgent:
        def __init__(self, model, env, **kwargs):  # noqa: ANN001, ANN003
            del model, env
            self.kwargs = dict(kwargs)
            _V2DefaultAgent.last_kwargs = dict(kwargs)

        def run(self, *, task: str, **kwargs):  # noqa: ANN003, ANN201
            del kwargs
            return {"submission": task}

    fake_modules = {
        "minisweagent": SimpleNamespace(),
        "minisweagent.exceptions": SimpleNamespace(
            Submitted=_FakeSubmitted,
            LimitsExceeded=_FakeLimitsExceeded,
        ),
        "minisweagent.agents.default": SimpleNamespace(
            AgentConfig=_V2AgentConfig,
            DefaultAgent=_V2DefaultAgent,
        ),
    }

    def _fake_import(name: str):  # noqa: ANN202
        if name in fake_modules:
            return fake_modules[name]
        raise ImportError(name)

    monkeypatch.setattr(mini_module.importlib, "import_module", _fake_import)

    agent_config_cls, run_config_cls, default_agent_cls, submitted_exc, limits_exc = (
        mini_module._import_mini_components()
    )

    assert submitted_exc is _FakeSubmitted
    assert limits_exc is _FakeLimitsExceeded

    config = agent_config_cls(system_template="sys", instance_template="{instruction}")
    run_config = run_config_cls(step_limit=7)
    agent = default_agent_cls(
        model=object(),
        environment=object(),
        config=config,
        run_config=run_config,
    )
    result = agent.run({"instruction": "hello"})

    assert result["submission"] == "hello"
    assert _V2DefaultAgent.last_kwargs["instance_template"] == "{{task}}"


def test_mini_architecture_reject_retry_submit_continues_until_valid(monkeypatch, tmp_path: Path):
    class _LoopingAgent:
        def __init__(self, *, model, environment, config, run_config):  # noqa: ANN001
            del config, run_config
            self._model = model
            self._environment = environment

        def run(self, instance_args):  # noqa: ANN001, ANN201
            messages = [{"role": "user", "content": instance_args["instruction"]}]
            self._environment.setup()
            try:
                while True:
                    assistant = self._model.query(messages)
                    actions = assistant.get("extra", {}).get("actions", [])
                    if not actions:
                        return {"submission": ""}
                    outputs = [self._environment.execute(action) for action in actions]
                    messages.extend(self._model.format_observation_messages(assistant, outputs, {}))
            except (_FakeSubmitted, _FakeLimitsExceeded) as exc:
                return exc.message
            finally:
                self._environment.teardown()

    responses = [
        GenerationResult(
            assistant_text="",
            tool_calls=[ToolCall(name="submit", arguments={"final_artifact": ""})],
        ),
        GenerationResult(
            assistant_text="",
            tool_calls=[
                ToolCall(
                    name="submit",
                    arguments={
                        "final_artifact": (
                            "diff --git a/a b/a\n"
                            "--- a/a\n"
                            "+++ b/a\n"
                            "@@ -1 +1 @@\n"
                            "-a\n"
                            "+b\n"
                        )
                    },
                )
            ],
        ),
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(
        tmp_path,
        allowed_tools={"submit"},
        patch_submit_policy="reject_retry",
        max_invalid_submit_attempts=3,
    )

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _LoopingAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.final_artifact.startswith("diff --git a/a b/a")
    assert result.metadata["terminated"] is True
    assert result.metadata["invalid_submit_attempts"] == 1
    assert result.metadata["last_invalid_submit_reason"] == "empty_output"
    assert payload["events"][0]["success"] is False
    assert payload["events"][1]["success"] is True


def test_mini_architecture_reject_retry_exhaustion_sets_terminal_reason(monkeypatch, tmp_path: Path):
    class _LoopingAgent:
        def __init__(self, *, model, environment, config, run_config):  # noqa: ANN001
            del config, run_config
            self._model = model
            self._environment = environment

        def run(self, instance_args):  # noqa: ANN001, ANN201
            messages = [{"role": "user", "content": instance_args["instruction"]}]
            self._environment.setup()
            try:
                assistant = self._model.query(messages)
                actions = assistant.get("extra", {}).get("actions", [])
                outputs = [self._environment.execute(action) for action in actions]
                messages.extend(self._model.format_observation_messages(assistant, outputs, {}))
                return {"submission": ""}
            except (_FakeSubmitted, _FakeLimitsExceeded) as exc:
                return exc.message
            finally:
                self._environment.teardown()

    responses = [
        GenerationResult(
            assistant_text="",
            tool_calls=[ToolCall(name="submit", arguments={"final_artifact": ""})],
        )
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(
        tmp_path,
        allowed_tools={"submit"},
        patch_submit_policy="reject_retry",
        max_invalid_submit_attempts=1,
    )

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _LoopingAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.final_artifact == ""
    assert result.metadata["terminated"] is False
    assert result.metadata["invalid_submit_attempts"] == 1
    assert result.metadata["last_invalid_submit_reason"] == "empty_output"
    assert payload["loop_exit_reason"] == "invalid_submission_retries_exhausted"


def test_mini_architecture_tools_enabled_no_tool_calls_repair_retry_recovers(monkeypatch, tmp_path: Path):
    class _LoopingAgent:
        def __init__(self, *, model, environment, config, run_config):  # noqa: ANN001
            del config, run_config
            self._model = model
            self._environment = environment

        def run(self, instance_args):  # noqa: ANN001, ANN201
            messages = [{"role": "user", "content": instance_args["instruction"]}]
            self._environment.setup()
            try:
                while True:
                    assistant = self._model.query(messages)
                    actions = assistant.get("extra", {}).get("actions", [])
                    outputs = [self._environment.execute(action) for action in actions]
                    messages.extend(self._model.format_observation_messages(assistant, outputs, {}))
            except (_FakeSubmitted, _FakeLimitsExceeded) as exc:
                return exc.message
            finally:
                self._environment.teardown()

    responses = [
        GenerationResult(assistant_text="thinking", tool_calls=[]),
        GenerationResult(
            assistant_text="",
            tool_calls=[ToolCall(name="submit", arguments={"final_artifact": "diff --git a/a b/a"})],
        ),
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(
        tmp_path,
        allowed_tools={"submit"},
        mode_name="tools_enabled",
    )

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _LoopingAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is True
    assert payload["loop_exit_reason"] == "submitted"
    assert payload["no_tool_call_repair_attempted"] is True
    assert payload["no_tool_call_failure_after_repair"] is False
    assert payload["no_tool_call_cap_hit"] is False
    assert payload["no_tool_call_terminal_artifact_emitted"] is False


def test_mini_architecture_tools_enabled_no_tool_calls_after_retry_fails(monkeypatch, tmp_path: Path):
    class _LoopingAgent:
        def __init__(self, *, model, environment, config, run_config):  # noqa: ANN001
            del config, run_config
            self._model = model
            self._environment = environment

        def run(self, instance_args):  # noqa: ANN001, ANN201
            messages = [{"role": "user", "content": instance_args["instruction"]}]
            self._environment.setup()
            try:
                self._model.query(messages)
                return {"submission": ""}
            except (_FakeSubmitted, _FakeLimitsExceeded) as exc:
                return exc.message
            finally:
                self._environment.teardown()

    responses = [
        GenerationResult(assistant_text="thinking", tool_calls=[]),
        GenerationResult(assistant_text="still thinking", tool_calls=[]),
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(
        tmp_path,
        allowed_tools={"submit"},
        mode_name="tools_enabled",
    )

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _LoopingAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is False
    assert result.final_artifact == "CANNOT PRODUCE OUTPUT no_tool_calls_without_submit:after_repair"
    assert payload["loop_exit_reason"] == "no_tool_calls_without_submit"
    assert payload["no_tool_call_repair_attempted"] is True
    assert payload["no_tool_call_failure_after_repair"] is True
    assert payload["no_tool_call_cap_hit"] is False
    assert payload["no_tool_call_terminal_artifact_emitted"] is True


def test_mini_architecture_tools_enabled_no_tool_calls_after_retry_cap_hit(monkeypatch, tmp_path: Path):
    class _LoopingAgent:
        def __init__(self, *, model, environment, config, run_config):  # noqa: ANN001
            del config, run_config
            self._model = model
            self._environment = environment

        def run(self, instance_args):  # noqa: ANN001, ANN201
            messages = [{"role": "user", "content": instance_args["instruction"]}]
            self._environment.setup()
            try:
                self._model.query(messages)
                return {"submission": ""}
            except (_FakeSubmitted, _FakeLimitsExceeded) as exc:
                return exc.message
            finally:
                self._environment.teardown()

    responses = [
        GenerationResult(
            assistant_text="thinking",
            tool_calls=[],
            finish_reason="length",
            completion_tokens=4096,
        ),
        GenerationResult(
            assistant_text="still thinking",
            tool_calls=[],
            finish_reason="stop",
            completion_tokens=4096,
        ),
    ]
    backend = _FakeBackend(responses)
    request, _submitted = _make_request(
        tmp_path,
        allowed_tools={"submit"},
        mode_name="tools_enabled",
    )
    request.decoding_defaults = {"max_tokens": 4096}

    monkeypatch.setattr("agent_architectures.mini_swe_agent.build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        "agent_architectures.mini_swe_agent._import_mini_components",
        lambda: (_FakeAgentConfig, _FakeRunConfig, _LoopingAgent, _FakeSubmitted, _FakeLimitsExceeded),
    )

    result = MiniSweAgentArchitecture().run_task(request)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is False
    assert result.final_artifact == (
        "CANNOT PRODUCE OUTPUT no_tool_calls_without_submit:completion_cap_reached"
    )
    assert payload["loop_exit_reason"] == "no_tool_calls_without_submit"
    assert payload["no_tool_call_repair_attempted"] is True
    assert payload["no_tool_call_failure_after_repair"] is True
    assert payload["no_tool_call_cap_hit"] is True
    assert payload["no_tool_call_terminal_artifact_emitted"] is True
