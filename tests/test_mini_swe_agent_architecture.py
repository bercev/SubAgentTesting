from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_architectures.base import ArchitectureRunRequest
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


def _make_request(tmp_path: Path, *, allowed_tools: set[str]):
    submitted_artifact: dict[str, str] = {}

    def _submit_cb(value: str) -> None:
        submitted_artifact["value"] = value

    registry = ToolRegistry(ToolContext(workspace_root=tmp_path, submit_callback=_submit_cb))
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
        mode_name="patch_only",
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

    assert result.final_artifact == "diff --git a/a b/a"
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
