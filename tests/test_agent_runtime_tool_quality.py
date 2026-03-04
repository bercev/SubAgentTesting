from __future__ import annotations

import copy
from typing import Any, Dict, List

from runtime.agent_runtime import AgentRuntime
from runtime.model_backend import GenerationResult, ToolCall
from runtime.schemas import BenchmarkTask


class _SequenceBackend:
    def __init__(self, responses: List[GenerationResult]) -> None:
        self._responses = list(responses)

    def generate(self, messages, tools=None, decoding=None) -> GenerationResult:  # noqa: ANN001, ANN201
        if not self._responses:
            raise AssertionError("Unexpected generate() call with no remaining responses")
        return self._responses.pop(0)


class _CapturingSequenceBackend(_SequenceBackend):
    def __init__(self, responses: List[GenerationResult]) -> None:
        super().__init__(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate(self, messages, tools=None, decoding=None) -> GenerationResult:  # noqa: ANN001, ANN201
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "decoding": copy.deepcopy(decoding),
            }
        )
        return super().generate(messages, tools=tools, decoding=decoding)


class _ToolRegistryStub:
    def __init__(self, outputs: Dict[str, Dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls: List[str] = []

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(name)
        return self.outputs.get(name, {})


class _ExplodingToolRegistry:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(name)
        raise RuntimeError("boom")


class _SequentialToolRegistry:
    def __init__(self, outputs: List[Dict[str, Any]]) -> None:
        self._outputs = list(outputs)
        self.calls: List[str] = []

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(name)
        if not self._outputs:
            raise AssertionError("Unexpected execute call with no remaining tool outputs")
        return self._outputs.pop(0)


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="astropy__astropy-12907",
        instruction="Fix the bug",
        resources={"repo": "astropy/astropy"},
        expected_output_type="patch",
    )


def test_agent_runtime_records_successful_call_and_submit_termination():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_list", arguments={"path": "."})],
            ),
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="submit", arguments={"final_artifact": "diff --git a/a b/a"})],
            ),
        ]
    )
    registry = _ToolRegistryStub(
        {
            "workspace_list": {"entries": []},
            "submit": {"submitted": True},
        }
    )
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"workspace_list", "submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]
    events = payload["events"]

    assert result.metadata["terminated"] is True
    assert payload["termination_ack"] is True
    assert payload["loop_exit_reason"] == "submitted"
    assert len(events) == 2
    assert events[0]["tool_name"] == "workspace_list"
    assert events[0]["success"] is True
    assert events[0]["error_code"] == "none"
    assert events[1]["tool_name"] == "submit"
    assert events[1]["is_termination_tool"] is True


def test_agent_runtime_records_not_allowed_tool_call():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_write", arguments={"path": "a.txt", "content": "x"})],
            ),
            GenerationResult(
                assistant_text="diff --git a/a b/a",
                tool_calls=[],
            ),
        ]
    )
    registry = _ToolRegistryStub({})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]
    events = payload["events"]

    assert result.metadata["terminated"] is False
    assert result.final_artifact == ""
    assert payload["loop_exit_reason"] == "no_tool_calls_without_submit"
    assert len(events) == 1
    assert events[0]["allowed"] is False
    assert events[0]["executed"] is False
    assert events[0]["success"] is False
    assert events[0]["error_code"] == "not_allowed"
    assert registry.calls == []


def test_agent_runtime_allows_tools_when_allowlist_is_none():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_list", arguments={"path": "."})],
            ),
            GenerationResult(assistant_text="diff --git a/a b/a", tool_calls=[]),
        ]
    )
    registry = _ToolRegistryStub({"workspace_list": {"entries": []}})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools=None,
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    events = result.metadata["tool_quality_runtime"]["events"]

    assert len(events) == 1
    assert events[0]["tool_name"] == "workspace_list"
    assert events[0]["allowed"] is True
    assert events[0]["success"] is True
    assert registry.calls == ["workspace_list"]


def test_agent_runtime_denies_tools_when_allowlist_is_empty_set():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_list", arguments={"path": "."})],
            ),
            GenerationResult(assistant_text="diff --git a/a b/a", tool_calls=[]),
        ]
    )
    registry = _ToolRegistryStub({"workspace_list": {"entries": []}})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools=set(),
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    events = result.metadata["tool_quality_runtime"]["events"]

    assert len(events) == 1
    assert events[0]["tool_name"] == "workspace_list"
    assert events[0]["allowed"] is False
    assert events[0]["error_code"] == "not_allowed"
    assert registry.calls == []


def test_agent_runtime_marks_nonzero_returncode_tool_result_as_failure():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="bash", arguments={"cmd": "pytest"})],
            ),
            GenerationResult(assistant_text="", tool_calls=[]),
        ]
    )
    registry = _ToolRegistryStub({"bash": {"returncode": 2, "output": "failed"}})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"bash"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]
    events = payload["events"]

    assert len(events) == 1
    assert events[0]["tool_name"] == "bash"
    assert events[0]["success"] is False
    assert events[0]["error_code"] == "nonzero_returncode"
    assert events[0]["return_code"] == 2


def test_agent_runtime_records_tool_budget_exhaustion_exit_reason():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_list", arguments={"path": "."})],
            ),
        ]
    )
    registry = _ToolRegistryStub({"workspace_list": {"entries": []}})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"workspace_list"},
        max_tool_calls=1,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is False
    assert payload["budget_exhausted"] is True
    assert payload["loop_exit_reason"] == "tool_budget_exhausted"
    assert len(payload["events"]) == 1


def test_agent_runtime_records_wall_time_exhaustion_exit_reason(monkeypatch):
    backend = _SequenceBackend([])
    registry = _ToolRegistryStub({})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"workspace_list"},
        max_tool_calls=5,
        max_wall_time_s=0,
        termination_tool="submit",
        mode_name="tools_enabled",
    )
    clock = iter([10.0, 11.0])
    monkeypatch.setattr("runtime.agent_runtime.time.monotonic", lambda: next(clock))

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is False
    assert payload["wall_time_exhausted"] is True
    assert payload["loop_exit_reason"] == "wall_time_exhausted"
    assert payload["events"] == []


def test_agent_runtime_continues_after_tool_execution_exception():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_open", arguments={"raw": "bad"})],
            ),
            GenerationResult(
                assistant_text="diff --git a/a b/a\nindex 1111111..2222222 100644",
                tool_calls=[],
            ),
        ]
    )
    registry = _ExplodingToolRegistry()
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"workspace_open"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]
    events = payload["events"]

    assert result.metadata["terminated"] is False
    assert result.final_artifact == ""
    assert payload["loop_exit_reason"] == "no_tool_calls_without_submit"
    assert len(events) == 1
    assert events[0]["tool_name"] == "workspace_open"
    assert events[0]["success"] is False
    assert events[0]["error_code"] == "execution_exception"


def test_agent_runtime_tools_mode_requires_submit_to_finalize():
    backend = _SequenceBackend([GenerationResult(assistant_text="diff --git a/a b/a", tool_calls=[])])
    registry = _ToolRegistryStub({})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is False
    assert result.final_artifact == ""
    assert payload["termination_ack"] is False
    assert payload["loop_exit_reason"] == "no_tool_calls_without_submit"


def test_agent_runtime_patch_only_still_accepts_assistant_text_without_submit():
    backend = _SequenceBackend([GenerationResult(assistant_text="diff --git a/a b/a", tool_calls=[])])
    registry = _ToolRegistryStub({})
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="patch_only",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=None, decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is True
    assert result.final_artifact == "diff --git a/a b/a"
    assert payload["loop_exit_reason"] == "no_tool_calls"


def test_agent_runtime_reject_retry_submit_continues_until_valid_submission():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="submit", arguments={"final_artifact": ""})],
            ),
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="submit", arguments={"final_artifact": "diff --git a/a b/a"})],
            ),
        ]
    )
    registry = _SequentialToolRegistry(
        [
            {
                "error": "invalid patch submission",
                "invalid_submission": True,
                "invalid_submission_reason": "empty_output",
                "invalid_submit_attempts": 1,
                "max_invalid_submit_attempts": 3,
                "retryable": True,
            },
            {"submitted": True},
        ]
    )
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="patch_only",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]
    events = payload["events"]

    assert result.metadata["terminated"] is True
    assert result.final_artifact == "diff --git a/a b/a"
    assert payload["loop_exit_reason"] == "submitted"
    assert result.metadata["invalid_submit_attempts"] == 1
    assert result.metadata["last_invalid_submit_reason"] == "empty_output"
    assert len(events) == 2
    assert events[0]["success"] is False
    assert events[1]["success"] is True


def test_agent_runtime_reject_retry_terminal_reason_stops_without_termination():
    backend = _SequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="submit", arguments={"final_artifact": ""})],
            ),
        ]
    )
    registry = _SequentialToolRegistry(
        [
            {
                "error": "invalid patch submission",
                "invalid_submission": True,
                "invalid_submission_reason": "empty_output",
                "invalid_submit_attempts": 3,
                "max_invalid_submit_attempts": 3,
                "retryable": False,
                "invalid_submission_terminal_reason": "invalid_submission_retries_exhausted",
            },
        ]
    )
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="patch_only",
    )

    result = runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)
    payload = result.metadata["tool_quality_runtime"]

    assert result.metadata["terminated"] is False
    assert result.final_artifact == ""
    assert payload["loop_exit_reason"] == "invalid_submission_retries_exhausted"
    assert result.metadata["invalid_submit_attempts"] == 3
    assert result.metadata["last_invalid_submit_reason"] == "empty_output"


def test_agent_runtime_serializes_and_truncates_tool_results_in_tool_messages():
    backend = _CapturingSequenceBackend(
        [
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="workspace_open", arguments={"path": "x.py"})],
            ),
            GenerationResult(
                assistant_text="",
                tool_calls=[ToolCall(name="submit", arguments={"final_artifact": ""})],
            ),
        ]
    )
    registry = _ToolRegistryStub(
        {
            "workspace_open": {
                "content": "x" * 13050,
                "start_line": 1,
                "end_line": 1,
                "truncated": True,
            },
            "submit": {"submitted": True},
        }
    )
    runtime = AgentRuntime(
        backend=backend,
        tool_registry=registry,
        allowed_tools={"workspace_open", "submit"},
        max_tool_calls=5,
        max_wall_time_s=60,
        termination_tool="submit",
        mode_name="tools_enabled",
    )

    runtime.run(task=_task(), system_prompt="prompt", initial_user_message="Fix the bug", tool_schemas=[], decoding_defaults=None)

    second_call_messages = backend.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0]["content"]
    assert content.startswith("{")
    assert '"content"' in content
    assert "...[truncated]" in content
