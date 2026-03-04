from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from runtime.backend_factory import build_backend
from runtime.schemas import AgentResult

from agent_architectures.base import ArchitectureRunRequest, AgentArchitecture, filter_tool_schemas
from agent_architectures.constants import ARCHITECTURE_MINI_SWE_AGENT
from agent_architectures.telemetry_adapter import (
    build_runtime_payload,
    json_size_bytes,
    pick_repo_metadata,
    serialize_tool_message,
)


def _import_mini_components() -> tuple[Any, Any, Any, Any, Any]:
    """Import mini-swe-agent runtime components lazily."""

    try:
        mini_module = importlib.import_module("minisweagent")
    except ImportError as exc:
        raise RuntimeError(
            "agent_architecture=mini-swe-agent requires the `mini-swe-agent` package"
        ) from exc

    try:
        exceptions_module = importlib.import_module("minisweagent.exceptions")
    except ImportError as exc:
        raise RuntimeError("Failed to import minisweagent.exceptions") from exc

    required_names = (
        "AgentConfig",
        "AgentRunConfig",
        "DefaultAgent",
    )
    missing = [name for name in required_names if not hasattr(mini_module, name)]
    if missing:
        raise RuntimeError(
            "mini-swe-agent import is missing required symbols: " + ", ".join(missing)
        )

    if not hasattr(exceptions_module, "Submitted") or not hasattr(exceptions_module, "LimitsExceeded"):
        raise RuntimeError("mini-swe-agent exceptions module is missing Submitted/LimitsExceeded")

    return (
        mini_module.AgentConfig,
        mini_module.AgentRunConfig,
        mini_module.DefaultAgent,
        exceptions_module.Submitted,
        exceptions_module.LimitsExceeded,
    )


@dataclass
class _MiniRunState:
    """Mutable telemetry and exit-state shared across model + environment."""

    mode_name: str
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0
    budget_exhausted: bool = False
    wall_time_exhausted: bool = False
    termination_ack: bool = False
    loop_exit_reason: str = "unknown"
    submitted_artifact: str = ""


class _MiniFunctionCallingModel:
    """Model adapter that reuses existing OpenRouter backend function calling."""

    def __init__(
        self,
        *,
        backend: Any,
        tool_schemas: list[dict[str, Any]],
        decoding_defaults: Mapping[str, Any],
        mode_name: str,
        run_state: _MiniRunState,
        limits_exceeded_exc: type[BaseException],
    ) -> None:
        self._backend = backend
        self._tool_schemas = list(tool_schemas)
        self._decoding_defaults = dict(decoding_defaults)
        self._mode_name = mode_name
        self._run_state = run_state
        self._limits_exceeded_exc = limits_exceeded_exc
        self._turn_index = 0

    def serialize(self) -> dict[str, Any]:
        return {
            "name": "portable_openrouter_function_model",
            "tool_count": len(self._tool_schemas),
        }

    def get_template_vars(self) -> dict[str, Any]:
        return {}

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    @staticmethod
    def _sanitize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Drop non-OpenAI fields before passing conversation history to backend."""

        allowed = {"role", "content", "tool_calls", "name", "tool_call_id"}
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            row = {k: v for k, v in message.items() if k in allowed}
            sanitized.append(row)
        return sanitized

    def _raise_no_tool_calls(self) -> None:
        self._run_state.loop_exit_reason = "no_tool_calls_without_submit"
        raise self._limits_exceeded_exc(
            {
                "role": "exit",
                "content": "Agent returned no tool calls",
                "extra": {
                    "exit_status": "NoToolCalls",
                    "submission": self._run_state.submitted_artifact,
                },
            }
        )

    def query(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        backend_messages = self._sanitize_messages(messages)
        result = self._backend.generate(
            backend_messages,
            tools=self._tool_schemas if self._tool_schemas else None,
            decoding=self._decoding_defaults,
        )

        if not result.tool_calls:
            self._raise_no_tool_calls()

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": result.assistant_text,
        }

        actions: list[dict[str, Any]] = []
        tool_calls_payload: list[dict[str, Any]] = []
        call_base = self._run_state.tool_calls_made
        for idx, tool_call in enumerate(result.tool_calls):
            call_id = f"call_{call_base}_{idx}"
            arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            actions.append(
                {
                    "command": tool_call.name,
                    "tool_name": tool_call.name,
                    "tool_arguments": arguments,
                    "tool_call_id": call_id,
                    "turn_index": self._turn_index,
                    "call_index": idx,
                }
            )
            tool_calls_payload.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )

        self._turn_index += 1
        assistant_message["tool_calls"] = tool_calls_payload
        assistant_message["extra"] = {"actions": actions}
        return assistant_message

    def format_observation_messages(
        self,
        message: Mapping[str, Any],
        outputs: Sequence[Any],
        template_vars: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        del template_vars
        actions = message.get("extra", {}).get("actions", []) if isinstance(message, Mapping) else []
        formatted: list[dict[str, Any]] = []
        for idx, output in enumerate(outputs):
            action = actions[idx] if idx < len(actions) and isinstance(actions[idx], Mapping) else {}
            tool_name = action.get("tool_name") if isinstance(action.get("tool_name"), str) else "tool"
            tool_call_id = (
                action.get("tool_call_id")
                if isinstance(action.get("tool_call_id"), str)
                else f"unknown_{idx}"
            )
            formatted.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tool_call_id,
                    "content": serialize_tool_message(output),
                }
            )
        return formatted


class _MiniToolEnvironment:
    """Environment adapter that dispatches mini actions into existing ToolRegistry."""

    def __init__(
        self,
        *,
        tool_registry: Any,
        allowed_tools: set[str],
        max_tool_calls: int,
        max_wall_time_s: int,
        termination_tool: str,
        run_state: _MiniRunState,
        submitted_exc: type[BaseException],
        limits_exceeded_exc: type[BaseException],
    ) -> None:
        self._tool_registry = tool_registry
        self._allowed_tools = set(allowed_tools)
        self._max_tool_calls = max(1, int(max_tool_calls))
        self._max_wall_time_s = max(1, int(max_wall_time_s))
        self._termination_tool = termination_tool
        self._run_state = run_state
        self._submitted_exc = submitted_exc
        self._limits_exceeded_exc = limits_exceeded_exc
        self._started_at = time.monotonic()

    def serialize(self) -> dict[str, Any]:
        return {
            "name": "portable_tool_environment",
            "max_tool_calls": self._max_tool_calls,
            "max_wall_time_s": self._max_wall_time_s,
            "allowed_tools": sorted(self._allowed_tools),
        }

    def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    def set_env_variables(self, env_variables: Mapping[str, Any]) -> None:
        del env_variables
        return None

    def _raise_limits_exceeded(self, reason: str) -> None:
        self._run_state.loop_exit_reason = reason
        raise self._limits_exceeded_exc(
            {
                "role": "exit",
                "content": reason,
                "extra": {
                    "exit_status": "LimitsExceeded",
                    "submission": self._run_state.submitted_artifact,
                },
            }
        )

    def _enforce_limits(self) -> None:
        if (time.monotonic() - self._started_at) > self._max_wall_time_s:
            self._run_state.wall_time_exhausted = True
            self._raise_limits_exceeded("wall_time_exhausted")
        if self._run_state.tool_calls_made >= self._max_tool_calls:
            self._run_state.budget_exhausted = True
            self._raise_limits_exceeded("tool_budget_exhausted")

    def execute(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        self._enforce_limits()

        tool_name = action.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return {"error": "invalid action: missing tool_name"}
        tool_name = tool_name.strip()

        arguments = action.get("tool_arguments")
        if not isinstance(arguments, MutableMapping):
            arguments = {}

        turn_index = action.get("turn_index") if isinstance(action.get("turn_index"), int) else 0
        call_index = action.get("call_index") if isinstance(action.get("call_index"), int) else 0

        self._run_state.tool_calls_made += 1
        event: dict[str, Any] = {
            "turn_index": max(0, turn_index),
            "call_index": max(0, call_index),
            "tool_name": tool_name,
            "is_termination_tool": tool_name == self._termination_tool,
            "allowed": False,
            "executed": False,
            "success": False,
            "error_code": "tool_error",
            "args_size_bytes": json_size_bytes(arguments),
            "result_size_bytes": 0,
            "latency_ms": 0,
            "return_code": None,
        }

        if tool_name not in self._allowed_tools:
            event.update({"error_code": "not_allowed"})
            self._run_state.events.append(event)
            return {"error": f"Tool {tool_name} not allowed"}

        event["allowed"] = True
        started = time.monotonic()
        execution_exception: Exception | None = None
        try:
            tool_result = self._tool_registry.execute(tool_name, dict(arguments))
        except Exception as exc:  # pragma: no cover - defensive compatibility guard
            execution_exception = exc
            tool_result = {
                "error": f"tool execution exception: {exc.__class__.__name__}: {exc}",
            }

        latency_ms = int(max(0.0, (time.monotonic() - started) * 1000.0))
        success = True
        error_code = "none"
        return_code = None

        if execution_exception is not None:
            success = False
            error_code = "execution_exception"
        elif isinstance(tool_result, dict):
            if "error" in tool_result:
                success = False
                error_code = "tool_error"
            elif isinstance(tool_result.get("returncode"), int) and tool_result.get("returncode") != 0:
                success = False
                error_code = "nonzero_returncode"
            elif tool_result.get("success") is False:
                success = False
                error_code = "tool_error"
            if isinstance(tool_result.get("returncode"), int):
                return_code = tool_result.get("returncode")

        event.update(
            {
                "executed": True,
                "success": success,
                "error_code": error_code,
                "result_size_bytes": json_size_bytes(tool_result),
                "latency_ms": latency_ms,
                "return_code": return_code,
            }
        )
        self._run_state.events.append(event)

        if (
            tool_name == self._termination_tool
            and isinstance(tool_result, Mapping)
            and bool(tool_result.get("submitted"))
        ):
            final_artifact = arguments.get("final_artifact", "")
            artifact_text = final_artifact if isinstance(final_artifact, str) else ""
            self._run_state.submitted_artifact = artifact_text
            self._run_state.termination_ack = True
            self._run_state.loop_exit_reason = "submitted"
            raise self._submitted_exc(
                {
                    "role": "exit",
                    "content": "submitted",
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": artifact_text,
                    },
                }
            )

        return dict(tool_result) if isinstance(tool_result, Mapping) else {"result": tool_result}


class MiniSweAgentArchitecture(AgentArchitecture):
    """mini-swe-agent loop with legacy backend/tools + telemetry compatibility."""

    architecture_id = ARCHITECTURE_MINI_SWE_AGENT

    def run_task(self, request: ArchitectureRunRequest) -> AgentResult:
        agent_config_cls, run_config_cls, default_agent_cls, submitted_exc, limits_exc = (
            _import_mini_components()
        )

        backend_type = request.backend_config.get("type", "openrouter")
        if backend_type != "openrouter":
            raise ValueError(
                "mini-swe-agent architecture currently supports only backend.type=openrouter"
            )

        backend = build_backend(
            request.backend_config,
            event_logger=request.api_log,
            full_log_previews=request.full_log_previews,
        )
        run_state = _MiniRunState(mode_name=request.mode_name)
        tool_schemas = filter_tool_schemas(request.tool_registry, request.allowed_tools)

        model = _MiniFunctionCallingModel(
            backend=backend,
            tool_schemas=tool_schemas,
            decoding_defaults=request.decoding_defaults,
            mode_name=request.mode_name,
            run_state=run_state,
            limits_exceeded_exc=limits_exc,
        )
        environment = _MiniToolEnvironment(
            tool_registry=request.tool_registry,
            allowed_tools=request.allowed_tools,
            max_tool_calls=request.max_tool_calls,
            max_wall_time_s=request.max_wall_time_s,
            termination_tool=request.termination_tool,
            run_state=run_state,
            submitted_exc=submitted_exc,
            limits_exceeded_exc=limits_exc,
        )

        agent_config = agent_config_cls(
            system_template=request.system_prompt,
            instance_template="{instruction}",
        )
        run_config = run_config_cls(step_limit=max(1, int(request.max_tool_calls)))
        agent = default_agent_cls(
            model=model,
            environment=environment,
            config=agent_config,
            run_config=run_config,
        )

        run_result = agent.run({"instruction": request.initial_user_message})

        final_artifact = run_state.submitted_artifact
        if not final_artifact and isinstance(run_result, Mapping):
            extra = run_result.get("extra")
            if isinstance(extra, Mapping):
                submission = extra.get("submission")
                if isinstance(submission, str):
                    final_artifact = submission

        if run_state.loop_exit_reason == "unknown":
            if run_state.termination_ack:
                run_state.loop_exit_reason = "submitted"
            elif run_state.wall_time_exhausted:
                run_state.loop_exit_reason = "wall_time_exhausted"
            elif run_state.budget_exhausted:
                run_state.loop_exit_reason = "tool_budget_exhausted"

        metadata: Dict[str, Any] = {
            "terminated": run_state.termination_ack,
            "tool_quality_runtime": build_runtime_payload(
                mode_name=request.mode_name,
                loop_exit_reason=run_state.loop_exit_reason,
                budget_exhausted=run_state.budget_exhausted,
                wall_time_exhausted=run_state.wall_time_exhausted,
                termination_ack=run_state.termination_ack,
                events=run_state.events,
            ),
        }
        metadata.update(pick_repo_metadata(request.task.resources))

        return AgentResult(
            task_id=request.task.task_id,
            final_artifact=final_artifact,
            metadata=metadata,
        )
