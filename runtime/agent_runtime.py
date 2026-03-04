import json
import time
from typing import Any, Dict, List, Optional, Set

from runtime.model_backend import GenerationResult, ModelBackend
from runtime.schemas import AgentResult, BenchmarkTask
from runtime.tools import ToolRegistry

NO_TOOL_CALL_REPAIR_INSTRUCTION = (
    "TOOLS MODE RECOVERY: Your previous turn returned zero tool calls. "
    "You must call at least one tool now. "
    "Do not answer with plain text. "
    "If you are done, call submit(final_artifact) with either a valid unified diff "
    "or CANNOT PRODUCE OUTPUT {reason}."
)


class AgentRuntime:
    """Task execution loop that drives model generation and tool calls."""

    MAX_TOOL_MESSAGE_CHARS = 12000

    def __init__(
        self,
        backend: ModelBackend,
        tool_registry: ToolRegistry,
        allowed_tools: Optional[Set[str]] = None,
        max_tool_calls: int = 20,
        max_wall_time_s: int = 600,
        max_completion_tokens: int = 512,
        termination_tool: str = "submit",
        mode_name: str = "patch_only",
    ) -> None:
        """Capture runtime dependencies and loop limits for one execution strategy."""

        self.backend = backend
        self.tool_registry = tool_registry
        self.allowed_tools = allowed_tools
        self.max_tool_calls = max_tool_calls
        self.max_wall_time_s = max_wall_time_s
        self.max_completion_tokens = max_completion_tokens
        self.termination_tool = termination_tool
        self.mode_name = mode_name

    @staticmethod
    def _json_size_bytes(payload: Any) -> int:
        """Approximate payload size as UTF-8 JSON bytes."""

        try:
            serialized = json.dumps(payload, default=str, ensure_ascii=False)
        except Exception:
            serialized = str(payload)
        return len(serialized.encode("utf-8", errors="ignore"))

    @classmethod
    def _truncate_text(cls, text: str, limit: int) -> str:
        """Truncate large tool payloads before appending them to model context."""

        if len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"

    @classmethod
    def _tool_message_content(cls, payload: Any) -> str:
        """Serialize tool results as JSON (with fallback) for consistent model parsing."""

        try:
            serialized = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            serialized = str(payload)
        return cls._truncate_text(serialized, cls.MAX_TOOL_MESSAGE_CHARS)

    @staticmethod
    def _resolve_completion_cap(
        *,
        decoding_defaults: Optional[Dict[str, Any]],
        fallback_max_completion_tokens: int,
    ) -> int | None:
        if isinstance(decoding_defaults, dict):
            raw = decoding_defaults.get("max_tokens")
            if not isinstance(raw, bool) and isinstance(raw, (int, float)):
                parsed = int(raw)
                if parsed > 0:
                    return parsed
        if fallback_max_completion_tokens > 0:
            return int(fallback_max_completion_tokens)
        return None

    @staticmethod
    def _is_completion_cap_hit(result: GenerationResult, completion_cap: int | None) -> bool:
        if isinstance(result.finish_reason, str) and result.finish_reason == "length":
            return True
        if (
            completion_cap is not None
            and isinstance(result.completion_tokens, int)
            and result.completion_tokens >= completion_cap
        ):
            return True
        return False

    def run(
        self,
        task: BenchmarkTask,
        system_prompt: str,
        initial_user_message: str,
        tool_schemas: Optional[List[Dict[str, Any]]],
        decoding_defaults: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Run model/tool loop until submission, timeout, or budget exhaustion."""

        start = time.monotonic()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]
        terminated = False
        final_artifact = ""
        tool_calls_made = 0
        turn_index = 0
        loop_exit_reason = "unknown"
        budget_exhausted = False
        wall_time_exhausted = False
        termination_ack = False
        tool_call_events: List[Dict[str, Any]] = []
        invalid_submit_attempts = 0
        last_invalid_submit_reason: Optional[str] = None
        no_tool_call_repair_attempted = False
        no_tool_call_failure_after_repair = False
        no_tool_call_cap_hit = False
        no_tool_call_terminal_artifact_emitted = False
        completion_cap = self._resolve_completion_cap(
            decoding_defaults=decoding_defaults,
            fallback_max_completion_tokens=self.max_completion_tokens,
        )

        while True:
            halt_invalid_submission = False
            # Enforce wall-clock and tool-call budgets before each model turn.
            if time.monotonic() - start > self.max_wall_time_s:
                wall_time_exhausted = True
                loop_exit_reason = "wall_time_exhausted"
                break
            if tool_calls_made >= self.max_tool_calls:
                budget_exhausted = True
                loop_exit_reason = "tool_budget_exhausted"
                break

            tools = tool_schemas if tool_schemas else None
            result: GenerationResult = self.backend.generate(messages, tools=tools, decoding=decoding_defaults)
            assistant_msg = {"role": "assistant", "content": result.assistant_text}
            tool_call_ids: List[str] = []
            if result.tool_calls:
                tool_calls_payload = []
                for idx, tc in enumerate(result.tool_calls):
                    call_id = f"call_{tool_calls_made}_{idx}"
                    tool_call_ids.append(call_id)
                    tool_calls_payload.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                assistant_msg["tool_calls"] = tool_calls_payload
            messages.append(assistant_msg)

            if not result.tool_calls:
                if self.mode_name == "tools_enabled":
                    no_tool_call_cap_hit = no_tool_call_cap_hit or self._is_completion_cap_hit(
                        result, completion_cap
                    )
                    if not no_tool_call_repair_attempted:
                        no_tool_call_repair_attempted = True
                        messages.append(
                            {
                                "role": "user",
                                "content": NO_TOOL_CALL_REPAIR_INSTRUCTION,
                            }
                        )
                        turn_index += 1
                        continue
                    # Tools-enabled mode must terminate via explicit submit(...).
                    terminal_reason = "completion_cap_reached" if no_tool_call_cap_hit else "after_repair"
                    final_artifact = (
                        "CANNOT PRODUCE OUTPUT "
                        f"no_tool_calls_without_submit:{terminal_reason}"
                    )
                    terminated = False
                    loop_exit_reason = "no_tool_calls_without_submit"
                    no_tool_call_failure_after_repair = True
                    no_tool_call_terminal_artifact_emitted = True
                else:
                    # Patch-only mode often terminates by returning final artifact text directly.
                    final_artifact = result.assistant_text
                    terminated = True
                    loop_exit_reason = "no_tool_calls"
                break

            for idx, tc in enumerate(result.tool_calls):
                tool_calls_made += 1
                event: Dict[str, Any] = {
                    "turn_index": turn_index,
                    "call_index": idx,
                    "tool_name": tc.name,
                    "is_termination_tool": tc.name == self.termination_tool,
                    "allowed": False,
                    "executed": False,
                    "success": False,
                    "error_code": "tool_error",
                    "args_size_bytes": self._json_size_bytes(tc.arguments),
                    "result_size_bytes": 0,
                    "latency_ms": 0,
                    "return_code": None,
                }
                if self.allowed_tools is not None and tc.name not in self.allowed_tools:
                    event.update(
                        {
                            "allowed": False,
                            "executed": False,
                            "success": False,
                            "error_code": "not_allowed",
                        }
                    )
                    tool_call_events.append(event)
                    messages.append(
                        {
                            "role": "tool",
                            "name": tc.name,
                            "tool_call_id": tool_call_ids[idx] if idx < len(tool_call_ids) else "unknown",
                            "content": f"Tool {tc.name} not allowed",
                        }
                    )
                    continue

                event["allowed"] = True
                tool_started = time.monotonic()
                execution_exception: Optional[Exception] = None
                try:
                    tool_result = self.tool_registry.execute(tc.name, tc.arguments)
                except Exception as exc:
                    execution_exception = exc
                    tool_result = {
                        "error": f"tool execution exception: {exc.__class__.__name__}: {exc}",
                    }

                latency_ms = int(max(0.0, (time.monotonic() - tool_started) * 1000.0))
                error_code = "none"
                success = True
                return_code: Optional[int] = None
                if execution_exception is not None:
                    error_code = "execution_exception"
                    success = False
                elif isinstance(tool_result, dict):
                    invalid_attempts = tool_result.get("invalid_submit_attempts")
                    if isinstance(invalid_attempts, int):
                        invalid_submit_attempts = max(invalid_submit_attempts, invalid_attempts)
                    invalid_reason = tool_result.get("invalid_submission_reason")
                    if isinstance(invalid_reason, str) and invalid_reason:
                        last_invalid_submit_reason = invalid_reason
                    if "error" in tool_result:
                        error_code = "tool_error"
                        success = False
                    elif isinstance(tool_result.get("returncode"), int) and tool_result.get("returncode") != 0:
                        error_code = "nonzero_returncode"
                        success = False
                    elif tool_result.get("success") is False:
                        error_code = "tool_error"
                        success = False
                    if isinstance(tool_result.get("returncode"), int):
                        return_code = tool_result.get("returncode")
                event.update(
                    {
                        "executed": True,
                        "success": success,
                        "error_code": error_code,
                        "result_size_bytes": self._json_size_bytes(tool_result),
                        "latency_ms": latency_ms,
                        "return_code": return_code,
                    }
                )
                tool_call_events.append(event)
                messages.append(
                    {
                        "role": "tool",
                        "name": tc.name,
                        "tool_call_id": tool_call_ids[idx] if idx < len(tool_call_ids) else "unknown",
                        "content": self._tool_message_content(tool_result),
                    }
                )
                if execution_exception is not None:
                    continue

                if tc.name == self.termination_tool and isinstance(tool_result, dict):
                    invalid_terminal_reason = tool_result.get("invalid_submission_terminal_reason")
                    if isinstance(invalid_terminal_reason, str) and invalid_terminal_reason:
                        loop_exit_reason = invalid_terminal_reason
                        halt_invalid_submission = True
                        break
                    if tool_result.get("submitted"):
                        # The configured termination tool is the explicit stop signal.
                        final_artifact = tc.arguments.get("final_artifact", "")
                        terminated = True
                        termination_ack = True
                        loop_exit_reason = "submitted"
                        break

            if terminated or halt_invalid_submission:
                break

            turn_index += 1

        if not terminated:
            # Preserve last artifact state even when loop exits on limits.
            final_artifact = final_artifact or ""

        metadata = {
            "terminated": terminated,
            "invalid_submit_attempts": invalid_submit_attempts,
            "last_invalid_submit_reason": last_invalid_submit_reason,
            "tool_quality_runtime": {
                "mode": self.mode_name,
                "loop_exit_reason": loop_exit_reason,
                "budget_exhausted": budget_exhausted,
                "wall_time_exhausted": wall_time_exhausted,
                "termination_ack": termination_ack,
                "no_tool_call_repair_attempted": no_tool_call_repair_attempted,
                "no_tool_call_failure_after_repair": no_tool_call_failure_after_repair,
                "no_tool_call_cap_hit": no_tool_call_cap_hit,
                "no_tool_call_terminal_artifact_emitted": no_tool_call_terminal_artifact_emitted,
                "events": tool_call_events,
            },
        }
        if task.resources and "repo" in task.resources:
            metadata["repo"] = task.resources["repo"]
        return AgentResult(task_id=task.task_id, final_artifact=final_artifact, metadata=metadata)
