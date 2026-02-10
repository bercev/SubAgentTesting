import time
from typing import Any, Dict, List, Optional, Set

from runtime.model_backend import GenerationResult, ModelBackend
from runtime.schemas import AgentResult, BenchmarkTask
from runtime.tools import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        backend: ModelBackend,
        tool_registry: ToolRegistry,
        allowed_tools: Optional[Set[str]] = None,
        max_tool_calls: int = 20,
        max_wall_time_s: int = 600,
        max_completion_tokens: int = 512,
    ) -> None:
        self.backend = backend
        self.tool_registry = tool_registry
        self.allowed_tools = allowed_tools
        self.max_tool_calls = max_tool_calls
        self.max_wall_time_s = max_wall_time_s
        self.max_completion_tokens = max_completion_tokens

    def run(
        self,
        task: BenchmarkTask,
        prompt: str,
        tool_schemas: Optional[List[Dict[str, Any]]],
        decoding_defaults: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        start = time.monotonic()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.instruction},
        ]
        terminated = False
        final_artifact = ""
        tool_calls_made = 0

        while True:
            if time.monotonic() - start > self.max_wall_time_s:
                break
            if tool_calls_made >= self.max_tool_calls:
                break

            tools = tool_schemas if tool_schemas else None
            result: GenerationResult = self.backend.generate(messages, tools=tools, decoding=decoding_defaults)
            assistant_msg = {"role": "assistant", "content": result.assistant_text}
            if result.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in result.tool_calls
                ]
            messages.append(assistant_msg)

            if not result.tool_calls:
                # If no tool calls, treat the assistant text as the final artifact.
                final_artifact = result.assistant_text
                terminated = True
                break

            for tc in result.tool_calls:
                tool_calls_made += 1
                if self.allowed_tools and tc.name not in self.allowed_tools:
                    messages.append(
                        {
                            "role": "tool",
                            "name": tc.name,
                            "content": f"Tool {tc.name} not allowed",
                        }
                    )
                    continue

                tool_result = self.tool_registry.execute(tc.name, tc.arguments)
                messages.append({"role": "tool", "name": tc.name, "content": str(tool_result)})

                if tc.name == "submit" and tool_result.get("submitted"):
                    final_artifact = tc.arguments.get("final_artifact", "")
                    terminated = True
                    break

            if terminated:
                break

        if not terminated:
            # forced termination
            final_artifact = final_artifact or ""

        metadata = {"terminated": terminated}
        if task.resources and "repo" in task.resources:
            metadata["repo"] = task.resources["repo"]
        return AgentResult(task_id=task.task_id, final_artifact=final_artifact, metadata=metadata)
