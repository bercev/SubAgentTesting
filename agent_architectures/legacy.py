from __future__ import annotations

from runtime.agent_runtime import AgentRuntime
from runtime.backend_factory import build_backend

from agent_architectures.base import ArchitectureRunRequest, AgentArchitecture, filter_tool_schemas
from agent_architectures.constants import ARCHITECTURE_NONE


class LegacyArchitecture(AgentArchitecture):
    """Existing runner architecture used when agent_architecture=none."""

    architecture_id = ARCHITECTURE_NONE

    def run_task(self, request: ArchitectureRunRequest):
        """Execute one task using the existing model backend + tool runtime loop."""

        tool_schemas = None if request.mode_name == "patch_only" else filter_tool_schemas(
            request.tool_registry,
            request.allowed_tools,
        )
        backend = build_backend(
            request.backend_config,
            event_logger=request.api_log,
            full_log_previews=request.full_log_previews,
        )
        runtime = AgentRuntime(
            backend=backend,
            tool_registry=request.tool_registry,
            allowed_tools=request.allowed_tools,
            max_tool_calls=request.max_tool_calls,
            max_wall_time_s=request.max_wall_time_s,
            termination_tool=request.termination_tool,
            mode_name=request.mode_name,
        )
        return runtime.run(
            task=request.task,
            system_prompt=request.system_prompt,
            initial_user_message=request.initial_user_message,
            tool_schemas=tool_schemas,
            decoding_defaults=dict(request.decoding_defaults or {}),
        )
