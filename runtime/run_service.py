from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent_architectures.constants import ARCHITECTURE_MINI_SWE_AGENT
from agent_architectures.base import ArchitectureRunRequest
from agent_architectures.factory import get_agent_architecture, resolve_agent_architecture
from agents.spec_loader import AgentSpecLoader
from benchmarks.registry import BenchmarkRegistry
from runtime.artifact_policy import apply_artifact_policy
from runtime.config_loader import apply_run_overrides
from runtime.config_models import RunConfig
from runtime.manifest_store import append_log, manifest_path, new_run_id, now_iso, write_manifest
from runtime.metrics import zero_eval_metrics
from runtime.prompt_messages import build_initial_user_message
from runtime.tool_quality import (
    build_run_summary,
    build_task_summary,
    format_run_tool_quality_log,
    format_task_tool_quality_log,
    serialize_tool_call_row,
)
from runtime.tools import ToolContext, ToolRegistry


@dataclass
class RunOutcome:
    """Structured metadata returned after generating predictions."""

    run_id: str
    benchmark_name: str
    split_name: str
    mode_name: str
    tasks_total: int
    valid_artifacts: int
    invalid_artifacts: int
    model_name_or_path: str
    predictions_path: Path
    manifest_path: Path
    run_log_path: Path
    manifest_payload: Dict[str, Any]


def _preview_text_for_log(text: str, limit: Optional[int] = 400) -> str:
    """Normalize and truncate multi-line text for compact run-log diagnostics."""

    compact = " ".join((text or "").split())
    if limit is None or len(compact) <= limit:
        return compact
    return compact[:limit] + "...[truncated]"


def _resolve_allowed_tools(
    *,
    mode_name: str,
    architecture_id: str,
    profile_explicit_tools: Optional[list[str]],
    profile_allowed_tools: Optional[set[str]],
    tool_registry: ToolRegistry,
) -> set[str]:
    """Resolve per-task tool allowlist from mode + profile/skill policy."""

    registry_tool_names = {
        schema.get("function", {}).get("name")
        for schema in tool_registry.schemas
        if isinstance(schema.get("function", {}).get("name"), str)
    }

    if mode_name == "patch_only":
        return {"submit"}
    # Profile alias: tools: [mini-swe-agent] => full registered toolset.
    if profile_explicit_tools is not None and ARCHITECTURE_MINI_SWE_AGENT in profile_explicit_tools:
        return registry_tool_names
    if architecture_id == ARCHITECTURE_MINI_SWE_AGENT:
        # mini-swe-agent defaults to full registry tools unless the profile
        # explicitly sets `tools`.
        if profile_explicit_tools is None:
            return registry_tool_names
        return set(profile_explicit_tools)
    if profile_allowed_tools is None:
        return registry_tool_names
    return set(profile_allowed_tools)


def _fallback_cannot_produce_output_artifact(
    *,
    loop_exit_reason: str,
    artifact_reason: str,
) -> str:
    """Build deterministic non-empty terminal artifact for no-submit exits."""

    reason = (loop_exit_reason or "unknown").strip() or "unknown"
    artifact_state = (artifact_reason or "unknown").strip() or "unknown"
    return (
        "CANNOT PRODUCE OUTPUT "
        f"no_submit_without_termination:{reason};artifact_reason:{artifact_state}"
    )


def execute_run(
    *,
    agent_path: str,
    config: RunConfig,
    benchmark: Optional[str] = None,
    split: Optional[str] = None,
    selector: Optional[int] = None,
    mode: Optional[str] = None,
    agent_architecture: Optional[str] = None,
    verbose: bool = False,
    full_log_previews: bool = False,
) -> RunOutcome:
    """Execute benchmark tasks and write predictions/logs/manifest artifacts."""

    effective_config = apply_run_overrides(
        config,
        benchmark=benchmark,
        split=split,
        selector=selector,
        mode=mode,
    )

    benchmark_name = effective_config.benchmark.name
    split_name = effective_config.benchmark.split
    mode_name = effective_config.runtime.mode

    base_dir = Path.cwd()
    spec_loader = AgentSpecLoader(base_dir)
    spec, prompt, allowed_tools = spec_loader.load(Path(agent_path), runtime_mode=mode_name)
    resolved_architecture = resolve_agent_architecture(
        cli_override=agent_architecture,
        run_override=effective_config.runtime.agent_architecture_override,
        profile_architecture=spec.agent_architecture,
    )
    architecture_runtime = None

    adapter_cls = BenchmarkRegistry().get_adapter(benchmark_name)
    adapter = adapter_cls.from_config(effective_config)
    tasks = adapter.load_tasks(split_name, effective_config.runtime.selector)

    model_name_or_path = spec.backend.get("model", spec.name)

    run_id = new_run_id()
    artifacts_dir = Path(effective_config.output.artifacts_dir)
    run_root = artifacts_dir / run_id
    out_path = run_root / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path = run_root / "tool_telemetry.jsonl"
    mini_trace_path = run_root / "mini_swe_agent_trace.jsonl"
    run_log_path = run_root / "run.log"
    append_log(
        run_log_path,
        (
            "Starting run:"
            f" run_id={run_id}"
            f" benchmark={benchmark_name}"
            f" split={split_name}"
            f" mode={mode_name}"
            f" tasks={len(tasks)}"
            f" model={model_name_or_path}"
            f" architecture={resolved_architecture}"
            f" agent_profile={agent_path}"
        ),
    )

    valid_artifacts = 0
    invalid_artifacts = 0
    tool_quality_weights = effective_config.runtime.tool_quality_weights.model_dump(mode="python")
    tool_quality_enabled = effective_config.runtime.tool_quality_enabled
    task_tool_summaries: list[Dict[str, Any]] = []
    termination_tool = "submit"
    if isinstance(spec.termination, dict):
        raw_termination_tool = spec.termination.get("tool")
        if isinstance(raw_termination_tool, str) and raw_termination_tool.strip():
            termination_tool = raw_termination_tool.strip()

    with out_path.open("w", encoding="utf-8") as out_file, telemetry_path.open(
        "w",
        encoding="utf-8",
    ) as telemetry_file:
        mini_trace_file = None
        if resolved_architecture == ARCHITECTURE_MINI_SWE_AGENT:
            mini_trace_file = mini_trace_path.open("w", encoding="utf-8")
        try:
            for task in tasks:
                # Rebuild tooling/runtime per task so workspace context stays isolated.
                workspace = adapter.workspace_context_for_task(task)
                submitted_artifact: Dict[str, str] = {}
                append_log(
                    run_log_path,
                    f"task={task.task_id} task_start workspace_root={workspace.workspace_root}",
                )
                append_log(
                    run_log_path,
                    (
                        f"task={task.task_id} workspace_context "
                        f"workspace_root={workspace.workspace_root} "
                        f"workspace_kind={workspace.workspace_kind} "
                        f"workspace_exists={workspace.workspace_exists} "
                        f"tools_ready={workspace.tools_ready} "
                        f"repo={workspace.repo or ''} "
                        f"reason={_preview_text_for_log(workspace.reason or '', limit=None)}"
                    ),
                )

                def _submit_callback(artifact: str):
                    submitted_artifact["artifact"] = artifact

                tool_ctx = ToolContext(
                    workspace_root=workspace.workspace_root,
                    submit_callback=_submit_callback,
                    expected_output_type=task.expected_output_type,
                    patch_submit_policy=effective_config.runtime.patch_submit_policy,
                    max_invalid_submit_attempts=effective_config.runtime.max_invalid_submit_attempts,
                )
                tool_registry = ToolRegistry(tool_ctx)

                def _api_log(line: str, task_id: str = task.task_id) -> None:
                    event_name = line.split(" ", 1)[0] if line else ""
                    if event_name == "api_error":
                        level = "ERROR"
                    elif event_name in {"api_retry", "api_tool_args_parse_error"}:
                        level = "WARNING"
                    else:
                        level = "INFO"
                    append_log(
                        run_log_path,
                        f"task={task_id} {line}",
                        level=level,
                        source="model_backend.py",
                    )

                if mode_name == "tools_enabled" and not workspace.tools_ready:
                    raise ValueError(
                        "tools_enabled task workspace is not tool-ready: "
                        f"task_id={task.task_id} "
                        f"workspace_root={workspace.workspace_root} "
                        f"workspace_kind={workspace.workspace_kind} "
                        f"reason={workspace.reason or 'unspecified'} "
                        "Remediation: configure a local benchmark workspace with "
                        "benchmark.data_source=local and benchmark.data_root containing repo checkouts "
                        "under <data_root>/<repo>."
                    )
                allowed = _resolve_allowed_tools(
                    mode_name=mode_name,
                    architecture_id=resolved_architecture,
                    profile_explicit_tools=spec.tools,
                    profile_allowed_tools=allowed_tools,
                    tool_registry=tool_registry,
                )
                if architecture_runtime is None:
                    architecture_runtime = get_agent_architecture(resolved_architecture)

                initial_user_message = build_initial_user_message(task, workspace, mode_name)
                result = architecture_runtime.run_task(
                    ArchitectureRunRequest(
                        task=task,
                        system_prompt=prompt,
                        initial_user_message=initial_user_message,
                        mode_name=mode_name,
                        backend_config=spec.backend,
                        decoding_defaults=spec.decoding_defaults,
                        tool_registry=tool_registry,
                        allowed_tools=allowed,
                        max_tool_calls=effective_config.runtime.max_tool_calls,
                        max_wall_time_s=effective_config.runtime.max_wall_time_s,
                        termination_tool=termination_tool,
                        full_log_previews=full_log_previews,
                        api_log=_api_log,
                        architecture_config=spec.agent_architecture_config,
                    )
                )
                # Explicit submit tool payload takes precedence over assistant free text.
                if submitted_artifact.get("artifact"):
                    result.final_artifact = submitted_artifact["artifact"]
                if result.metadata is None:
                    result.metadata = {}
                if isinstance(result.metadata, dict):
                    result.metadata["invalid_submit_attempts"] = tool_ctx.invalid_submit_attempts
                    if isinstance(tool_ctx.last_invalid_submit_reason, str):
                        result.metadata["last_invalid_submit_reason"] = tool_ctx.last_invalid_submit_reason

                terminated = bool(result.metadata.get("terminated")) if isinstance(result.metadata, dict) else False
                runtime_tool_payload = (
                    result.metadata.get("tool_quality_runtime")
                    if isinstance(result.metadata, dict)
                    and isinstance(result.metadata.get("tool_quality_runtime"), dict)
                    else None
                )
                loop_exit_reason = (
                    runtime_tool_payload.get("loop_exit_reason")
                    if isinstance(runtime_tool_payload, dict)
                    and isinstance(runtime_tool_payload.get("loop_exit_reason"), str)
                    else "unknown"
                )
                # Harden tools-mode termination semantics: if the model never submitted and
                # patch artifact is invalid/empty, emit an explicit non-empty failure sentinel.
                if mode_name == "tools_enabled" and task.expected_output_type == "patch" and not terminated:
                    pre_fallback_policy = apply_artifact_policy(result.final_artifact, "patch")
                    if (not pre_fallback_policy.valid) and (
                        pre_fallback_policy.reason != "cannot_produce_output"
                    ):
                        fallback_artifact = _fallback_cannot_produce_output_artifact(
                            loop_exit_reason=loop_exit_reason,
                            artifact_reason=pre_fallback_policy.reason,
                        )
                        result.final_artifact = fallback_artifact
                        if isinstance(result.metadata, dict):
                            result.metadata["no_submit_fallback_applied"] = True
                            result.metadata["no_submit_fallback_reason"] = (
                                f"no_submit_without_termination:{loop_exit_reason}"
                            )
                            result.metadata["no_submit_original_artifact_reason"] = pre_fallback_policy.reason
                        append_log(
                            run_log_path,
                            (
                                f"task={task.task_id} applied_no_submit_fallback=true "
                                f"loop_exit_reason={loop_exit_reason} "
                                f"original_artifact_reason={pre_fallback_policy.reason}"
                            ),
                            level="WARNING",
                        )

                policy_result = apply_artifact_policy(result.final_artifact, task.expected_output_type)
                artifact = result.final_artifact
                if task.expected_output_type == "patch":
                    # Keep raw invalid patch output for diagnostics, but persist normalized valid patches.
                    if policy_result.valid:
                        artifact = policy_result.artifact
                else:
                    artifact = policy_result.artifact
                append_log(
                    run_log_path,
                    (
                        f"task={task.task_id} artifact_bytes={len(artifact.encode('utf-8', errors='ignore'))} "
                        f"artifact_preview={_preview_text_for_log(artifact, limit=None if full_log_previews else 400)}"
                    ),
                    level="INFO" if policy_result.valid else "WARNING",
                )

                if policy_result.valid:
                    valid_artifacts += 1
                else:
                    invalid_artifacts += 1

                record = adapter.to_prediction_record(
                    task=task,
                    artifact=artifact,
                    model_name_or_path=model_name_or_path,
                    model_name=spec.name,
                    metadata=result.metadata,
                )
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

                per_task_line = (
                    f"task={task.task_id} terminated={terminated} "
                    f"output_type={task.expected_output_type} "
                    f"artifact_valid={policy_result.valid} "
                    f"artifact_reason={policy_result.reason} "
                    f"invalid_submit_attempts={tool_ctx.invalid_submit_attempts} "
                    f"last_invalid_submit_reason={tool_ctx.last_invalid_submit_reason or 'none'}"
                )
                append_log(run_log_path, per_task_line)
                if verbose:
                    print(per_task_line)

                if isinstance(runtime_tool_payload, dict):
                    raw_events = runtime_tool_payload.get("events")
                    if isinstance(raw_events, list):
                        for event in raw_events:
                            if not isinstance(event, dict):
                                continue
                            row = serialize_tool_call_row(
                                run_id=run_id,
                                task_id=task.task_id,
                                mode=mode_name,
                                event=event,
                            )
                            telemetry_file.write(json.dumps(row) + "\n")
                if (
                    mini_trace_file is not None
                    and isinstance(result.metadata, dict)
                    and isinstance(result.metadata.get("mini_turn_trace"), list)
                ):
                    for turn_text in result.metadata.get("mini_turn_trace", []):
                        if not isinstance(turn_text, str) or not turn_text.strip():
                            continue
                        mini_trace_file.write(
                            json.dumps(
                                {
                                    "task_id": task.task_id,
                                    "turn": turn_text,
                                }
                            )
                            + "\n"
                        )
                    mini_trace_file.flush()

                task_summary = build_task_summary(
                    run_id=run_id,
                    task_id=task.task_id,
                    mode=mode_name,
                    runtime_payload=runtime_tool_payload,
                    weights=tool_quality_weights,
                    enabled=tool_quality_enabled,
                )
                task_tool_summaries.append(task_summary)
                telemetry_file.write(json.dumps(task_summary) + "\n")
                telemetry_file.flush()
                append_log(run_log_path, format_task_tool_quality_log(task_summary))
        finally:
            if mini_trace_file is not None:
                mini_trace_file.close()

    run_tool_quality_summary = build_run_summary(
        telemetry_path=telemetry_path,
        task_summaries=task_tool_summaries,
        weights=tool_quality_weights,
        enabled=tool_quality_enabled,
    )
    append_log(run_log_path, format_run_tool_quality_log(run_tool_quality_summary))

    created_at = now_iso()
    manifest_payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": created_at,
        "agent_profile": agent_path,
        "agent_architecture": resolved_architecture,
        "model_name": spec.name,
        "model_name_or_path": model_name_or_path,
        "benchmark_name": benchmark_name,
        "dataset_name": effective_config.benchmark.dataset_name,
        "split": split_name,
        "mode": mode_name,
        "predictions_path": str(out_path.resolve()),
        "tool_quality": run_tool_quality_summary,
        "evaluation": {
            "status": "not_run",
            "returncode": None,
            "report_path": None,
            "harness_log_root": None,
            "metrics": zero_eval_metrics(),
        },
        "config_snapshot": effective_config.model_dump(mode="python"),
    }
    out_manifest_path = manifest_path(run_root)
    write_manifest(out_manifest_path, manifest_payload)

    run_summary_line = (
        f"Run summary: run_id={run_id} tasks={len(tasks)} "
        f"valid_artifacts={valid_artifacts} invalid_artifacts={invalid_artifacts}"
    )
    append_log(run_log_path, run_summary_line, level="INFO" if invalid_artifacts == 0 else "WARNING")
    append_log(run_log_path, f"Predictions written to {out_path}")
    if resolved_architecture == ARCHITECTURE_MINI_SWE_AGENT:
        append_log(run_log_path, f"Mini trace written to {mini_trace_path}")
    append_log(run_log_path, f"Manifest written to {out_manifest_path}")
    append_log(run_log_path, f"Run log written to {run_log_path}")

    return RunOutcome(
        run_id=run_id,
        benchmark_name=benchmark_name,
        split_name=split_name,
        mode_name=mode_name,
        tasks_total=len(tasks),
        valid_artifacts=valid_artifacts,
        invalid_artifacts=invalid_artifacts,
        model_name_or_path=model_name_or_path,
        predictions_path=out_path,
        manifest_path=out_manifest_path,
        run_log_path=run_log_path,
        manifest_payload=manifest_payload,
    )
