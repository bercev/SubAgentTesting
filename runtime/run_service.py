from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agents.spec_loader import AgentSpecLoader
from benchmarks.registry import BenchmarkRegistry
from runtime.agent_runtime import AgentRuntime
from runtime.artifact_policy import apply_artifact_policy
from runtime.backend_factory import build_backend
from runtime.config_loader import apply_run_overrides
from runtime.config_models import RunConfig
from runtime.manifest_store import append_log, manifest_path, new_run_id, now_iso, write_manifest
from runtime.metrics import zero_eval_metrics
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


def _filter_tool_schemas(registry: ToolRegistry, allowed: set[str]) -> list[dict[str, Any]]:
    """Keep only tool schemas explicitly allowed by the active policy."""

    schemas = []
    for schema in registry.schemas:
        name = schema.get("function", {}).get("name")
        if name in allowed:
            schemas.append(schema)
    return schemas


def execute_run(
    *,
    agent_path: str,
    config: RunConfig,
    benchmark: Optional[str] = None,
    split: Optional[str] = None,
    selector: Optional[int] = None,
    mode: Optional[str] = None,
    verbose: bool = False,
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
    spec, prompt, allowed_tools = spec_loader.load(Path(agent_path))

    adapter_cls = BenchmarkRegistry().get_adapter(benchmark_name)
    adapter = adapter_cls.from_config(effective_config)
    tasks = adapter.load_tasks(split_name, effective_config.runtime.selector)

    model_name_or_path = spec.backend.get("model", spec.name)

    run_id = new_run_id()
    artifacts_dir = Path(effective_config.output.artifacts_dir)
    run_root = artifacts_dir / run_id
    out_path = run_root / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            f" agent_profile={agent_path}"
        ),
    )

    valid_artifacts = 0
    invalid_artifacts = 0

    with out_path.open("w", encoding="utf-8") as out_file:
        for task in tasks:
            # Rebuild tooling/runtime per task so workspace context stays isolated.
            workspace_root = adapter.workspace_root_for_task(task)
            submitted_artifact: Dict[str, str] = {}

            def _submit_callback(artifact: str):
                submitted_artifact["artifact"] = artifact

            tool_ctx = ToolContext(workspace_root=workspace_root, submit_callback=_submit_callback)
            tool_registry = ToolRegistry(tool_ctx)

            backend = build_backend(spec.backend)
            if mode_name == "patch_only":
                allowed = {"submit"}
                tool_schemas = None
            else:
                allowed = allowed_tools or {
                    schema.get("function", {}).get("name") for schema in tool_registry.schemas
                }
                tool_schemas = _filter_tool_schemas(tool_registry, allowed)

            runtime = AgentRuntime(
                backend=backend,
                tool_registry=tool_registry,
                allowed_tools=allowed,
                max_tool_calls=effective_config.runtime.max_tool_calls,
                max_wall_time_s=effective_config.runtime.max_wall_time_s,
            )

            result = runtime.run(
                task=task,
                prompt=prompt,
                tool_schemas=tool_schemas,
                decoding_defaults=spec.decoding_defaults,
            )
            # Explicit submit tool payload takes precedence over assistant free text.
            if submitted_artifact.get("artifact"):
                result.final_artifact = submitted_artifact["artifact"]

            policy_result = apply_artifact_policy(result.final_artifact, task.expected_output_type)
            artifact = result.final_artifact
            # Patch output remains pass-through; non-patch outputs use normalized artifact.
            if task.expected_output_type != "patch":
                artifact = policy_result.artifact

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

            terminated = result.metadata.get("terminated") if result.metadata else False
            per_task_line = (
                f"task={task.task_id} terminated={terminated} "
                f"output_type={task.expected_output_type} "
                f"artifact_valid={policy_result.valid} "
                f"artifact_reason={policy_result.reason}"
            )
            append_log(run_log_path, per_task_line)
            if verbose:
                print(per_task_line)

    created_at = now_iso()
    manifest_payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": created_at,
        "agent_profile": agent_path,
        "model_name": spec.name,
        "model_name_or_path": model_name_or_path,
        "benchmark_name": benchmark_name,
        "dataset_name": effective_config.benchmark.dataset_name,
        "split": split_name,
        "mode": mode_name,
        "predictions_path": str(out_path.resolve()),
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
    append_log(run_log_path, run_summary_line)
    append_log(run_log_path, f"Predictions written to {out_path}")
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
