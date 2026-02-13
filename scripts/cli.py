import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml
from dotenv import load_dotenv
from rich import print

from agents.spec_loader import AgentSpecLoader
from benchmarks.registry import BenchmarkRegistry
from runtime.agent_runtime import AgentRuntime
from runtime.artifact_policy import apply_artifact_policy
from runtime.model_backend import OpenRouterBackend
from runtime.tools import ToolContext, ToolRegistry

app = typer.Typer(add_completion=False)
load_dotenv()

RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
EVAL_COUNT_KEYS = (
    "total_instances",
    "submitted_instances",
    "completed_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "error_instances",
)


def _default_run_config() -> Dict[str, Any]:
    return {
        "benchmark": {
            "name": "swebench_verified",
            "dataset_name": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "data_source": "hf",
            "data_root": None,
        },
        "evaluation": {
            "enabled": True,
            "harness_cmd": "python -m swebench.harness.run_evaluation",
            "eval_root": "./external/SWE-bench",
            "workdir": ".",
            "report_dir": "reports",
        },
        "runtime": {
            "mode": "patch_only",
            "selector": 5,
            "max_tool_calls": 20,
            "max_wall_time_s": 600,
        },
        "output": {
            "artifacts_dir": "artifacts",
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _normalize_run_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_run_config()
    if any(key in raw_config for key in ("benchmark", "evaluation", "runtime", "output")):
        return _deep_merge(defaults, raw_config)

    # Legacy flat config compatibility
    legacy_as_nested = {
        "benchmark": {
            "name": raw_config.get("benchmark"),
            "dataset_name": raw_config.get("dataset_name"),
            "split": raw_config.get("default_split") or raw_config.get("split"),
            "data_source": raw_config.get("data_source"),
            "data_root": raw_config.get("data_root"),
        },
        "evaluation": {
            "harness_cmd": raw_config.get("harness_cmd"),
            "eval_root": raw_config.get("eval_root"),
            "workdir": raw_config.get("workdir"),
            "report_dir": raw_config.get("report_dir"),
        },
        "runtime": {
            "mode": raw_config.get("mode"),
            "selector": raw_config.get("selector"),
            "max_tool_calls": raw_config.get("max_tool_calls"),
            "max_wall_time_s": raw_config.get("max_wall_time_s"),
        },
        "output": {
            "artifacts_dir": raw_config.get("artifacts_dir"),
        },
    }
    return _deep_merge(defaults, legacy_as_nested)


def _load_run_config(run_config_path: Path) -> Dict[str, Any]:
    if not run_config_path.exists():
        raise FileNotFoundError(
            "Missing run config: "
            f"{run_config_path}. Create one from `configs/runs/example.swebench_verified.hf.yaml`."
        )
    with run_config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return _normalize_run_config(raw_config)


def _normalize_mode(mode_value: str) -> str:
    aliases = {
        "A": "patch_only",
        "PATCH_ONLY": "patch_only",
        "PATCH-ONLY": "patch_only",
        "SUBMIT_ONLY": "patch_only",
        "B": "tools_enabled",
        "TOOLS_ENABLED": "tools_enabled",
        "TOOLS-ENABLED": "tools_enabled",
        "TOOLS": "tools_enabled",
    }
    normalized = aliases.get(str(mode_value).strip().upper())
    if normalized:
        return normalized
    raise ValueError(
        f"Unsupported mode '{mode_value}'. Use one of: patch_only, tools_enabled (legacy: A, B)."
    )


def _build_backend(backend_config: Dict[str, Any]) -> OpenRouterBackend:
    backend_type = backend_config.get("type", "openrouter")
    if backend_type == "openrouter":
        model_id = backend_config.get("model")
        if not model_id:
            raise ValueError("Missing model id in agent spec backend.model")
        return OpenRouterBackend(
            model=model_id,
            base_url=backend_config.get("base_url", "https://openrouter.ai/api/v1"),
            max_retries=backend_config.get("max_retries", 8),
            initial_backoff_s=backend_config.get("initial_backoff_s", 1.0),
            max_backoff_s=backend_config.get("max_backoff_s", 10.0),
        )
    raise ValueError(f"Unsupported backend type: {backend_type}")


def _filter_tool_schemas(registry: ToolRegistry, allowed: set) -> List[dict]:
    schemas = []
    for schema in registry.schemas:
        name = schema.get("function", {}).get("name")
        if name in allowed:
            schemas.append(schema)
    return schemas


def _build_adapter_from_config(config: Dict[str, Any], benchmark_name: str):
    benchmark_config = config["benchmark"]
    evaluation_config = config["evaluation"]
    adapter_cls = BenchmarkRegistry().get_adapter(benchmark_name)
    return adapter_cls(
        data_source=benchmark_config.get("data_source", "hf"),
        data_root=benchmark_config.get("data_root"),
        dataset_name=benchmark_config.get("dataset_name", "SWE-bench/SWE-bench_Verified"),
        eval_root=evaluation_config.get("eval_root"),
        harness_cmd=evaluation_config.get("harness_cmd"),
        workdir=evaluation_config.get("workdir"),
        report_dir=evaluation_config.get("report_dir"),
    )


def _serialize_prediction_record(
    adapter: Any,
    task: Any,
    artifact: str,
    model_name_or_path: str,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    serializer = getattr(adapter, "to_prediction_record", None)
    if not callable(serializer):
        raise AttributeError(
            f"Adapter {type(adapter).__name__} must implement to_prediction_record(...)"
        )
    return serializer(
        task=task,
        artifact=artifact,
        model_name_or_path=model_name_or_path,
        model_name=model_name,
        metadata=metadata or {},
    )


def _new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _is_valid_run_id(value: str) -> bool:
    return bool(RUN_ID_PATTERN.match(value))


def _derive_run_id_from_predictions(predictions_path: Path, artifacts_dir: Path) -> str:
    abs_predictions = predictions_path.resolve()
    abs_artifacts = artifacts_dir.resolve()

    try:
        rel = abs_predictions.relative_to(abs_artifacts)
    except Exception:
        raise ValueError(
            "Predictions path must be under "
            f"{abs_artifacts}/<run_id>/predictions.jsonl; got {abs_predictions}"
        )

    if len(rel.parts) != 2 or rel.parts[1] != "predictions.jsonl":
        raise ValueError(
            "Predictions path must match artifacts/<run_id>/predictions.jsonl; "
            f"got {abs_predictions}"
        )

    run_id = rel.parts[0]
    if not _is_valid_run_id(run_id):
        raise ValueError(f"Invalid run_id in predictions path: {run_id}")
    return run_id


def _manifest_path(run_root: Path) -> Path:
    return run_root / "manifest.json"


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{_now_iso()}] {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value.strip()))
        except ValueError:
            return 0
    return 0


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _zero_eval_metrics() -> Dict[str, Any]:
    return {
        "total_instances": 0,
        "submitted_instances": 0,
        "completed_instances": 0,
        "resolved_instances": 0,
        "unresolved_instances": 0,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "accuracy_resolved_submitted": 0.0,
        "accuracy_resolved_completed": 0.0,
        "completion_rate_submitted": 0.0,
    }


def _read_eval_metrics(report_path: Optional[Path]) -> Tuple[Dict[str, Any], Optional[str]]:
    metrics = _zero_eval_metrics()
    if report_path is None or not report_path.exists():
        return metrics, "report_not_found"

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return metrics, "report_parse_failed"

    if not isinstance(payload, dict):
        return metrics, "report_invalid_shape"

    for key in EVAL_COUNT_KEYS:
        metrics[key] = _to_int(payload.get(key))

    resolved = metrics["resolved_instances"]
    submitted = metrics["submitted_instances"]
    completed = metrics["completed_instances"]
    metrics["accuracy_resolved_submitted"] = _safe_div(resolved, submitted)
    metrics["accuracy_resolved_completed"] = _safe_div(resolved, completed)
    metrics["completion_rate_submitted"] = _safe_div(completed, submitted)
    return metrics, None


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def _read_prediction_identity(predictions_path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not predictions_path.exists():
        return None, None
    try:
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                model_name = rec.get("model_name") if isinstance(rec, dict) else None
                model_name_or_path = rec.get("model_name_or_path") if isinstance(rec, dict) else None
                return (
                    model_name if isinstance(model_name, str) else None,
                    model_name_or_path if isinstance(model_name_or_path, str) else None,
                )
    except Exception:
        return None, None
    return None, None


@app.command()
def list():
    """List available agents, benchmarks, and run configs."""
    agents = [path.name for path in Path("agents").glob("*.yaml")]
    benchmarks = ["swebench_verified"]
    run_configs = [path.name for path in Path("configs/runs").glob("*.yaml")]
    print(f"Agents: {', '.join(sorted(agents))}" if agents else "Agents: (none)")
    print(f"Benchmarks: {', '.join(sorted(benchmarks))}" if benchmarks else "Benchmarks: (none)")
    print(f"Run configs: {', '.join(sorted(run_configs))}" if run_configs else "Run configs: (none)")


@app.command()
def run(
    agent: str = typer.Option("agents/qwen2_5_coder.yaml", help="Agent profile path"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark override"),
    split: Optional[str] = typer.Option(None, help="Split override"),
    selector: Optional[int] = typer.Option(None, help="Number of tasks override"),
    mode: Optional[str] = typer.Option(None, help="Mode override: patch_only or tools_enabled"),
    run_config: str = typer.Option("configs/runs/default.yaml", help="Run config path"),
    verbose: bool = typer.Option(
        False,
        "--verbose/--quiet",
        help="Quiet by default; use --verbose to print per-task terminal output.",
    ),
):
    config = _load_run_config(Path(run_config))
    if benchmark:
        config["benchmark"]["name"] = benchmark
    if split:
        config["benchmark"]["split"] = split
    if selector is not None:
        config["runtime"]["selector"] = selector
    if mode:
        config["runtime"]["mode"] = mode

    benchmark_name = config["benchmark"]["name"]
    split_name = config["benchmark"]["split"]
    mode_name = _normalize_mode(str(config["runtime"]["mode"]))

    base_dir = Path.cwd()
    spec_loader = AgentSpecLoader(base_dir)
    spec, prompt, allowed_tools = spec_loader.load(Path(agent))

    adapter = _build_adapter_from_config(config, benchmark_name)
    tasks = adapter.load_tasks(split_name, config["runtime"].get("selector"))

    model_name_or_path = spec.backend.get("model", spec.name)

    run_id = _new_run_id()
    artifacts_dir = Path(config["output"].get("artifacts_dir", "artifacts"))
    run_root = artifacts_dir / run_id
    out_path = run_root / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_path = run_root / "run.log"
    _append_log(
        run_log_path,
        (
            "Starting run:"
            f" run_id={run_id}"
            f" benchmark={benchmark_name}"
            f" split={split_name}"
            f" mode={mode_name}"
            f" tasks={len(tasks)}"
            f" model={model_name_or_path}"
            f" agent_profile={agent}"
        ),
    )

    start_line = (
        "Starting run:"
        f" run_id={run_id}"
        f" benchmark={benchmark_name}"
        f" split={split_name}"
        f" mode={mode_name}"
        f" tasks={len(tasks)}"
        f" model={model_name_or_path}"
    )
    print(start_line)

    valid_artifacts = 0
    invalid_artifacts = 0

    with out_path.open("w", encoding="utf-8") as out_file:
        for task in tasks:
            workspace_root = adapter.workspace_root_for_task(task)
            submitted_artifact: Dict[str, str] = {}

            def _submit_callback(artifact: str):
                submitted_artifact["artifact"] = artifact

            tool_ctx = ToolContext(workspace_root=workspace_root, submit_callback=_submit_callback)
            tool_registry = ToolRegistry(tool_ctx)

            backend = _build_backend(spec.backend)
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
                max_tool_calls=config["runtime"].get("max_tool_calls", 20),
                max_wall_time_s=config["runtime"].get("max_wall_time_s", 600),
            )

            result = runtime.run(
                task=task,
                prompt=prompt,
                tool_schemas=tool_schemas,
                decoding_defaults=spec.decoding_defaults,
            )
            if submitted_artifact.get("artifact"):
                result.final_artifact = submitted_artifact["artifact"]

            policy_result = apply_artifact_policy(result.final_artifact, task.expected_output_type)
            # Patch outputs are pass-through by design. Policy remains diagnostics-only.
            artifact = result.final_artifact
            if task.expected_output_type != "patch":
                artifact = policy_result.artifact

            if policy_result.valid:
                valid_artifacts += 1
            else:
                invalid_artifacts += 1

            record = _serialize_prediction_record(
                adapter=adapter,
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
            _append_log(run_log_path, per_task_line)
            if verbose:
                print(per_task_line)

    created_at = _now_iso()
    manifest_payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": created_at,
        "agent_profile": agent,
        "model_name": spec.name,
        "model_name_or_path": model_name_or_path,
        "benchmark_name": benchmark_name,
        "dataset_name": config["benchmark"].get("dataset_name"),
        "split": split_name,
        "mode": mode_name,
        "predictions_path": str(out_path.resolve()),
        "evaluation": {
            "status": "not_run",
            "returncode": None,
            "report_path": None,
            "harness_log_root": None,
            "metrics": _zero_eval_metrics(),
        },
        "config_snapshot": config,
    }
    manifest_path = _manifest_path(run_root)
    _write_manifest(manifest_path, manifest_payload)

    print(f"Predictions written to {out_path}")
    print("Patch handling: pass-through (invalid patches are retained; diagnostics only).")
    run_summary_line = (
        f"Run summary: run_id={run_id} tasks={len(tasks)} "
        f"valid_artifacts={valid_artifacts} invalid_artifacts={invalid_artifacts}"
    )
    manifest_line = f"Manifest written to {manifest_path}"
    log_file_line = f"Run log written to {run_log_path}"
    _append_log(run_log_path, run_summary_line)
    _append_log(run_log_path, f"Predictions written to {out_path}")
    _append_log(run_log_path, manifest_line)
    _append_log(run_log_path, log_file_line)
    print(run_summary_line)
    print(manifest_line)
    print(log_file_line)


@app.command()
def predict(
    agent: str = typer.Option("agents/qwen2_5_coder.yaml"),
    split: Optional[str] = typer.Option(None),
    selector: Optional[int] = typer.Option(1),
    run_config: str = typer.Option("configs/runs/default.yaml"),
    verbose: bool = typer.Option(
        False,
        "--verbose/--quiet",
        help="Quiet by default; use --verbose to print per-task terminal output.",
    ),
):
    run(
        agent=agent,
        benchmark=None,
        split=split,
        selector=selector,
        mode="patch_only",
        run_config=run_config,
        verbose=verbose,
    )


@app.command()
def eval(
    predictions: str = typer.Argument("artifacts/<run_id>/predictions.jsonl"),
    run_config: str = typer.Option("configs/runs/default.yaml"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark override"),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Verbose by default; use --quiet to only show harness output on failure.",
    ),
):
    config = _load_run_config(Path(run_config))
    if benchmark:
        config["benchmark"]["name"] = benchmark

    benchmark_name = config["benchmark"]["name"]
    predictions_path = Path(predictions)
    artifacts_dir = Path(config["output"].get("artifacts_dir", "artifacts"))
    try:
        run_id = _derive_run_id_from_predictions(predictions_path, artifacts_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="predictions") from exc

    adapter = _build_adapter_from_config(config, benchmark_name)
    evaluator = adapter.get_evaluator()

    run_root = artifacts_dir / run_id
    manifest_path = _manifest_path(run_root)
    existing_manifest = _read_manifest(manifest_path)
    existing_split = existing_manifest.get("split")
    split_name = existing_split if isinstance(existing_split, str) and existing_split else config["benchmark"]["split"]

    print(
        "Starting evaluation:"
        f" run_id={run_id}"
        f" benchmark={benchmark_name}"
        f" split={split_name}"
        f" predictions={predictions_path}"
    )

    proc = evaluator.run_harness(
        predictions_path=predictions_path,
        dataset_name=config["benchmark"]["dataset_name"],
        split=split_name,
        run_id=run_id,
        artifacts_dir=artifacts_dir,
    )

    if verbose:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    elif proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

    relocated_report = getattr(evaluator, "last_summary_report", None)
    harness_log_root = getattr(evaluator, "last_harness_log_root", None)
    metrics, metrics_warning = _read_eval_metrics(relocated_report)
    manifest = existing_manifest
    model_name, model_name_or_path = _read_prediction_identity(predictions_path)
    now = _now_iso()

    if not manifest:
        manifest = {
            "run_id": run_id,
            "created_at": now,
        }

    manifest.update(
        {
            "run_id": run_id,
            "updated_at": now,
            "agent_profile": manifest.get("agent_profile"),
            "model_name": manifest.get("model_name") or model_name,
            "model_name_or_path": manifest.get("model_name_or_path") or model_name_or_path,
            "benchmark_name": benchmark_name,
            "dataset_name": config["benchmark"].get("dataset_name"),
            "split": split_name,
            "mode": manifest.get("mode"),
            "predictions_path": str(predictions_path.resolve()),
            "evaluation": {
                "status": "success" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "report_path": str(relocated_report) if relocated_report else None,
                "harness_log_root": str(harness_log_root) if harness_log_root else None,
                "metrics": metrics,
            },
            "config_snapshot": config,
        }
    )
    _write_manifest(manifest_path, manifest)

    print(
        f"Evaluation summary: run_id={run_id} returncode={proc.returncode} "
        f"status={'success' if proc.returncode == 0 else 'failed'}"
    )
    print(
        "Metrics: "
        f"resolved={metrics['resolved_instances']}/{metrics['submitted_instances']} "
        f"accuracy={_fmt_pct(metrics['accuracy_resolved_submitted'])} "
        f"completed={metrics['completed_instances']} "
        f"unresolved={metrics['unresolved_instances']} "
        f"errors={metrics['error_instances']} "
        f"empty_patch={metrics['empty_patch_instances']}"
    )
    print(
        "Rates: "
        f"resolved/completed={_fmt_pct(metrics['accuracy_resolved_completed'])} "
        f"completion(submitted)={_fmt_pct(metrics['completion_rate_submitted'])}"
    )
    if metrics_warning:
        print(f"Metrics warning: {metrics_warning}")
    print(f"Report: {str(relocated_report) if relocated_report else '(not found)'}")
    print(f"Harness logs: {str(harness_log_root) if harness_log_root else '(not found)'}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    app()
