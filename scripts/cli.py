import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            "report_dir": "logs/reports",
        },
        "runtime": {
            "mode": "patch_only",
            "selector": 5,
            "max_tool_calls": 20,
            "max_wall_time_s": 600,
        },
        "output": {
            "runs_dir": "runs",
            "logs_dir": "logs",
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
        },
        "runtime": {
            "mode": raw_config.get("mode"),
            "selector": raw_config.get("selector"),
        },
        "output": {
            "runs_dir": raw_config.get("runs_dir"),
            "logs_dir": raw_config.get("logs_dir"),
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


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


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


@app.command()
def list():
    """List available agents, benchmarks, and run configs."""
    agents = [path.name for path in Path("agents").glob("*.yaml")]
    benchmarks = ["swebench_verified"]
    run_configs = [path.name for path in Path("configs/runs").glob("*.yaml")]
    print({"agents": agents, "benchmarks": benchmarks, "run_configs": run_configs})


@app.command()
def run(
    agent: str = typer.Option("agents/qwen2_5_coder.yaml", help="Agent profile path"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark override"),
    split: Optional[str] = typer.Option(None, help="Split override"),
    selector: Optional[int] = typer.Option(None, help="Number of tasks override"),
    mode: Optional[str] = typer.Option(None, help="Mode override: patch_only or tools_enabled"),
    run_config: str = typer.Option("configs/runs/default.yaml", help="Run config path"),
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

    model_name = _sanitize_model_name(spec.backend.get("model", spec.name))
    date_tag = datetime.now().strftime("%Y-%m-%d_%H%M")
    runs_dir = Path(config["output"].get("runs_dir", "runs"))
    out_path = runs_dir / model_name / split_name / mode_name / f"{date_tag}_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
            artifact = policy_result.artifact
            if not policy_result.valid and task.expected_output_type == "patch":
                artifact = ""

            record = _serialize_prediction_record(
                adapter=adapter,
                task=task,
                artifact=artifact,
                model_name_or_path=spec.backend.get("model", spec.name),
                model_name=spec.name,
                metadata=result.metadata,
            )
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
            print(
                {
                    "task": task.task_id,
                    "terminated": result.metadata.get("terminated") if result.metadata else False,
                    "output_type": task.expected_output_type,
                    "artifact_valid": policy_result.valid,
                    "artifact_reason": policy_result.reason,
                }
            )

    print(f"Wrote predictions to {out_path}")


@app.command()
def predict(
    agent: str = typer.Option("agents/qwen2_5_coder.yaml"),
    split: Optional[str] = typer.Option(None),
    selector: Optional[int] = typer.Option(1),
    run_config: str = typer.Option("configs/runs/default.yaml"),
):
    run(
        agent=agent,
        benchmark=None,
        split=split,
        selector=selector,
        mode="patch_only",
        run_config=run_config,
    )


@app.command()
def eval(
    predictions: str = typer.Argument("runs/predictions.jsonl"),
    run_config: str = typer.Option("configs/runs/default.yaml"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark override"),
):
    config = _load_run_config(Path(run_config))
    if benchmark:
        config["benchmark"]["name"] = benchmark

    benchmark_name = config["benchmark"]["name"]
    adapter = _build_adapter_from_config(config, benchmark_name)
    evaluator = adapter.get_evaluator()

    predictions_path = Path(predictions)
    split_name = config["benchmark"]["split"]
    runs_dir = Path(config["output"].get("runs_dir", "runs"))
    try:
        parts = predictions_path.parts
        if runs_dir.name in parts:
            split_name = parts[parts.index(runs_dir.name) + 2]
    except Exception:
        pass

    proc = evaluator.run_harness(
        predictions_path=predictions_path,
        dataset_name=config["benchmark"]["dataset_name"],
        split=split_name,
        run_id=datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    )
    print(proc.stdout)
    relocated_report = getattr(evaluator, "last_summary_report", None)
    if relocated_report:
        print(f"Relocated report to {relocated_report}")
    if proc.returncode != 0:
        print(proc.stderr)


if __name__ == "__main__":
    app()
