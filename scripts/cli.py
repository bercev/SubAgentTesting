import json
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from agents.spec_loader import AgentSpecLoader
from benchmarks.registry import BenchmarkRegistry
from runtime.agent_runtime import AgentRuntime
from runtime.model_backend import OpenRouterBackend
from runtime.schemas import AgentResult
from runtime.tools import ToolContext, ToolRegistry

app = typer.Typer(add_completion=False)


@app.command()
def list():
    """List available agents and benchmarks."""
    agents = [p.name for p in Path("agents").glob("*.yaml")]
    benchmarks = ["swebench_verified"]
    print({"agents": agents, "benchmarks": benchmarks})


def _build_backend(kind: str, model: Optional[str]) -> OpenRouterBackend:
    if kind == "openrouter":
        return OpenRouterBackend(model=model)
    raise ValueError(f"Unknown backend type {kind}")


def _filter_tool_schemas(registry: ToolRegistry, allowed: set) -> List[dict]:
    schemas = []
    for s in registry.schemas:
        name = s.get("function", {}).get("name")
        if name in allowed:
            schemas.append(s)
    return schemas


@app.command()
def run(
    agent: str = typer.Option("agents/qwen2_5_coder.yaml", help="Agent spec path"),
    benchmark: str = typer.Option("swebench_verified"),
    split: str = typer.Option("dev"),
    selector: Optional[int] = typer.Option(None, help="Number of tasks"),
    mode: str = typer.Option("B", help="Mode A (patch-only) or B (tools-enabled)"),
):
    base_dir = Path.cwd()
    loader = AgentSpecLoader(base_dir)
    spec, prompt, allowed_tools = loader.load(Path(agent))

    registry = BenchmarkRegistry()
    adapter_cls = registry.get_adapter(benchmark)
    adapter = adapter_cls()
    tasks = adapter.load_tasks(split, selector)

    results: List[AgentResult] = []
    for task in tasks:
        workspace_root = adapter.workspace_root_for_task(task)
        submitted_artifact: dict = {}

        def _submit_cb(artifact: str):
            submitted_artifact["artifact"] = artifact

        tool_ctx = ToolContext(workspace_root=workspace_root, submit_callback=_submit_cb)
        tool_registry = ToolRegistry(tool_ctx)

        if mode.upper() == "A":
            backend = _build_backend(spec.backend.get("type", "openrouter"), spec.backend.get("model"))
            allowed = {"submit"}
        else:
            backend = _build_backend(spec.backend.get("type", "openrouter"), spec.backend.get("model"))
            allowed = allowed_tools or set([s.get("function", {}).get("name") for s in tool_registry.schemas])
        tool_schemas = _filter_tool_schemas(tool_registry, allowed)

        runtime = AgentRuntime(
            backend=backend,
            tool_registry=tool_registry,
            allowed_tools=allowed,
            max_tool_calls=20,
            max_wall_time_s=600,
        )

        result = runtime.run(
            task=task,
            prompt=prompt,
            tool_schemas=tool_schemas,
            decoding_defaults=spec.decoding_defaults,
        )
        if submitted_artifact.get("artifact"):
            result.final_artifact = submitted_artifact["artifact"]
        results.append(result)
        print({"task": task.task_id, "terminated": result.metadata.get("terminated")})

    # Persist predictions for SWE-bench shape
    evalr = adapter.get_evaluator()
    out_path = Path("runs/predictions.jsonl")
    evalr.write_predictions(results, spec.name, out_path)
    print(f"Wrote predictions to {out_path}")


@app.command()
def predict(
    agent: str = typer.Option("agents/qwen2_5_coder.yaml"),
    split: str = typer.Option("dev"),
    selector: Optional[int] = typer.Option(1),
):
    # Reuse run logic in Mode A (patch-only)
    run(agent=agent, benchmark="swebench_verified", split=split, selector=selector, mode="A")


@app.command()
def eval(
    predictions: str = typer.Argument("runs/predictions.jsonl"),
):
    adapter = BenchmarkRegistry().get_adapter("swebench_verified")()
    evaluator = adapter.get_evaluator()
    proc = evaluator.run_harness(Path(predictions))
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)


if __name__ == "__main__":
    app()
