from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print

from benchmarks.registry import BenchmarkRegistry
from runtime.config_loader import load_run_config
from runtime.eval_service import execute_eval, format_metrics_lines
from runtime.run_service import execute_run

app = typer.Typer(add_completion=False)
load_dotenv()


@app.command()
def list():
    """List available agents, benchmarks, and run configs."""
    agents = [path.name for path in Path("agents").glob("*.yaml")]
    benchmarks = BenchmarkRegistry().list_benchmarks()
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
    """Generate predictions for selected benchmark tasks."""

    config = load_run_config(Path(run_config))
    outcome = execute_run(
        agent_path=agent,
        config=config,
        benchmark=benchmark,
        split=split,
        selector=selector,
        mode=mode,
        verbose=verbose,
    )

    print(
        "Starting run:"
        f" run_id={outcome.run_id}"
        f" benchmark={outcome.benchmark_name}"
        f" split={outcome.split_name}"
        f" mode={outcome.mode_name}"
        f" tasks={outcome.tasks_total}"
        f" model={outcome.model_name_or_path}"
    )
    print(f"Predictions written to {outcome.predictions_path}")
    print("Patch handling: pass-through (invalid patches are retained; diagnostics only).")
    print(
        f"Run summary: run_id={outcome.run_id} tasks={outcome.tasks_total} "
        f"valid_artifacts={outcome.valid_artifacts} invalid_artifacts={outcome.invalid_artifacts}"
    )
    print(f"Manifest written to {outcome.manifest_path}")
    print(f"Run log written to {outcome.run_log_path}")


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
    """Convenience wrapper for `run` forced to patch-only mode."""

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
    """Evaluate a canonical predictions file with benchmark harness."""

    config = load_run_config(Path(run_config))
    predictions_path = Path(predictions)

    try:
        outcome = execute_eval(
            predictions_path=predictions_path,
            config=config,
            benchmark=benchmark,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="predictions") from exc

    print(
        "Starting evaluation:"
        f" run_id={outcome.run_id}"
        f" benchmark={outcome.benchmark_name}"
        f" split={outcome.split_name}"
        f" predictions={predictions_path}"
    )

    if verbose:
        if outcome.proc.stdout:
            print(outcome.proc.stdout)
        if outcome.proc.stderr:
            print(outcome.proc.stderr)
    elif outcome.proc.returncode != 0:
        if outcome.proc.stdout:
            print(outcome.proc.stdout)
        if outcome.proc.stderr:
            print(outcome.proc.stderr)

    print(
        f"Evaluation summary: run_id={outcome.run_id} returncode={outcome.proc.returncode} "
        f"status={'success' if outcome.proc.returncode == 0 else 'failed'}"
    )
    summary_line, rates_line = format_metrics_lines(outcome.metrics)
    print(summary_line)
    print(rates_line)
    if outcome.metrics_warning:
        print(f"Metrics warning: {outcome.metrics_warning}")
    print(f"Report: {str(outcome.report_path) if outcome.report_path else '(not found)'}")
    print(f"Harness logs: {str(outcome.harness_log_root) if outcome.harness_log_root else '(not found)'}")
    print(f"Manifest written to {outcome.manifest_path}")


if __name__ == "__main__":
    app()
