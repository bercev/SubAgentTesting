from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from benchmarks.registry import BenchmarkRegistry
from runtime.config_loader import apply_run_overrides
from runtime.config_models import RunConfig
from runtime.manifest_store import manifest_path, now_iso, read_manifest, write_manifest
from runtime.metrics import fmt_pct, read_eval_metrics


RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


@dataclass
class EvalOutcome:
    """Structured result returned after one evaluation invocation."""

    run_id: str
    benchmark_name: str
    split_name: str
    proc: subprocess.CompletedProcess[str]
    report_path: Optional[Path]
    harness_log_root: Optional[Path]
    manifest_path: Path
    metrics: Dict[str, Any]
    metrics_warning: Optional[str]


def is_valid_run_id(value: str) -> bool:
    """Validate canonical timestamp-based run id format."""

    return bool(RUN_ID_PATTERN.match(value))


def derive_run_id_from_predictions(predictions_path: Path, artifacts_dir: Path) -> str:
    """Derive run id strictly from canonical predictions path layout."""

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
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run_id in predictions path: {run_id}")
    return run_id


def read_prediction_identity(predictions_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Read model identity fields from first non-empty prediction row."""

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


def format_metrics_lines(metrics: Dict[str, Any]) -> Tuple[str, str]:
    """Render compact summary lines for terminal evaluation output."""

    summary_line = (
        "Metrics: "
        f"resolved={metrics['resolved_instances']}/{metrics['submitted_instances']} "
        f"accuracy={fmt_pct(metrics['accuracy_resolved_submitted'])} "
        f"completed={metrics['completed_instances']} "
        f"unresolved={metrics['unresolved_instances']} "
        f"errors={metrics['error_instances']} "
        f"empty_patch={metrics['empty_patch_instances']}"
    )
    rates_line = (
        "Rates: "
        f"resolved/completed={fmt_pct(metrics['accuracy_resolved_completed'])} "
        f"completion(submitted)={fmt_pct(metrics['completion_rate_submitted'])}"
    )
    return summary_line, rates_line


def execute_eval(
    *,
    predictions_path: Path,
    config: RunConfig,
    benchmark: Optional[str] = None,
) -> EvalOutcome:
    """Run benchmark evaluator and persist evaluation status into manifest."""

    effective_config = apply_run_overrides(config, benchmark=benchmark)
    benchmark_name = effective_config.benchmark.name

    artifacts_dir = Path(effective_config.output.artifacts_dir)
    run_id = derive_run_id_from_predictions(predictions_path, artifacts_dir)

    run_root = artifacts_dir / run_id
    out_manifest_path = manifest_path(run_root)
    existing_manifest = read_manifest(out_manifest_path)
    existing_split = existing_manifest.get("split")
    split_name = (
        existing_split
        if isinstance(existing_split, str) and existing_split
        else effective_config.benchmark.split
    )
    effective_config.benchmark.split = split_name

    adapter_cls = BenchmarkRegistry().get_adapter(benchmark_name)
    adapter = adapter_cls.from_config(effective_config)
    evaluator = adapter.get_evaluator(effective_config)

    proc = evaluator.run_harness(
        predictions_path=predictions_path,
        run_id=run_id,
        config=effective_config,
    )

    relocated_report = getattr(evaluator, "last_summary_report", None)
    harness_log_root = getattr(evaluator, "last_harness_log_root", None)
    metrics, metrics_warning = read_eval_metrics(relocated_report)
    manifest = existing_manifest
    model_name, model_name_or_path = read_prediction_identity(predictions_path)
    now = now_iso()

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
            "dataset_name": effective_config.benchmark.dataset_name,
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
            "config_snapshot": effective_config.model_dump(mode="python"),
        }
    )
    write_manifest(out_manifest_path, manifest)

    return EvalOutcome(
        run_id=run_id,
        benchmark_name=benchmark_name,
        split_name=split_name,
        proc=proc,
        report_path=relocated_report,
        harness_log_root=harness_log_root,
        manifest_path=out_manifest_path,
        metrics=metrics,
        metrics_warning=metrics_warning,
    )
