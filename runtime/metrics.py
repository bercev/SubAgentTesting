from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


EVAL_COUNT_KEYS = (
    "total_instances",
    "submitted_instances",
    "completed_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "error_instances",
)


def _to_int(value: Any) -> int:
    """Convert metric-like values to non-negative integers."""

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
    """Return zero-safe division result for derived rate metrics."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def zero_eval_metrics() -> Dict[str, Any]:
    """Return zeroed metrics payload used when report parsing fails."""

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


def read_eval_metrics(report_path: Optional[Path]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse harness report JSON and compute derived accuracy/completion rates."""

    metrics = zero_eval_metrics()
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


def fmt_pct(value: Any) -> str:
    """Format ratios as percentage strings for CLI summaries."""

    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"
