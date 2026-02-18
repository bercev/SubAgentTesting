from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


COMPONENT_KEYS = (
    "execution_quality",
    "policy_quality",
    "termination_quality",
    "budget_quality",
)


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert an arbitrary value into an integer, falling back to default."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _safe_div(numerator: int, denominator: int) -> float:
    """Return a zero-safe division result."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _null_components() -> Dict[str, Any]:
    """Return component payload with null values for non-applicable runs/tasks."""

    return {key: None for key in COMPONENT_KEYS}


def _weights_dict(weights: Mapping[str, Any]) -> Dict[str, float]:
    """Normalize weights to a float-only dictionary."""

    return {key: float(weights.get(key, 0.0)) for key in COMPONENT_KEYS}


def _weighted_score(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Compute weighted linear score from component/weight maps."""

    return sum(float(components[key]) * float(weights[key]) for key in COMPONENT_KEYS)


def serialize_tool_call_row(
    *,
    run_id: str,
    task_id: str,
    mode: str,
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach run/task context and normalize one runtime tool-call event row."""

    return {
        "row_type": "tool_call",
        "run_id": run_id,
        "task_id": task_id,
        "mode": mode,
        "turn_index": max(0, _safe_int(event.get("turn_index"))),
        "call_index": max(0, _safe_int(event.get("call_index"))),
        "tool_name": str(event.get("tool_name", "")),
        "is_termination_tool": bool(event.get("is_termination_tool", False)),
        "allowed": bool(event.get("allowed", False)),
        "executed": bool(event.get("executed", False)),
        "success": bool(event.get("success", False)),
        "error_code": str(event.get("error_code", "tool_error")),
        "args_size_bytes": max(0, _safe_int(event.get("args_size_bytes"))),
        "result_size_bytes": max(0, _safe_int(event.get("result_size_bytes"))),
        "latency_ms": max(0, _safe_int(event.get("latency_ms"))),
        "return_code": (
            _safe_int(event.get("return_code"))
            if isinstance(event.get("return_code"), int)
            else None
        ),
    }


def build_task_summary(
    *,
    run_id: str,
    task_id: str,
    mode: str,
    runtime_payload: Mapping[str, Any] | None,
    weights: Mapping[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    """Build one task-level tool-quality summary row from runtime telemetry."""

    payload = runtime_payload if isinstance(runtime_payload, Mapping) else {}
    raw_events = payload.get("events")
    events = [event for event in raw_events if isinstance(event, Mapping)] if isinstance(raw_events, list) else []

    tool_calls_total = len(events)
    tool_calls_success = sum(1 for event in events if bool(event.get("success", False)))
    tool_calls_denied = sum(
        1
        for event in events
        if event.get("error_code") == "not_allowed" or bool(event.get("allowed", True)) is False
    )
    tool_calls_failed = max(0, tool_calls_total - tool_calls_success)

    termination_ack = bool(payload.get("termination_ack", False))
    budget_exhausted = bool(payload.get("budget_exhausted", False))
    wall_time_exhausted = bool(payload.get("wall_time_exhausted", False))
    loop_exit_reason = str(payload.get("loop_exit_reason", "unknown"))

    applicable = bool(enabled and mode == "tools_enabled" and tool_calls_total > 0)
    normalized_weights = _weights_dict(weights)

    if applicable:
        components: Dict[str, Any] = {
            "execution_quality": _safe_div(tool_calls_success, tool_calls_total),
            "policy_quality": 1.0 - _safe_div(tool_calls_denied, tool_calls_total),
            "termination_quality": 1.0 if termination_ack else 0.0,
            "budget_quality": 0.0 if budget_exhausted else 1.0,
        }
        score: float | None = _weighted_score(components, normalized_weights)
    else:
        components = _null_components()
        score = None

    return {
        "row_type": "task_summary",
        "run_id": run_id,
        "task_id": task_id,
        "tool_quality_applicable": applicable,
        "tool_calls_total": tool_calls_total,
        "tool_calls_success": tool_calls_success,
        "tool_calls_failed": tool_calls_failed,
        "tool_calls_denied": tool_calls_denied,
        "termination_ack": termination_ack,
        "budget_exhausted": budget_exhausted,
        "wall_time_exhausted": wall_time_exhausted,
        "loop_exit_reason": loop_exit_reason,
        "components": components,
        "score": score,
    }


def build_run_summary(
    *,
    telemetry_path: Path,
    task_summaries: Iterable[Mapping[str, Any]],
    weights: Mapping[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    """Aggregate task-level summaries into one run-level manifest payload."""

    summaries = list(task_summaries)
    normalized_weights = _weights_dict(weights)

    tool_calls_total = sum(max(0, _safe_int(summary.get("tool_calls_total"))) for summary in summaries)
    tool_calls_success = sum(max(0, _safe_int(summary.get("tool_calls_success"))) for summary in summaries)
    tool_calls_failed = sum(max(0, _safe_int(summary.get("tool_calls_failed"))) for summary in summaries)
    tool_calls_denied = sum(max(0, _safe_int(summary.get("tool_calls_denied"))) for summary in summaries)

    applicable_summaries = [summary for summary in summaries if bool(summary.get("tool_quality_applicable", False))]
    tasks_total = len(summaries)
    tasks_applicable = len(applicable_summaries)
    tasks_budget_exhausted = sum(1 for summary in applicable_summaries if bool(summary.get("budget_exhausted", False)))
    tasks_termination_ack = sum(1 for summary in applicable_summaries if bool(summary.get("termination_ack", False)))

    run_applicable = bool(enabled and tasks_applicable > 0)

    if run_applicable:
        components: Dict[str, Any] = {
            "execution_quality": _safe_div(tool_calls_success, tool_calls_total),
            "policy_quality": 1.0 - _safe_div(tool_calls_denied, tool_calls_total),
            "termination_quality": _safe_div(tasks_termination_ack, tasks_applicable),
            "budget_quality": 1.0 - _safe_div(tasks_budget_exhausted, tasks_applicable),
        }
        score: float | None = _weighted_score(components, normalized_weights)
    else:
        components = _null_components()
        score = None

    return {
        "version": "v1",
        "applicable": run_applicable,
        "score": score,
        "weights": normalized_weights,
        "components": components,
        "counts": {
            "tool_calls_total": tool_calls_total,
            "tool_calls_success": tool_calls_success,
            "tool_calls_failed": tool_calls_failed,
            "tool_calls_denied": tool_calls_denied,
            "tasks_total": tasks_total,
            "tasks_applicable": tasks_applicable,
            "tasks_budget_exhausted": tasks_budget_exhausted,
            "tasks_termination_ack": tasks_termination_ack,
        },
        "telemetry_path": str(telemetry_path.resolve()),
    }


def format_task_tool_quality_log(summary: Mapping[str, Any]) -> str:
    """Render a compact per-task log line."""

    score = summary.get("score")
    score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
    return (
        "tool_quality "
        f"task={summary.get('task_id')} "
        f"applicable={summary.get('tool_quality_applicable')} "
        f"score={score_text} "
        f"calls={summary.get('tool_calls_success')}/{summary.get('tool_calls_total')} "
        f"denied={summary.get('tool_calls_denied')} "
        f"failed={summary.get('tool_calls_failed')} "
        f"termination_ack={summary.get('termination_ack')} "
        f"budget_exhausted={summary.get('budget_exhausted')} "
        f"exit={summary.get('loop_exit_reason')}"
    )


def format_run_tool_quality_log(summary: Mapping[str, Any]) -> str:
    """Render a compact run-level log line."""

    score = summary.get("score")
    score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
    counts = summary.get("counts", {}) if isinstance(summary.get("counts"), Mapping) else {}
    return (
        "tool_quality summary "
        f"applicable={summary.get('applicable')} "
        f"score={score_text} "
        f"tasks_applicable={counts.get('tasks_applicable')}/{counts.get('tasks_total')} "
        f"calls={counts.get('tool_calls_success')}/{counts.get('tool_calls_total')} "
        f"denied={counts.get('tool_calls_denied')} "
        f"failed={counts.get('tool_calls_failed')} "
        f"termination_ack={counts.get('tasks_termination_ack')} "
        f"budget_exhausted={counts.get('tasks_budget_exhausted')}"
    )
