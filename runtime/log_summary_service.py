from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from runtime.manifest_store import append_log, manifest_path, now_iso, read_manifest, write_manifest

RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
LOG_LINE_SPLIT = " | "
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TRUNCATED_MARKER = "...[truncated]"
POST_RUN_SUMMARY_BLOCK_BEGIN = "post_run_summary_block_begin"
POST_RUN_SUMMARY_BLOCK_LINE = "post_run_summary_line"
POST_RUN_SUMMARY_BLOCK_END = "post_run_summary_block_end"


@dataclass
class RunLogSummaryOutcome:
    """Structured result for one post-run log summarization pass."""

    run_id: str
    run_root: Path
    run_log_path: Path
    manifest_path: Path
    summary: Dict[str, Any]
    terminal_lines: List[str]


def is_valid_run_id(value: str) -> bool:
    """Validate canonical timestamp-based run identifier."""

    return bool(RUN_ID_PATTERN.match(value))


def derive_run_id_from_run_log(run_log_path: Path, artifacts_dir: Path) -> str:
    """Derive run id strictly from canonical run-log path layout."""

    abs_run_log = run_log_path.resolve()
    abs_artifacts = artifacts_dir.resolve()

    try:
        rel = abs_run_log.relative_to(abs_artifacts)
    except Exception as exc:
        raise ValueError(
            "Run log path must be under "
            f"{abs_artifacts}/<run_id>/run.log; got {abs_run_log}"
        ) from exc

    if len(rel.parts) != 2 or rel.parts[1] != "run.log":
        raise ValueError(
            "Run log path must match artifacts/<run_id>/run.log; "
            f"got {abs_run_log}"
        )

    run_id = rel.parts[0]
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run_id in run log path: {run_id}")
    return run_id


def _parse_prefix(line: str) -> Optional[Dict[str, str]]:
    """Parse one structured run-log line prefix."""

    parts = line.rstrip("\n").split(LOG_LINE_SPLIT, 3)
    if len(parts) != 4:
        return None
    return {
        "timestamp": parts[0].strip(),
        "level": parts[1].strip(),
        "source": parts[2].strip(),
        "message": parts[3].strip(),
    }


def _parse_timestamp(value: str) -> Optional[datetime]:
    """Parse log timestamps using the run-log format."""

    try:
        return datetime.strptime(value, LOG_TIME_FORMAT)
    except ValueError:
        return None


def _extract_value(message: str, key: str) -> Optional[str]:
    """Extract a single scalar key=value token (space-delimited value)."""

    match = re.search(rf"(?:^| ){re.escape(key)}=([^ ]+)", message)
    if not match:
        return None
    return match.group(1)


def _extract_trailing_value(message: str, key: str) -> Optional[str]:
    """Extract a trailing key=value payload, assuming it is the last field."""

    marker = f"{key}="
    idx = message.find(marker)
    if idx == -1:
        return None
    return message[idx + len(marker) :].strip()


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _percentile(values: List[int], percentile: float) -> Optional[int]:
    """Nearest-rank percentile for small integer latency samples."""

    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = max(1, int(round(percentile * len(sorted_values))))
    rank = min(rank, len(sorted_values))
    return sorted_values[rank - 1]


def _latency_summary(values: List[int]) -> Dict[str, Any]:
    """Build latency summary stats for API responses."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "avg": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "avg": round(sum(values) / len(values), 2),
        "max": max(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _format_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _format_usd(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"${value:.6f}"


def _parse_legacy_usage_from_api_response_preview(preview: str) -> Optional[Dict[str, Any]]:
    """Best-effort fallback parser for older logs without api_usage lines."""

    if not preview or TRUNCATED_MARKER in preview:
        return None
    if not preview.startswith("{"):
        return None
    try:
        payload = json.loads(preview)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "response_id": payload.get("id") if isinstance(payload.get("id"), str) else None,
        "provider": payload.get("provider") if isinstance(payload.get("provider"), str) else None,
        "model": payload.get("model") if isinstance(payload.get("model"), str) else None,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), (int, float)) else None,
        "completion_tokens": (
            usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), (int, float))
            else None
        ),
        "total_tokens": usage.get("total_tokens") if isinstance(usage.get("total_tokens"), (int, float)) else None,
        "cost_usd": usage.get("cost") if isinstance(usage.get("cost"), (int, float)) else None,
        "is_byok": usage.get("is_byok") if isinstance(usage.get("is_byok"), bool) else None,
    }


def _summarize_tool_telemetry(telemetry_path: Path, warnings: List[str]) -> Dict[str, Any]:
    """Aggregate `tool_telemetry.jsonl` rows when present."""

    if not telemetry_path.exists():
        return {
            "telemetry_present": False,
            "telemetry_path": str(telemetry_path.resolve()),
            "tool_calls_total": 0,
            "tool_calls_success": 0,
            "tool_calls_failed": 0,
            "task_summaries_total": 0,
            "by_tool": {},
        }

    tool_counts: Dict[str, Dict[str, Any]] = {}
    tool_calls_total = 0
    tool_calls_success = 0
    tool_calls_failed = 0
    task_summaries_total = 0

    for line_no, line in enumerate(telemetry_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            warnings.append(f"tool_telemetry parse failed at line {line_no}")
            continue
        if not isinstance(row, dict):
            continue
        row_type = row.get("row_type")
        if row_type == "tool_call":
            tool_calls_total += 1
            success = bool(row.get("success"))
            if success:
                tool_calls_success += 1
            else:
                tool_calls_failed += 1
            tool_name = row.get("tool_name") if isinstance(row.get("tool_name"), str) else "unknown"
            bucket = tool_counts.setdefault(
                tool_name,
                {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "latency_ms_total": 0,
                },
            )
            bucket["total"] += 1
            bucket["success"] += 1 if success else 0
            bucket["failed"] += 0 if success else 1
            latency_ms = row.get("latency_ms")
            if isinstance(latency_ms, (int, float)):
                bucket["latency_ms_total"] += int(latency_ms)
        elif row_type == "task_summary":
            task_summaries_total += 1

    by_tool_sorted = dict(
        sorted(
            tool_counts.items(),
            key=lambda item: (-item[1]["total"], item[0]),
        )
    )
    return {
        "telemetry_present": True,
        "telemetry_path": str(telemetry_path.resolve()),
        "tool_calls_total": tool_calls_total,
        "tool_calls_success": tool_calls_success,
        "tool_calls_failed": tool_calls_failed,
        "task_summaries_total": task_summaries_total,
        "by_tool": by_tool_sorted,
    }


def _event_name(message: str) -> Optional[str]:
    """Extract event token, accounting for `task=<id>` prefixed backend logs."""

    if not message:
        return None
    tokens = message.split(" ", 2)
    if tokens[0].startswith("task="):
        if len(tokens) >= 2 and "=" not in tokens[1]:
            return tokens[1]
        return None
    first = tokens[0]
    if "=" in first:
        return None
    return first


def _format_terminal_lines(summary: Dict[str, Any], manifest: Dict[str, Any]) -> List[str]:
    """Render a concise human-readable post-run summary for terminal output."""

    tasks = summary.get("tasks", {})
    api = summary.get("api", {})
    cost = summary.get("openrouter_cost", {})
    latency = api.get("latency_ms", {})
    warnings = summary.get("warnings", [])
    tools = summary.get("tools", {})
    per_task = tasks.get("per_task", [])

    benchmark = manifest.get("benchmark_name") or "unknown"
    split = manifest.get("split") or "unknown"
    mode = manifest.get("mode") or "unknown"
    model = manifest.get("model_name_or_path") or manifest.get("model_name") or "unknown"

    lines = [
        (
            "Post-run summary:"
            f" run_id={summary.get('run_id')}"
            f" benchmark={benchmark}"
            f" split={split}"
            f" mode={mode}"
            f" model={model}"
        ),
        (
            "Run timing:"
            f" start={summary.get('started_at') or 'unknown'}"
            f" end={summary.get('ended_at') or 'unknown'}"
            f" duration_s={summary.get('duration_s')}"
        ),
        (
            "Tasks:"
            f" started={tasks.get('started_count', 0)}"
            f" completed={tasks.get('completed_count', 0)}"
            f" valid={tasks.get('artifact_valid_true', 0)}"
            f" invalid={tasks.get('artifact_valid_false', 0)}"
            f" unknown={tasks.get('artifact_valid_unknown', 0)}"
        ),
        (
            "API:"
            f" requests={api.get('requests', 0)}"
            f" responses={api.get('responses', 0)}"
            f" errors={api.get('errors', 0)}"
            f" retries={api.get('retries', 0)}"
            f" usage_events={api.get('usage_events', 0)}"
        ),
        (
            "API latency ms:"
            f" count={latency.get('count', 0)}"
            f" min={latency.get('min')}"
            f" avg={latency.get('avg')}"
            f" p50={latency.get('p50')}"
            f" p95={latency.get('p95')}"
            f" max={latency.get('max')}"
        ),
        (
            "OpenRouter cost:"
            f" total={_format_usd(cost.get('cost_usd_total'))}"
            f" source={cost.get('source')}"
            f" coverage={cost.get('usage_covered_responses', 0)}/{cost.get('responses_total', 0)}"
            f" prompt_tokens={cost.get('prompt_tokens_total')}"
            f" completion_tokens={cost.get('completion_tokens_total')}"
            f" total_tokens={cost.get('total_tokens_total')}"
        ),
    ]

    status_codes = api.get("status_codes", {})
    if status_codes:
        compact_codes = ",".join(f"{code}:{count}" for code, count in sorted(status_codes.items()))
        lines.append(f"API status codes: {compact_codes}")

    if per_task:
        lines.append("Task outcomes:")
        for item in per_task[:10]:
            lines.append(
                "  "
                f"{item.get('task_id')}:"
                f" terminated={item.get('terminated')}"
                f" artifact_valid={item.get('artifact_valid')}"
                f" reason={item.get('artifact_reason')}"
                f" artifact_bytes={item.get('artifact_bytes')}"
                f" invalid_submit_attempts={item.get('invalid_submit_attempts')}"
                f" last_invalid_submit_reason={item.get('last_invalid_submit_reason')}"
            )
        if len(per_task) > 10:
            lines.append(f"  ... {len(per_task) - 10} more tasks")

    if tools.get("telemetry_present"):
        lines.append(
            "Tools:"
            f" calls={tools.get('tool_calls_total', 0)}"
            f" success={tools.get('tool_calls_success', 0)}"
            f" failed={tools.get('tool_calls_failed', 0)}"
        )
        top_tools = list((tools.get("by_tool") or {}).items())[:5]
        if top_tools:
            lines.append("Top tools:")
            for tool_name, counts in top_tools:
                lines.append(
                    "  "
                    f"{tool_name}: total={counts.get('total', 0)}"
                    f" success={counts.get('success', 0)}"
                    f" failed={counts.get('failed', 0)}"
                    f" latency_ms_total={counts.get('latency_ms_total', 0)}"
                )

    if warnings:
        lines.append("Summary warnings:")
        for warning in warnings[:10]:
            lines.append(f"  {warning}")
        if len(warnings) > 10:
            lines.append(f"  ... {len(warnings) - 10} more warnings")

    return lines


def _append_summary_block_to_run_log(
    *,
    run_log_path: Path,
    terminal_lines: List[str],
    status: str,
) -> None:
    """Append a marked generated summary block into the run log."""

    append_log(
        run_log_path,
        f"{POST_RUN_SUMMARY_BLOCK_BEGIN} version=v1",
        source="log_summary_service.py",
    )
    for index, line in enumerate(terminal_lines, start=1):
        normalized_text = (line or "").replace("\r", " ").replace("\n", " ")
        append_log(
            run_log_path,
            f"{POST_RUN_SUMMARY_BLOCK_LINE} index={index} text={normalized_text}",
            source="log_summary_service.py",
        )
    append_log(
        run_log_path,
        f"{POST_RUN_SUMMARY_BLOCK_END} version=v1 status={status}",
        source="log_summary_service.py",
    )


def execute_run_log_summary(*, run_log_path: Path) -> RunLogSummaryOutcome:
    """Parse one run log, update manifest with summary, and return terminal lines."""

    if not run_log_path.exists():
        raise ValueError(f"Run log not found: {run_log_path}")
    if not run_log_path.is_file():
        raise ValueError(f"Run log path is not a file: {run_log_path}")

    artifacts_dir = run_log_path.parent.parent
    run_id = derive_run_id_from_run_log(run_log_path, artifacts_dir)
    run_root = artifacts_dir / run_id
    out_manifest_path = manifest_path(run_root)
    manifest = read_manifest(out_manifest_path)
    warnings: List[str] = []

    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    started_at_text: Optional[str] = None
    ended_at_text: Optional[str] = None

    run_meta: Dict[str, Any] = {}
    run_summary_line_counts: Dict[str, int] = {}
    tasks: Dict[str, Dict[str, Any]] = {}
    artifact_reason_counts: Counter[str] = Counter()
    api_status_codes: Counter[str] = Counter()
    api_error_kinds: Counter[str] = Counter()
    api_retry_reasons: Counter[str] = Counter()
    response_latencies_ms: List[int] = []
    assistant_chars_total = 0
    tool_calls_total = 0

    api_counters = {
        "requests": 0,
        "responses": 0,
        "responses_success": 0,
        "errors": 0,
        "retries": 0,
        "parsed": 0,
        "results": 0,
        "usage_events": 0,
    }

    usage_rows: List[Dict[str, Any]] = []
    legacy_response_previews: List[str] = []
    tool_quality_summary: Dict[str, Any] = {}

    in_generated_summary_block = False

    for line_no, raw_line in enumerate(run_log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        parsed_line = _parse_prefix(raw_line)
        if parsed_line is None:
            warnings.append(f"malformed run.log line {line_no}")
            continue

        message = parsed_line["message"]
        if message.startswith(POST_RUN_SUMMARY_BLOCK_BEGIN):
            in_generated_summary_block = True
            continue
        if message.startswith(POST_RUN_SUMMARY_BLOCK_END):
            in_generated_summary_block = False
            continue
        if in_generated_summary_block or message.startswith(POST_RUN_SUMMARY_BLOCK_LINE):
            continue

        timestamp_text = parsed_line["timestamp"]
        dt = _parse_timestamp(timestamp_text)
        if dt is None:
            warnings.append(f"invalid timestamp at line {line_no}")
        else:
            if first_ts is None or dt < first_ts:
                first_ts = dt
                started_at_text = timestamp_text
            if last_ts is None or dt > last_ts:
                last_ts = dt
                ended_at_text = timestamp_text

        level = parsed_line["level"]
        task_id = _extract_value(message, "task")

        if message.startswith("Starting run:"):
            for key in ("run_id", "benchmark", "split", "mode", "tasks", "model", "agent_profile"):
                value = _extract_value(message, key)
                if value is not None:
                    run_meta[key] = value
        elif message.startswith("Run summary:"):
            for key in ("tasks", "valid_artifacts", "invalid_artifacts"):
                value = _to_int(_extract_value(message, key))
                if value is not None:
                    run_summary_line_counts[key] = value
        elif message.startswith("tool_quality summary"):
            score = _to_float(_extract_value(message, "score"))
            applicable = _to_bool(_extract_value(message, "applicable"))
            calls = _extract_value(message, "calls")
            tool_quality_summary = {
                "score": score,
                "applicable": applicable,
                "calls": calls,
                "denied": _to_int(_extract_value(message, "denied")),
                "failed": _to_int(_extract_value(message, "failed")),
                "termination_ack": _to_int(_extract_value(message, "termination_ack")),
                "budget_exhausted": _to_int(_extract_value(message, "budget_exhausted")),
            }

        task_entry: Optional[Dict[str, Any]] = None
        if task_id:
            task_entry = tasks.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "started_at": None,
                    "completed_at": None,
                    "terminated": None,
                    "artifact_valid": None,
                    "artifact_reason": None,
                    "artifact_bytes": None,
                    "artifact_log_level": None,
                    "invalid_submit_attempts": 0,
                    "last_invalid_submit_reason": None,
                },
            )

        if task_entry is not None:
            if " task_start " in f" {message} ":
                task_entry["started_at"] = task_entry["started_at"] or timestamp_text
            artifact_bytes = _to_int(_extract_value(message, "artifact_bytes"))
            if artifact_bytes is not None:
                task_entry["artifact_bytes"] = artifact_bytes
                task_entry["artifact_log_level"] = level
            terminated = _to_bool(_extract_value(message, "terminated"))
            artifact_valid = _to_bool(_extract_value(message, "artifact_valid"))
            artifact_reason = _extract_value(message, "artifact_reason")
            invalid_submit_attempts = _to_int(_extract_value(message, "invalid_submit_attempts"))
            last_invalid_submit_reason = _extract_value(message, "last_invalid_submit_reason")
            if terminated is not None or artifact_valid is not None or artifact_reason is not None:
                task_entry["completed_at"] = timestamp_text
                if terminated is not None:
                    task_entry["terminated"] = terminated
                if artifact_valid is not None:
                    task_entry["artifact_valid"] = artifact_valid
                if artifact_reason is not None:
                    task_entry["artifact_reason"] = artifact_reason
            if invalid_submit_attempts is not None:
                task_entry["invalid_submit_attempts"] = invalid_submit_attempts
            if last_invalid_submit_reason is not None:
                task_entry["last_invalid_submit_reason"] = (
                    None if last_invalid_submit_reason == "none" else last_invalid_submit_reason
                )

        event_name = _event_name(message)
        if event_name == "api_request":
            api_counters["requests"] += 1
        elif event_name == "api_response":
            api_counters["responses"] += 1
            status_code = _extract_value(message, "status_code")
            if status_code:
                api_status_codes[status_code] += 1
                if status_code.startswith("2"):
                    api_counters["responses_success"] += 1
            latency_ms = _to_int(_extract_value(message, "latency_ms"))
            if latency_ms is not None:
                response_latencies_ms.append(latency_ms)
            preview = _extract_trailing_value(message, "body_preview")
            if preview:
                legacy_response_previews.append(preview)
        elif event_name == "api_error":
            api_counters["errors"] += 1
            kind = _extract_value(message, "kind")
            if kind:
                api_error_kinds[kind] += 1
        elif event_name == "api_retry":
            api_counters["retries"] += 1
            reason = _extract_value(message, "reason")
            if reason:
                api_retry_reasons[reason] += 1
        elif event_name == "api_parsed":
            api_counters["parsed"] += 1
        elif event_name == "api_result":
            api_counters["results"] += 1
            assistant_chars_total += _to_int(_extract_value(message, "assistant_chars")) or 0
            tool_calls_total += _to_int(_extract_value(message, "tool_calls")) or 0
        elif event_name == "api_usage":
            api_counters["usage_events"] += 1
            usage_rows.append(
                {
                    "response_id": _extract_value(message, "response_id"),
                    "prompt_tokens": _to_int(_extract_value(message, "prompt_tokens")),
                    "completion_tokens": _to_int(_extract_value(message, "completion_tokens")),
                    "total_tokens": _to_int(_extract_value(message, "total_tokens")),
                    "cost_usd": _to_float(_extract_value(message, "cost_usd")),
                    "is_byok": _to_bool(_extract_value(message, "is_byok")),
                }
            )

    if not usage_rows and legacy_response_previews:
        legacy_parsed = 0
        for preview in legacy_response_previews:
            usage = _parse_legacy_usage_from_api_response_preview(preview)
            if usage is None:
                if TRUNCATED_MARKER in preview:
                    warnings.append("legacy cost fallback skipped truncated api_response body_preview")
                continue
            usage_rows.append(
                {
                    "response_id": usage.get("response_id"),
                    "prompt_tokens": _to_int(str(usage["prompt_tokens"])) if usage.get("prompt_tokens") is not None else None,
                    "completion_tokens": _to_int(str(usage["completion_tokens"])) if usage.get("completion_tokens") is not None else None,
                    "total_tokens": _to_int(str(usage["total_tokens"])) if usage.get("total_tokens") is not None else None,
                    "cost_usd": _to_float(str(usage["cost_usd"])) if usage.get("cost_usd") is not None else None,
                    "is_byok": usage.get("is_byok"),
                }
            )
            legacy_parsed += 1
        if legacy_parsed > 0:
            warnings.append("openrouter cost estimated via legacy api_response body_preview fallback")

    if in_generated_summary_block:
        warnings.append("unterminated generated post-run summary block found in run.log")

    duration_s: Optional[int] = None
    if first_ts is not None and last_ts is not None:
        duration_s = int((last_ts - first_ts).total_seconds())

    per_task = sorted(tasks.values(), key=lambda item: item["task_id"])
    completed_count = sum(1 for item in per_task if item.get("completed_at"))
    started_count = sum(1 for item in per_task if item.get("started_at"))
    artifact_valid_true = 0
    artifact_valid_false = 0
    artifact_valid_unknown = 0
    terminated_true = 0
    terminated_false = 0
    invalid_submit_attempts_total = 0
    tasks_with_invalid_submit = 0

    for item in per_task:
        if item.get("artifact_valid") is True:
            artifact_valid_true += 1
        elif item.get("artifact_valid") is False:
            artifact_valid_false += 1
        else:
            artifact_valid_unknown += 1
        if item.get("terminated") is True:
            terminated_true += 1
        elif item.get("terminated") is False:
            terminated_false += 1
        reason = item.get("artifact_reason")
        if isinstance(reason, str) and reason:
            artifact_reason_counts[reason] += 1
        attempts = item.get("invalid_submit_attempts")
        if isinstance(attempts, int):
            invalid_submit_attempts_total += max(0, attempts)
            if attempts > 0:
                tasks_with_invalid_submit += 1

    prompt_tokens_total = sum(item["prompt_tokens"] or 0 for item in usage_rows)
    completion_tokens_total = sum(item["completion_tokens"] or 0 for item in usage_rows)
    total_tokens_total = sum(item["total_tokens"] or 0 for item in usage_rows)
    cost_values = [item["cost_usd"] for item in usage_rows if isinstance(item.get("cost_usd"), float)]
    cost_usd_total = round(sum(cost_values), 12) if cost_values else None

    usage_source = "api_usage" if api_counters["usage_events"] > 0 else ("legacy_body_preview" if usage_rows else "none")
    responses_total_for_coverage = api_counters["responses_success"]
    usage_covered_responses = len(usage_rows)
    missing_usage_responses = max(0, responses_total_for_coverage - usage_covered_responses)
    if missing_usage_responses > 0:
        warnings.append(
            "missing usage for "
            f"{missing_usage_responses}/{responses_total_for_coverage} successful api_response events"
        )

    telemetry_path = run_root / "tool_telemetry.jsonl"
    tool_quality_manifest = manifest.get("tool_quality")
    if isinstance(tool_quality_manifest, dict):
        manifest_telemetry = tool_quality_manifest.get("telemetry_path")
        if isinstance(manifest_telemetry, str) and manifest_telemetry:
            telemetry_path = Path(manifest_telemetry)
    tools_summary = _summarize_tool_telemetry(telemetry_path, warnings)

    if run_summary_line_counts:
        if (
            run_summary_line_counts.get("valid_artifacts") is not None
            and run_summary_line_counts["valid_artifacts"] != artifact_valid_true
        ):
            warnings.append(
                "run.log valid_artifacts summary does not match parsed per-task artifact_valid counts"
            )
        if (
            run_summary_line_counts.get("invalid_artifacts") is not None
            and run_summary_line_counts["invalid_artifacts"] != artifact_valid_false
        ):
            warnings.append(
                "run.log invalid_artifacts summary does not match parsed per-task artifact_valid counts"
            )

    summary_status = "success"
    if not per_task and api_counters["requests"] == 0 and api_counters["responses"] == 0:
        warnings.append("no recognizable task/api events found in run.log")
        summary_status = "failed"
    elif warnings:
        summary_status = "partial"

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "status": summary_status,
        "generated_at": now_iso(),
        "run_log_path": str(run_log_path.resolve()),
        "started_at": started_at_text,
        "ended_at": ended_at_text,
        "duration_s": duration_s,
        "run": {
            "benchmark": run_meta.get("benchmark"),
            "split": run_meta.get("split"),
            "mode": run_meta.get("mode"),
            "tasks_declared": _to_int(run_meta.get("tasks")) if isinstance(run_meta.get("tasks"), str) else None,
            "model": run_meta.get("model"),
            "agent_profile": run_meta.get("agent_profile"),
        },
        "tasks": {
            "started_count": started_count,
            "completed_count": completed_count,
            "terminated_true": terminated_true,
            "terminated_false": terminated_false,
            "artifact_valid_true": artifact_valid_true,
            "artifact_valid_false": artifact_valid_false,
            "artifact_valid_unknown": artifact_valid_unknown,
            "artifact_reason_counts": dict(sorted(artifact_reason_counts.items())),
            "invalid_submit_attempts_total": invalid_submit_attempts_total,
            "tasks_with_invalid_submit": tasks_with_invalid_submit,
            "per_task": per_task,
        },
        "api": {
            **api_counters,
            "status_codes": dict(sorted(api_status_codes.items())),
            "error_kinds": dict(sorted(api_error_kinds.items())),
            "retry_reasons": dict(sorted(api_retry_reasons.items())),
            "assistant_chars_total": assistant_chars_total,
            "tool_calls_total": tool_calls_total,
            "latency_ms": _latency_summary(response_latencies_ms),
        },
        "openrouter_cost": {
            "source": usage_source,
            "cost_usd_total": cost_usd_total,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "total_tokens_total": total_tokens_total,
            "responses_total": responses_total_for_coverage,
            "usage_covered_responses": usage_covered_responses,
            "missing_usage_responses": missing_usage_responses,
            "coverage_ratio": _format_ratio(usage_covered_responses, responses_total_for_coverage),
        },
        "tool_quality": tool_quality_summary,
        "tools": tools_summary,
        "warnings": warnings,
    }

    terminal_lines = _format_terminal_lines(summary, manifest)

    existing_manifest = manifest or {}
    now = now_iso()
    if not existing_manifest:
        existing_manifest = {
            "run_id": run_id,
            "created_at": now,
        }
    existing_manifest["run_id"] = run_id
    existing_manifest["updated_at"] = now
    existing_manifest["run_log_summary"] = summary
    write_manifest(out_manifest_path, existing_manifest)
    _append_summary_block_to_run_log(
        run_log_path=run_log_path,
        terminal_lines=terminal_lines,
        status=summary_status,
    )

    return RunLogSummaryOutcome(
        run_id=run_id,
        run_root=run_root,
        run_log_path=run_log_path,
        manifest_path=out_manifest_path,
        summary=summary,
        terminal_lines=terminal_lines,
    )
