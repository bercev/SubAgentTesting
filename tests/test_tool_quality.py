from __future__ import annotations

from pathlib import Path

import pytest

from runtime.tool_quality import build_run_summary, build_task_summary, serialize_tool_call_row


def _weights() -> dict[str, float]:
    return {
        "execution_quality": 0.45,
        "policy_quality": 0.25,
        "termination_quality": 0.20,
        "budget_quality": 0.10,
    }


def test_serialize_tool_call_row_preserves_required_fields():
    row = serialize_tool_call_row(
        run_id="2026-02-15_222505",
        task_id="astropy__astropy-12907",
        mode="tools_enabled",
        event={
            "turn_index": 2,
            "call_index": 1,
            "tool_name": "bash",
            "is_termination_tool": False,
            "allowed": True,
            "executed": True,
            "success": False,
            "error_code": "nonzero_returncode",
            "args_size_bytes": 33,
            "result_size_bytes": 100,
            "latency_ms": 41,
            "return_code": 2,
        },
    )
    assert row["row_type"] == "tool_call"
    assert row["run_id"] == "2026-02-15_222505"
    assert row["task_id"] == "astropy__astropy-12907"
    assert row["mode"] == "tools_enabled"
    assert row["tool_name"] == "bash"
    assert row["error_code"] == "nonzero_returncode"
    assert row["return_code"] == 2


def test_task_and_run_scoring_for_mixed_generic_tool_outcomes():
    task_summary_a = build_task_summary(
        run_id="2026-02-15_222505",
        task_id="t1",
        mode="tools_enabled",
        runtime_payload={
            "events": [
                {"success": True, "allowed": True, "error_code": "none"},
                {"success": False, "allowed": False, "error_code": "not_allowed"},
                {"success": False, "allowed": True, "error_code": "nonzero_returncode"},
            ],
            "termination_ack": True,
            "budget_exhausted": False,
            "wall_time_exhausted": False,
            "loop_exit_reason": "submitted",
        },
        weights=_weights(),
        enabled=True,
    )
    task_summary_b = build_task_summary(
        run_id="2026-02-15_222505",
        task_id="t2",
        mode="tools_enabled",
        runtime_payload={
            "events": [
                {"success": True, "allowed": True, "error_code": "none"},
                {"success": False, "allowed": True, "error_code": "tool_error"},
            ],
            "termination_ack": False,
            "budget_exhausted": True,
            "wall_time_exhausted": False,
            "loop_exit_reason": "tool_budget_exhausted",
        },
        weights=_weights(),
        enabled=True,
    )

    assert task_summary_a["tool_quality_applicable"] is True
    assert task_summary_a["tool_calls_total"] == 3
    assert task_summary_a["tool_calls_success"] == 1
    assert task_summary_a["tool_calls_denied"] == 1
    assert task_summary_a["tool_calls_failed"] == 2
    assert task_summary_a["components"]["execution_quality"] == pytest.approx(1 / 3)
    assert task_summary_a["components"]["policy_quality"] == pytest.approx(2 / 3)
    assert task_summary_a["components"]["termination_quality"] == pytest.approx(1.0)
    assert task_summary_a["components"]["budget_quality"] == pytest.approx(1.0)
    assert task_summary_a["score"] == pytest.approx(
        (0.45 * (1 / 3)) + (0.25 * (2 / 3)) + (0.2 * 1.0) + (0.1 * 1.0)
    )

    run_summary = build_run_summary(
        telemetry_path=Path("artifacts/2026-02-15_222505/tool_telemetry.jsonl"),
        task_summaries=[task_summary_a, task_summary_b],
        weights=_weights(),
        enabled=True,
    )
    counts = run_summary["counts"]

    assert run_summary["applicable"] is True
    assert counts["tool_calls_total"] == 5
    assert counts["tool_calls_success"] == 2
    assert counts["tool_calls_failed"] == 3
    assert counts["tool_calls_denied"] == 1
    assert counts["tasks_total"] == 2
    assert counts["tasks_applicable"] == 2
    assert counts["tasks_budget_exhausted"] == 1
    assert counts["tasks_termination_ack"] == 1
    assert run_summary["components"]["execution_quality"] == pytest.approx(0.4)
    assert run_summary["components"]["policy_quality"] == pytest.approx(0.8)
    assert run_summary["components"]["termination_quality"] == pytest.approx(0.5)
    assert run_summary["components"]["budget_quality"] == pytest.approx(0.5)
    assert run_summary["score"] == pytest.approx(0.53)


def test_tool_quality_not_applicable_when_no_tool_calls_or_patch_only():
    no_calls_summary = build_task_summary(
        run_id="2026-02-15_222505",
        task_id="t-no-calls",
        mode="tools_enabled",
        runtime_payload={
            "events": [],
            "termination_ack": False,
            "budget_exhausted": False,
            "wall_time_exhausted": False,
            "loop_exit_reason": "no_tool_calls",
        },
        weights=_weights(),
        enabled=True,
    )
    patch_mode_summary = build_task_summary(
        run_id="2026-02-15_222505",
        task_id="t-patch",
        mode="patch_only",
        runtime_payload={
            "events": [
                {"success": True, "allowed": True, "error_code": "none"},
            ],
            "termination_ack": True,
            "budget_exhausted": False,
            "wall_time_exhausted": False,
            "loop_exit_reason": "submitted",
        },
        weights=_weights(),
        enabled=True,
    )

    assert no_calls_summary["tool_quality_applicable"] is False
    assert no_calls_summary["score"] is None
    assert no_calls_summary["components"]["execution_quality"] is None

    assert patch_mode_summary["tool_quality_applicable"] is False
    assert patch_mode_summary["score"] is None
    assert patch_mode_summary["components"]["policy_quality"] is None

    run_summary = build_run_summary(
        telemetry_path=Path("artifacts/2026-02-15_222505/tool_telemetry.jsonl"),
        task_summaries=[no_calls_summary, patch_mode_summary],
        weights=_weights(),
        enabled=True,
    )
    assert run_summary["applicable"] is False
    assert run_summary["score"] is None
    assert run_summary["components"]["execution_quality"] is None
