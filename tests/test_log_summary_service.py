import json
from pathlib import Path

from runtime.log_summary_service import execute_run_log_summary


def _log_line(ts: str, message: str, *, level: str = "INFO", source: str = "run_service.py:1") -> str:
    return f"{ts} | {level:<8} | {source:<24} | {message}"


def test_execute_run_log_summary_aggregates_api_usage_and_preserves_manifest(tmp_path: Path):
    run_id = "2026-02-24_120000"
    run_root = tmp_path / "artifacts" / run_id
    run_root.mkdir(parents=True)
    run_log_path = run_root / "run.log"
    manifest_path = run_root / "manifest.json"
    telemetry_path = run_root / "tool_telemetry.jsonl"

    telemetry_rows = [
        {
            "row_type": "tool_call",
            "tool_name": "workspace_open",
            "success": True,
            "latency_ms": 4,
        },
        {
            "row_type": "task_summary",
            "task_id": "task-1",
        },
    ]
    telemetry_path.write_text("\n".join(json.dumps(row) for row in telemetry_rows) + "\n", encoding="utf-8")

    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tool_quality": {
                    "score": 0.5,
                    "telemetry_path": str(telemetry_path.resolve()),
                },
                "evaluation": {
                    "status": "not_run",
                    "metrics": {"resolved_instances": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    lines = [
        _log_line(
            "2026-02-24 12:00:00",
            (
                "Starting run: run_id=2026-02-24_120000 benchmark=swebench_verified split=test "
                "mode=tools_enabled tasks=1 model=openrouter/free agent_profile=profiles/agents/openrouter_free.yaml"
            ),
        ),
        _log_line(
            "2026-02-24 12:00:01",
            "task=task-1 task_start workspace_root=.",
        ),
        _log_line(
            "2026-02-24 12:00:02",
            "task=task-1 api_request provider=openrouter model=openrouter/free attempt=1/1 method=POST url=https://openrouter.ai/api/v1/chat/completions payload_bytes=100 payload_preview={}",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:03",
            "task=task-1 api_response provider=openrouter model=openrouter/free attempt=1/1 status_code=200 latency_ms=123 body_bytes=111 body_preview={\"id\":\"r1\"}",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:03",
            "task=task-1 api_usage provider=openrouter model=openrouter/free attempt=1/1 response_id=r1 prompt_tokens=10 completion_tokens=5 total_tokens=15 cost_usd=0.0123 is_byok=False",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:03",
            "task=task-1 api_result provider=openrouter model=openrouter/free assistant_chars=20 tool_calls=1",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:04",
            "task=task-1 artifact_bytes=42 artifact_preview=diff --git a/x b/x",
        ),
        _log_line(
            "2026-02-24 12:00:04",
            "task=task-1 terminated=True output_type=patch artifact_valid=True artifact_reason=ok",
        ),
        _log_line(
            "2026-02-24 12:00:05",
            "tool_quality summary applicable=True score=0.8 tasks_applicable=1/1 calls=1/1 denied=0 failed=0 termination_ack=1 budget_exhausted=0",
        ),
        _log_line(
            "2026-02-24 12:00:05",
            "Run summary: run_id=2026-02-24_120000 tasks=1 valid_artifacts=1 invalid_artifacts=0",
            level="WARNING",
        ),
    ]
    run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outcome = execute_run_log_summary(run_log_path=run_log_path)

    assert outcome.run_id == run_id
    assert any(line.startswith("Post-run summary:") for line in outcome.terminal_lines)
    assert any("OpenRouter cost:" in line for line in outcome.terminal_lines)
    assert outcome.summary["tasks"]["artifact_valid_true"] == 1
    assert outcome.summary["api"]["requests"] == 1
    assert outcome.summary["api"]["responses"] == 1
    assert outcome.summary["api"]["usage_events"] == 1
    assert outcome.summary["openrouter_cost"]["source"] == "api_usage"
    assert outcome.summary["openrouter_cost"]["cost_usd_total"] == 0.0123
    assert outcome.summary["openrouter_cost"]["prompt_tokens_total"] == 10
    assert outcome.summary["tools"]["telemetry_present"] is True

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tool_quality"]["score"] == 0.5
    assert manifest["evaluation"]["status"] == "not_run"
    assert manifest["run_log_summary"]["run_id"] == run_id
    assert manifest["run_log_summary"]["openrouter_cost"]["cost_usd_total"] == 0.0123
    updated_log = run_log_path.read_text(encoding="utf-8")
    assert "post_run_summary_block_begin version=v1" in updated_log
    assert "post_run_summary_line index=1 text=Post-run summary:" in updated_log
    assert "post_run_summary_block_end version=v1" in updated_log


def test_execute_run_log_summary_legacy_cost_fallback_from_api_response_preview(tmp_path: Path):
    run_id = "2026-02-24_120001"
    run_root = tmp_path / "artifacts" / run_id
    run_root.mkdir(parents=True)
    run_log_path = run_root / "run.log"

    preview_payload = {
        "id": "r-legacy",
        "provider": "TestProvider",
        "model": "test/model",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.5,
            "is_byok": False,
        },
    }
    preview_json = json.dumps(preview_payload, separators=(",", ":"))

    lines = [
        _log_line(
            "2026-02-24 12:00:00",
            "Starting run: run_id=2026-02-24_120001 benchmark=swebench_verified split=test mode=patch_only tasks=1 model=openrouter/free agent_profile=profiles/agents/openrouter_free.yaml",
        ),
        _log_line("2026-02-24 12:00:01", "task=task-legacy task_start workspace_root=."),
        _log_line(
            "2026-02-24 12:00:02",
            f"task=task-legacy api_response provider=openrouter model=openrouter/free attempt=1/1 status_code=200 latency_ms=10 body_bytes=200 body_preview={preview_json}",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:02",
            "task=task-legacy api_response provider=openrouter model=openrouter/free attempt=1/1 status_code=200 latency_ms=12 body_bytes=201 body_preview={\"usage\":{\"cost\":0.1}}...[truncated]",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:03",
            "task=task-legacy terminated=True output_type=patch artifact_valid=False artifact_reason=no_diff_found",
        ),
        _log_line(
            "2026-02-24 12:00:04",
            "Run summary: run_id=2026-02-24_120001 tasks=1 valid_artifacts=0 invalid_artifacts=1",
        ),
    ]
    run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outcome = execute_run_log_summary(run_log_path=run_log_path)

    cost = outcome.summary["openrouter_cost"]
    assert cost["source"] == "legacy_body_preview"
    assert cost["cost_usd_total"] == 0.5
    assert cost["prompt_tokens_total"] == 100
    assert cost["usage_covered_responses"] == 1
    assert cost["responses_total"] == 2
    assert any("legacy" in warning for warning in outcome.summary["warnings"])
    assert any("truncated" in warning for warning in outcome.summary["warnings"])


def test_execute_run_log_summary_repeated_runs_append_blocks_but_keep_metrics_stable(tmp_path: Path):
    run_id = "2026-02-24_120002"
    run_root = tmp_path / "artifacts" / run_id
    run_root.mkdir(parents=True)
    run_log_path = run_root / "run.log"

    lines = [
        _log_line(
            "2026-02-24 12:00:00",
            "Starting run: run_id=2026-02-24_120002 benchmark=swebench_verified split=test mode=patch_only tasks=1 model=openrouter/free agent_profile=profiles/agents/openrouter_free.yaml",
        ),
        _log_line("2026-02-24 12:00:01", "task=task-1 task_start workspace_root=."),
        _log_line(
            "2026-02-24 12:00:02",
            "task=task-1 api_request provider=openrouter model=openrouter/free attempt=1/1 method=POST url=https://openrouter.ai payload_bytes=10 payload_preview={}",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:03",
            "task=task-1 api_response provider=openrouter model=openrouter/free attempt=1/1 status_code=200 latency_ms=50 body_bytes=10 body_preview={\"id\":\"x\"}",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:03",
            "task=task-1 api_usage provider=openrouter model=openrouter/free attempt=1/1 response_id=x prompt_tokens=2 completion_tokens=3 total_tokens=5 cost_usd=0.1 is_byok=False",
            source="model_backend.py",
        ),
        _log_line(
            "2026-02-24 12:00:04",
            "task=task-1 terminated=True output_type=patch artifact_valid=True artifact_reason=ok",
        ),
        _log_line(
            "2026-02-24 12:00:05",
            "Run summary: run_id=2026-02-24_120002 tasks=1 valid_artifacts=1 invalid_artifacts=0",
        ),
    ]
    run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first = execute_run_log_summary(run_log_path=run_log_path)
    second = execute_run_log_summary(run_log_path=run_log_path)

    assert first.summary["duration_s"] == second.summary["duration_s"] == 5
    assert first.summary["api"]["requests"] == second.summary["api"]["requests"] == 1
    assert first.summary["api"]["responses"] == second.summary["api"]["responses"] == 1
    assert first.summary["openrouter_cost"]["cost_usd_total"] == second.summary["openrouter_cost"]["cost_usd_total"] == 0.1

    final_log = run_log_path.read_text(encoding="utf-8")
    assert final_log.count("post_run_summary_block_begin version=v1") == 2
    assert final_log.count("post_run_summary_block_end version=v1") == 2

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert "run_log_summary" in manifest
    assert manifest["run_log_summary"]["run_id"] == run_id
