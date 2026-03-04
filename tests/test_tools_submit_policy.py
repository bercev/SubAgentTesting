from __future__ import annotations

from pathlib import Path

import pytest

from runtime.tools import ToolContext, ToolRegistry


def _make_registry(
    *,
    policy: str,
    expected_output_type: str = "patch",
    max_invalid_submit_attempts: int = 3,
):
    captured: dict[str, str] = {}

    def _submit_cb(artifact: str) -> None:
        captured["artifact"] = artifact

    registry = ToolRegistry(
        ToolContext(
            workspace_root=Path("."),
            submit_callback=_submit_cb,
            expected_output_type=expected_output_type,
            patch_submit_policy=policy,
            max_invalid_submit_attempts=max_invalid_submit_attempts,
        )
    )
    return registry, captured


def test_submit_patch_allow_policy_accepts_empty_artifact():
    registry, captured = _make_registry(policy="allow")

    result = registry.submit("")

    assert result["submitted"] is True
    assert result["invalid_submission_reason"] == "empty_output"
    assert captured["artifact"] == ""
    assert registry.ctx.invalid_submit_attempts == 0


def test_submit_patch_reject_retry_policy_retries_then_exhausts():
    registry, captured = _make_registry(policy="reject_retry", max_invalid_submit_attempts=2)

    first = registry.submit("")
    second = registry.submit("")

    assert "error" in first
    assert first["retryable"] is True
    assert first["invalid_submission_reason"] == "empty_output"
    assert first["invalid_submit_attempts"] == 1

    assert "error" in second
    assert second["retryable"] is False
    assert second["invalid_submit_attempts"] == 2
    assert second["invalid_submission_terminal_reason"] == "invalid_submission_retries_exhausted"
    assert "artifact" not in captured


def test_submit_patch_reject_fast_policy_stops_immediately():
    registry, captured = _make_registry(policy="reject_fast")

    result = registry.submit("")

    assert "error" in result
    assert result["retryable"] is False
    assert result["invalid_submission_terminal_reason"] == "invalid_submission_fast_fail"
    assert "artifact" not in captured


def test_submit_patch_reject_retry_accepts_valid_diff_and_normalizes():
    registry, captured = _make_registry(policy="reject_retry")
    raw = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-a\n+b"

    result = registry.submit(raw)

    assert result["submitted"] is True
    assert captured["artifact"].startswith("diff --git a/a b/a")
    assert captured["artifact"].endswith("\n")
    assert registry.ctx.invalid_submit_attempts == 0


@pytest.mark.parametrize("policy", ["allow", "reject_retry", "reject_fast"])
def test_submit_patch_accepts_cannot_produce_output_sentinel_across_policies(policy: str):
    registry, captured = _make_registry(policy=policy)
    sentinel = "CANNOT PRODUCE OUTPUT {missing reproduction and failing test context}"

    result = registry.submit(sentinel)

    assert result["submitted"] is True
    assert result["submission_reason"] == "cannot_produce_output"
    assert result["artifact_valid"] is False
    assert captured["artifact"] == sentinel
    assert registry.ctx.invalid_submit_attempts == 0


def test_submit_patch_reject_retry_still_rejects_arbitrary_non_diff_text():
    registry, captured = _make_registry(policy="reject_retry")

    result = registry.submit("I could not figure this out")

    assert "error" in result
    assert result["invalid_submission_reason"] == "no_diff_found"
    assert "artifact" not in captured


def test_submit_non_patch_output_is_unconditionally_accepted():
    registry, captured = _make_registry(policy="reject_retry", expected_output_type="text")

    result = registry.submit("")

    assert result["submitted"] is True
    assert captured["artifact"] == ""
