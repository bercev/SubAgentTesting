import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import typer

import scripts.cli as cli
import runtime.eval_service as eval_service
from runtime.config_loader import normalize_run_config
from runtime.eval_service import EvalOutcome


class _FakeAdapter:
    benchmark_name = "swebench_verified"

    def __init__(self, evaluator):
        self._evaluator = evaluator

    @classmethod
    def from_config(cls, config):
        return cls(evaluator=config.benchmark.params["evaluator"])

    def get_evaluator(self, config):
        return self._evaluator

    def load_tasks(self, split, selector):
        raise NotImplementedError

    def workspace_root_for_task(self, task):
        raise NotImplementedError

    def to_prediction_record(self, *args, **kwargs):
        raise NotImplementedError


class _FakeRegistry:
    def get_adapter(self, name: str):
        assert name == "swebench_verified"
        return _FakeAdapter


class _FakeEvaluator:
    def __init__(
        self,
        report_path: Path | None,
        harness_log_root: Path | None,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ):
        self.report_path = report_path
        self.harness_log_root = harness_log_root
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.last_summary_report: Path | None = None
        self.last_harness_log_root: Path | None = None

    def run_harness(self, predictions_path, run_id, config):
        self.last_summary_report = self.report_path
        self.last_harness_log_root = self.harness_log_root
        return subprocess.CompletedProcess(
            args="fake-harness",
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _run_config(artifacts_dir: Path, evaluator: _FakeEvaluator) -> Any:
    return normalize_run_config(
        {
            "benchmark": {
                "name": "swebench_verified",
                "dataset_name": "SWE-bench/SWE-bench_Verified",
                "split": "test",
                "data_source": "hf",
                "data_root": None,
                "params": {"evaluator": evaluator},
            },
            "evaluation": {
                "enabled": True,
                "harness_cmd": "python -m swebench.harness.run_evaluation",
                "eval_root": "./external/SWE-bench",
                "workdir": ".",
                "report_dir": "reports",
            },
            "runtime": {
                "mode": "patch_only",
                "selector": 1,
                "max_tool_calls": 1,
                "max_wall_time_s": 10,
            },
            "output": {
                "artifacts_dir": str(artifacts_dir),
            },
        }
    )


def _write_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "instance_id": "astropy__astropy-12907",
        "model_patch": "",
        "model_name_or_path": "openrouter/free",
        "model_name": "openrouter-free",
        "repo": "astropy/astropy",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_execute_eval_updates_manifest(monkeypatch, tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    run_root = artifacts_dir / run_id
    predictions_path = run_root / "predictions.jsonl"
    report_path = run_root / "report.json"
    harness_log_root = run_root / "evaluation"

    _write_predictions(predictions_path)
    harness_log_root.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "total_instances": 500,
        "submitted_instances": 5,
        "completed_instances": 4,
        "resolved_instances": 2,
        "unresolved_instances": 2,
        "empty_patch_instances": 1,
        "error_instances": 1,
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    evaluator = _FakeEvaluator(report_path=report_path, harness_log_root=harness_log_root)
    cfg = _run_config(artifacts_dir, evaluator)
    monkeypatch.setattr(eval_service, "BenchmarkRegistry", lambda: _FakeRegistry())

    outcome = eval_service.execute_eval(
        predictions_path=predictions_path,
        config=cfg,
        benchmark=None,
    )

    assert outcome.run_id == run_id
    assert outcome.metrics["resolved_instances"] == 2
    assert outcome.metrics["submitted_instances"] == 5
    assert outcome.metrics["accuracy_resolved_submitted"] == 0.4
    assert outcome.metrics["accuracy_resolved_completed"] == 0.5
    assert outcome.metrics["completion_rate_submitted"] == 0.8

    payload = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    metrics = payload["evaluation"]["metrics"]
    assert metrics["resolved_instances"] == 2
    assert metrics["submitted_instances"] == 5


def test_execute_eval_rejects_non_canonical_predictions_path(monkeypatch, tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    bad_path = (
        artifacts_dir
        / "2026-02-13_010203"
        / "predictions"
        / "openrouter_free"
        / "test"
        / "patch_only"
        / "predictions.jsonl"
    )
    _write_predictions(bad_path)

    evaluator = _FakeEvaluator(report_path=None, harness_log_root=None)
    cfg = _run_config(artifacts_dir, evaluator)
    monkeypatch.setattr(eval_service, "BenchmarkRegistry", lambda: _FakeRegistry())

    with pytest.raises(ValueError):
        eval_service.execute_eval(
            predictions_path=bad_path,
            config=cfg,
            benchmark=None,
        )


def test_execute_eval_zero_metrics_malformed_report(monkeypatch, tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    run_root = artifacts_dir / run_id
    predictions_path = run_root / "predictions.jsonl"
    report_path = run_root / "report.json"
    harness_log_root = run_root / "evaluation"

    _write_predictions(predictions_path)
    harness_log_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text("not-json", encoding="utf-8")

    evaluator = _FakeEvaluator(report_path=report_path, harness_log_root=harness_log_root)
    cfg = _run_config(artifacts_dir, evaluator)
    monkeypatch.setattr(eval_service, "BenchmarkRegistry", lambda: _FakeRegistry())

    outcome = eval_service.execute_eval(
        predictions_path=predictions_path,
        config=cfg,
        benchmark=None,
    )

    assert outcome.metrics_warning == "report_parse_failed"
    assert outcome.metrics["resolved_instances"] == 0
    assert outcome.metrics["submitted_instances"] == 0
    assert outcome.metrics["accuracy_resolved_submitted"] == 0.0


def test_cli_eval_verbose_default_prints_harness_output(monkeypatch, tmp_path: Path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    run_root = artifacts_dir / run_id
    predictions_path = run_root / "predictions.jsonl"
    manifest_path = run_root / "manifest.json"

    outcome = EvalOutcome(
        run_id=run_id,
        benchmark_name="swebench_verified",
        split_name="test",
        proc=subprocess.CompletedProcess(
            args="fake",
            returncode=0,
            stdout="HARNESS_STDOUT",
            stderr="HARNESS_STDERR",
        ),
        report_path=run_root / "report.json",
        harness_log_root=run_root / "evaluation",
        manifest_path=manifest_path,
        metrics={
            "resolved_instances": 0,
            "submitted_instances": 1,
            "completed_instances": 1,
            "unresolved_instances": 1,
            "error_instances": 0,
            "empty_patch_instances": 0,
            "accuracy_resolved_submitted": 0.0,
            "accuracy_resolved_completed": 0.0,
            "completion_rate_submitted": 1.0,
        },
        metrics_warning=None,
    )

    monkeypatch.setattr(cli, "load_run_config", lambda _: object())
    monkeypatch.setattr(cli, "execute_eval", lambda **_kwargs: outcome)

    cli.eval(predictions=str(predictions_path), run_config="ignored.yaml", benchmark=None)

    out = capsys.readouterr().out
    assert "HARNESS_STDOUT" in out
    assert "HARNESS_STDERR" in out


def test_cli_eval_quiet_suppresses_harness_output_on_success(monkeypatch, tmp_path: Path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    run_root = artifacts_dir / run_id
    predictions_path = run_root / "predictions.jsonl"
    manifest_path = run_root / "manifest.json"

    outcome = EvalOutcome(
        run_id=run_id,
        benchmark_name="swebench_verified",
        split_name="test",
        proc=subprocess.CompletedProcess(
            args="fake",
            returncode=0,
            stdout="HARNESS_STDOUT",
            stderr="HARNESS_STDERR",
        ),
        report_path=run_root / "report.json",
        harness_log_root=run_root / "evaluation",
        manifest_path=manifest_path,
        metrics={
            "resolved_instances": 0,
            "submitted_instances": 1,
            "completed_instances": 1,
            "unresolved_instances": 1,
            "error_instances": 0,
            "empty_patch_instances": 0,
            "accuracy_resolved_submitted": 0.0,
            "accuracy_resolved_completed": 0.0,
            "completion_rate_submitted": 1.0,
        },
        metrics_warning=None,
    )

    monkeypatch.setattr(cli, "load_run_config", lambda _: object())
    monkeypatch.setattr(cli, "execute_eval", lambda **_kwargs: outcome)

    cli.eval(
        predictions=str(predictions_path),
        run_config="ignored.yaml",
        benchmark=None,
        verbose=False,
    )

    out = capsys.readouterr().out
    assert "HARNESS_STDOUT" not in out
    assert "HARNESS_STDERR" not in out


def test_cli_eval_invalid_predictions_path_maps_to_bad_parameter(monkeypatch, tmp_path: Path):
    predictions_path = tmp_path / "bad.jsonl"
    monkeypatch.setattr(cli, "load_run_config", lambda _: object())
    monkeypatch.setattr(
        cli,
        "execute_eval",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad predictions path")),
    )
    with pytest.raises(typer.BadParameter):
        cli.eval(predictions=str(predictions_path), run_config="ignored.yaml", benchmark=None)
