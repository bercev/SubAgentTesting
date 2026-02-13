import json
import subprocess
from pathlib import Path

import pytest
import typer

import scripts.cli as cli


class _FakeAdapter:
    def __init__(self, evaluator):
        self._evaluator = evaluator

    def get_evaluator(self):
        return self._evaluator


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

    def run_harness(self, predictions_path, dataset_name, split, run_id, artifacts_dir):
        self.last_summary_report = self.report_path
        self.last_harness_log_root = self.harness_log_root
        return subprocess.CompletedProcess(
            args="fake-harness",
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _run_config(artifacts_dir: Path) -> dict:
    return {
        "benchmark": {
            "name": "swebench_verified",
            "dataset_name": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "data_source": "hf",
            "data_root": None,
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


def test_eval_prints_metrics_and_updates_manifest(monkeypatch, tmp_path: Path, capsys):
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
    adapter = _FakeAdapter(evaluator=evaluator)

    monkeypatch.setattr(cli, "_load_run_config", lambda _: _run_config(artifacts_dir))
    monkeypatch.setattr(cli, "_build_adapter_from_config", lambda *_args, **_kwargs: adapter)

    cli.eval(
        predictions=str(predictions_path),
        run_config="ignored.yaml",
        benchmark=None,
        verbose=False,
    )

    out = capsys.readouterr().out
    assert "Metrics: resolved=2/5 accuracy=40.00%" in out
    assert "resolved/completed=50.00%" in out
    assert "completion(submitted)=80.00%" in out

    manifest_path = run_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = payload["evaluation"]["metrics"]
    assert metrics["resolved_instances"] == 2
    assert metrics["submitted_instances"] == 5
    assert metrics["accuracy_resolved_submitted"] == 0.4
    assert metrics["accuracy_resolved_completed"] == 0.5
    assert metrics["completion_rate_submitted"] == 0.8


def test_eval_rejects_non_canonical_predictions_path(monkeypatch, tmp_path: Path):
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
    adapter = _FakeAdapter(evaluator=evaluator)

    monkeypatch.setattr(cli, "_load_run_config", lambda _: _run_config(artifacts_dir))
    monkeypatch.setattr(cli, "_build_adapter_from_config", lambda *_args, **_kwargs: adapter)

    with pytest.raises(typer.BadParameter):
        cli.eval(
            predictions=str(bad_path),
            run_config="ignored.yaml",
            benchmark=None,
            verbose=False,
        )


def test_eval_uses_zero_metrics_when_report_is_malformed(monkeypatch, tmp_path: Path, capsys):
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
    adapter = _FakeAdapter(evaluator=evaluator)

    monkeypatch.setattr(cli, "_load_run_config", lambda _: _run_config(artifacts_dir))
    monkeypatch.setattr(cli, "_build_adapter_from_config", lambda *_args, **_kwargs: adapter)

    cli.eval(
        predictions=str(predictions_path),
        run_config="ignored.yaml",
        benchmark=None,
        verbose=False,
    )

    out = capsys.readouterr().out
    assert "Metrics warning: report_parse_failed" in out
    assert "accuracy=0.00%" in out

    manifest_path = run_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = payload["evaluation"]["metrics"]
    assert metrics["resolved_instances"] == 0
    assert metrics["submitted_instances"] == 0
    assert metrics["accuracy_resolved_submitted"] == 0.0


def test_eval_verbose_default_prints_harness_output(monkeypatch, tmp_path: Path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    run_root = artifacts_dir / run_id
    predictions_path = run_root / "predictions.jsonl"
    report_path = run_root / "report.json"
    harness_log_root = run_root / "evaluation"

    _write_predictions(predictions_path)
    harness_log_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "total_instances": 1,
                "submitted_instances": 1,
                "completed_instances": 1,
                "resolved_instances": 0,
                "unresolved_instances": 1,
                "empty_patch_instances": 0,
                "error_instances": 0,
            }
        ),
        encoding="utf-8",
    )

    evaluator = _FakeEvaluator(
        report_path=report_path,
        harness_log_root=harness_log_root,
        stdout="HARNESS_STDOUT",
        stderr="HARNESS_STDERR",
    )
    adapter = _FakeAdapter(evaluator=evaluator)

    monkeypatch.setattr(cli, "_load_run_config", lambda _: _run_config(artifacts_dir))
    monkeypatch.setattr(cli, "_build_adapter_from_config", lambda *_args, **_kwargs: adapter)

    cli.eval(
        predictions=str(predictions_path),
        run_config="ignored.yaml",
        benchmark=None,
    )

    out = capsys.readouterr().out
    assert "HARNESS_STDOUT" in out
    assert "HARNESS_STDERR" in out


def test_eval_quiet_suppresses_harness_output_on_success(monkeypatch, tmp_path: Path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    run_root = artifacts_dir / run_id
    predictions_path = run_root / "predictions.jsonl"
    report_path = run_root / "report.json"
    harness_log_root = run_root / "evaluation"

    _write_predictions(predictions_path)
    harness_log_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "total_instances": 1,
                "submitted_instances": 1,
                "completed_instances": 1,
                "resolved_instances": 0,
                "unresolved_instances": 1,
                "empty_patch_instances": 0,
                "error_instances": 0,
            }
        ),
        encoding="utf-8",
    )

    evaluator = _FakeEvaluator(
        report_path=report_path,
        harness_log_root=harness_log_root,
        stdout="HARNESS_STDOUT",
        stderr="HARNESS_STDERR",
    )
    adapter = _FakeAdapter(evaluator=evaluator)

    monkeypatch.setattr(cli, "_load_run_config", lambda _: _run_config(artifacts_dir))
    monkeypatch.setattr(cli, "_build_adapter_from_config", lambda *_args, **_kwargs: adapter)

    cli.eval(
        predictions=str(predictions_path),
        run_config="ignored.yaml",
        benchmark=None,
        verbose=False,
    )

    out = capsys.readouterr().out
    assert "HARNESS_STDOUT" not in out
    assert "HARNESS_STDERR" not in out
