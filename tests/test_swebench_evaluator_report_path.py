import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.swebench_verified.evaluator import SWEbenchEvaluator
from runtime.config_loader import normalize_run_config


def _run_config(artifacts_dir: Path, eval_root: Path, workdir: Path):
    return normalize_run_config(
        {
            "benchmark": {
                "name": "swebench_verified",
                "dataset_name": "SWE-bench/SWE-bench_Verified",
                "split": "test",
                "data_source": "hf",
            },
            "evaluation": {
                "harness_cmd": "python -m swebench.harness.run_evaluation",
                "eval_root": str(eval_root),
                "workdir": str(workdir),
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


def test_run_harness_relocates_summary_report_to_report_dir(monkeypatch, tmp_path: Path):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True)
    eval_root = tmp_path / "external" / "SWE-bench"
    eval_root.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    predictions_path = tmp_path / "predictions.jsonl"
    record = {
        "instance_id": "astropy__astropy-12907",
        "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
        "model_name_or_path": "qwen/qwen3-coder:free",
        "model_name": "qwen3-coder-free",
        "repo": "astropy/astropy",
    }
    predictions_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cfg = _run_config(artifacts_dir=artifacts_dir, eval_root=eval_root, workdir=workdir)
    evaluator = SWEbenchEvaluator()
    run_id = "2026-02-12_165543"
    source_name = f"qwen__qwen3-coder:free.{run_id}.json"
    source_path = workdir / source_name
    source_harness_logs = (
        workdir
        / "logs"
        / "run_evaluation"
        / run_id
        / "qwen__qwen3-coder:free"
        / "astropy__astropy-12907"
    )

    def _fake_run(*args, **kwargs):
        source_path.write_text("{}", encoding="utf-8")
        source_harness_logs.mkdir(parents=True, exist_ok=True)
        (source_harness_logs / "run_instance.log").write_text("hello", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args[0] if args else "",
            returncode=0,
            stdout=f"Report written to {source_name}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    proc = evaluator.run_harness(predictions_path=predictions_path, run_id=run_id, config=cfg)

    relocated = artifacts_dir / run_id / "report.json"
    relocated_harness = artifacts_dir / run_id / "evaluation"
    relocated_instance = relocated_harness / "astropy__astropy-12907"
    assert source_path.exists() is False
    assert relocated.exists() is True
    assert evaluator.last_summary_report == relocated.resolve()
    assert source_harness_logs.exists() is False
    assert relocated_harness.exists() is True
    assert relocated_instance.exists() is True
    assert (artifacts_dir / run_id / "evaluation" / "harness").exists() is False
    assert evaluator.last_harness_log_root == relocated_harness.resolve()
    assert "Report relocated to" in proc.stdout
    assert "Harness logs relocated to" in proc.stdout


def test_run_harness_relocates_with_instance_collision(monkeypatch, tmp_path: Path):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True)
    eval_root = tmp_path / "external" / "SWE-bench"
    eval_root.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "astropy__astropy-12907",
                "model_patch": "",
                "model_name_or_path": "qwen/qwen3-coder:free",
                "model_name": "qwen3-coder-free",
                "repo": "astropy/astropy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = _run_config(artifacts_dir=artifacts_dir, eval_root=eval_root, workdir=workdir)
    evaluator = SWEbenchEvaluator()
    run_id = "2026-02-12_165544"
    source_name = f"qwen__qwen3-coder:free.{run_id}.json"
    source_path = workdir / source_name

    model_a = workdir / "logs" / "run_evaluation" / run_id / "model-a" / "same-instance"
    model_b = workdir / "logs" / "run_evaluation" / run_id / "model-b" / "same-instance"

    def _fake_run(*args, **kwargs):
        source_path.write_text("{}", encoding="utf-8")
        model_a.mkdir(parents=True, exist_ok=True)
        model_b.mkdir(parents=True, exist_ok=True)
        (model_a / "run_instance.log").write_text("a", encoding="utf-8")
        (model_b / "run_instance.log").write_text("b", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args[0] if args else "",
            returncode=0,
            stdout=f"Report written to {source_name}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    evaluator.run_harness(predictions_path=predictions_path, run_id=run_id, config=cfg)

    dest_root = artifacts_dir / run_id / "evaluation"
    assert (dest_root / "same-instance").exists()
    assert (dest_root / "same-instance__model-b").exists()


def test_run_harness_fails_when_model_name_or_path_missing(tmp_path: Path):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True)
    eval_root = tmp_path / "external" / "SWE-bench"
    eval_root.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "astropy__astropy-12907",
                "model_patch": "",
                "model_name": "openrouter-free",
                "repo": "astropy/astropy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = _run_config(artifacts_dir=artifacts_dir, eval_root=eval_root, workdir=workdir)
    evaluator = SWEbenchEvaluator()
    with pytest.raises(ValueError, match="model_name_or_path"):
        evaluator.run_harness(
            predictions_path=predictions_path,
            run_id="2026-02-13_010203",
            config=cfg,
        )
