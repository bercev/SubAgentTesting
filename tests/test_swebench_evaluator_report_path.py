import json
import subprocess
from pathlib import Path

from benchmarks.swebench_verified.evaluator import SWEbenchEvaluator


def test_run_harness_relocates_summary_report_to_report_dir(monkeypatch, tmp_path: Path):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True)
    eval_root = tmp_path / "external" / "SWE-bench"
    eval_root.mkdir(parents=True)
    report_dir = tmp_path / "logs" / "reports"
    report_dir.mkdir(parents=True)

    predictions_path = tmp_path / "predictions.jsonl"
    record = {
        "instance_id": "astropy__astropy-12907",
        "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n",
        "model_name_or_path": "qwen/qwen3-coder:free",
        "model_name": "qwen3-coder-free",
        "repo": "astropy/astropy",
    }
    predictions_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    evaluator = SWEbenchEvaluator(
        eval_root=eval_root,
        harness_cmd="python -m swebench.harness.run_evaluation",
        workdir=workdir,
        report_dir=report_dir,
    )
    run_id = "2026-02-12_165543"
    source_name = f"qwen__qwen3-coder:free.{run_id}.json"
    source_path = workdir / source_name

    def _fake_run(*args, **kwargs):
        source_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args[0] if args else "",
            returncode=0,
            stdout=f"Report written to {source_name}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    proc = evaluator.run_harness(
        predictions_path=predictions_path,
        dataset_name="SWE-bench/SWE-bench_Verified",
        split="test",
        run_id=run_id,
    )

    relocated = report_dir / source_name
    assert source_path.exists() is False
    assert relocated.exists() is True
    assert evaluator.last_summary_report == relocated.resolve()
    assert "Report relocated to" in proc.stdout
