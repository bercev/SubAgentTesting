import json
from pathlib import Path

import pytest

import scripts.cli as cli
from agents.spec_loader import AgentSpec
from runtime.schemas import AgentResult, BenchmarkTask


class _FakeAdapter:
    def __init__(self, task: BenchmarkTask):
        self._task = task

    def load_tasks(self, split: str, selector: int | None = None):
        return [self._task]

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path:
        return Path(".")

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: dict | None = None,
    ) -> dict:
        return {
            "instance_id": task.task_id,
            "model_patch": artifact,
            "model_name_or_path": model_name_or_path,
            "model_name": model_name,
            "repo": (metadata or {}).get("repo"),
        }


def _run_once(monkeypatch, tmp_path: Path, raw_artifact: str, *, verbose: bool = False):
    task = BenchmarkTask(
        task_id="astropy__astropy-12907",
        instruction="Fix issue",
        resources={"repo": "astropy/astropy"},
        expected_output_type="patch",
    )
    adapter = _FakeAdapter(task=task)

    run_config = {
        "benchmark": {
            "name": "swebench_verified",
            "dataset_name": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "data_source": "hf",
            "data_root": None,
        },
        "evaluation": {
            "enabled": False,
            "harness_cmd": "python -m swebench.harness.run_evaluation",
            "eval_root": "./external/SWE-bench",
            "workdir": ".",
            "report_dir": "logs/reports",
        },
        "runtime": {
            "mode": "patch_only",
            "selector": 1,
            "max_tool_calls": 1,
            "max_wall_time_s": 10,
        },
        "output": {
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    }

    spec = AgentSpec(
        name="fake-agent",
        backend={"type": "openrouter", "model": "fake/model"},
        prompt_template="Prompt",
        tools=[],
        skills=[],
        tool_to_skill_map={},
        termination={"tool": "submit", "output_type": "patch"},
        decoding_defaults={},
    )

    class _FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, task, prompt, tool_schemas, decoding_defaults=None):
            return AgentResult(
                task_id=task.task_id,
                final_artifact=raw_artifact,
                metadata={"terminated": True, "repo": "astropy/astropy"},
            )

    monkeypatch.setattr(cli, "_load_run_config", lambda _: run_config)
    monkeypatch.setattr(cli, "_build_adapter_from_config", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(cli.AgentSpecLoader, "load", lambda *_args, **_kwargs: (spec, "Prompt", set()))
    monkeypatch.setattr(cli, "_build_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "AgentRuntime", _FakeRuntime)

    cli.run(
        agent="agents/qwen3_coder_free.yaml",
        benchmark=None,
        split=None,
        selector=None,
        mode="patch_only",
        run_config="ignored.yaml",
        verbose=verbose,
    )
    out_files = list((tmp_path / "artifacts").glob("*/predictions.jsonl"))
    assert len(out_files) == 1
    records = [json.loads(line) for line in out_files[0].read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    run_root = out_files[0].parent
    manifest_path = run_root / "manifest.json"
    run_log_path = run_root / "run.log"
    assert manifest_path.exists()
    assert run_log_path.exists()
    return records[0], out_files[0], manifest_path, run_log_path


def test_cli_writes_empty_patch_for_invalid_patch_output(monkeypatch, tmp_path: Path):
    record, _, _, _ = _run_once(monkeypatch, tmp_path, raw_artifact="I'll inspect files first.")
    assert record["instance_id"] == "astropy__astropy-12907"
    assert record["model_patch"] == ""


def test_cli_preserves_valid_patch_output(monkeypatch, tmp_path: Path):
    valid_patch = (
        "diff --git a/example.py b/example.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    record, _, _, _ = _run_once(monkeypatch, tmp_path, raw_artifact=valid_patch)
    assert record["model_patch"] == valid_patch


def test_cli_creates_manifest(monkeypatch, tmp_path: Path):
    record, predictions_path, manifest_path, _ = _run_once(monkeypatch, tmp_path, raw_artifact="")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == manifest_path.parent.name
    assert payload["predictions_path"] == str(predictions_path.resolve())
    assert payload["model_name"] == "fake-agent"
    assert payload["model_name_or_path"] == "fake/model"
    assert payload["evaluation"]["status"] == "not_run"
    assert payload["benchmark_name"] == "swebench_verified"
    assert payload["dataset_name"] == "SWE-bench/SWE-bench_Verified"
    assert payload["split"] == "test"
    assert payload["mode"] == "patch_only"
    assert record["instance_id"] == "astropy__astropy-12907"


def test_cli_quiet_default_suppresses_per_task_logs(monkeypatch, tmp_path: Path, capsys):
    _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    out = capsys.readouterr().out
    assert "artifact_valid" not in out
    assert "Predictions written to" in out


def test_cli_writes_run_log_file(monkeypatch, tmp_path: Path):
    _, _, _, run_log_path = _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    content = run_log_path.read_text(encoding="utf-8")
    assert "Starting run:" in content
    assert "Run summary:" in content


def test_derive_run_id_from_artifacts_path(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    p = artifacts_dir / run_id / "predictions.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    assert cli._derive_run_id_from_predictions(p, artifacts_dir) == run_id


def test_derive_run_id_rejects_non_canonical_layout(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    p = artifacts_dir / "2026-02-13_010203" / "predictions" / "model" / "test" / "patch_only" / "predictions.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        cli._derive_run_id_from_predictions(p, artifacts_dir)
