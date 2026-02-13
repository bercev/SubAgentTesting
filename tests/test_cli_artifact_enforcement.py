import json
from pathlib import Path

import pytest

import scripts.cli as cli
import runtime.eval_service as eval_service
import runtime.run_service as run_service
from agents.spec_loader import AgentSpec
from runtime.config_loader import normalize_run_config
from runtime.schemas import AgentResult, BenchmarkTask


class _FakeAdapter:
    benchmark_name = "swebench_verified"

    def __init__(self, task: BenchmarkTask):
        self._task = task

    @classmethod
    def from_config(cls, config):
        task = BenchmarkTask(
            task_id="astropy__astropy-12907",
            instruction="Fix issue",
            resources={"repo": "astropy/astropy"},
            expected_output_type="patch",
        )
        return cls(task=task)

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

    def get_evaluator(self, config):
        raise NotImplementedError


class _FakeRegistry:
    def get_adapter(self, name: str):
        assert name == "swebench_verified"
        return _FakeAdapter


def _run_once(monkeypatch, tmp_path: Path, raw_artifact: str, *, verbose: bool = False):
    run_config = normalize_run_config(
        {
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
                "report_dir": "reports",
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
    )

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

    monkeypatch.setattr(run_service, "BenchmarkRegistry", lambda: _FakeRegistry())
    monkeypatch.setattr(
        run_service.AgentSpecLoader,
        "load",
        lambda *_args, **_kwargs: (spec, "Prompt", set()),
    )
    monkeypatch.setattr(run_service, "build_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_service, "AgentRuntime", _FakeRuntime)

    outcome = run_service.execute_run(
        agent_path="agents/qwen3_coder_free.yaml",
        config=run_config,
        verbose=verbose,
    )

    records = [
        json.loads(line)
        for line in outcome.predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 1
    return records[0], outcome


def test_run_service_preserves_invalid_patch_output(monkeypatch, tmp_path: Path):
    record, _ = _run_once(monkeypatch, tmp_path, raw_artifact="I'll inspect files first.")
    assert record["instance_id"] == "astropy__astropy-12907"
    assert record["model_patch"] == "I'll inspect files first."


def test_run_service_preserves_valid_patch_output(monkeypatch, tmp_path: Path):
    valid_patch = (
        "diff --git a/example.py b/example.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    record, _ = _run_once(monkeypatch, tmp_path, raw_artifact=valid_patch)
    assert record["model_patch"] == valid_patch


def test_run_service_creates_manifest(monkeypatch, tmp_path: Path):
    record, outcome = _run_once(monkeypatch, tmp_path, raw_artifact="")
    payload = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == outcome.run_id
    assert payload["predictions_path"] == str(outcome.predictions_path.resolve())
    assert payload["model_name"] == "fake-agent"
    assert payload["model_name_or_path"] == "fake/model"
    assert payload["evaluation"]["status"] == "not_run"
    assert payload["benchmark_name"] == "swebench_verified"
    assert payload["dataset_name"] == "SWE-bench/SWE-bench_Verified"
    assert payload["split"] == "test"
    assert payload["mode"] == "patch_only"
    assert record["instance_id"] == "astropy__astropy-12907"


def test_run_service_quiet_suppresses_per_task_logs(monkeypatch, tmp_path: Path, capsys):
    _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    out = capsys.readouterr().out
    assert "artifact_valid" not in out


def test_cli_run_verbose_option_defaults_to_quiet():
    verbose_option = cli.run.__defaults__[-1]
    assert getattr(verbose_option, "default", None) is False


def test_run_service_verbose_prints_per_task_logs(monkeypatch, tmp_path: Path, capsys):
    _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=True)
    out = capsys.readouterr().out
    assert "artifact_valid=" in out


def test_run_service_writes_run_log_file(monkeypatch, tmp_path: Path):
    _, outcome = _run_once(monkeypatch, tmp_path, raw_artifact="", verbose=False)
    content = outcome.run_log_path.read_text(encoding="utf-8")
    assert "Starting run:" in content
    assert "Run summary:" in content
    assert "artifact_valid=" in content


def test_derive_run_id_from_artifacts_path(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    run_id = "2026-02-13_010203"
    p = artifacts_dir / run_id / "predictions.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    assert eval_service.derive_run_id_from_predictions(p, artifacts_dir) == run_id


def test_derive_run_id_rejects_non_canonical_layout(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    p = artifacts_dir / "2026-02-13_010203" / "predictions" / "model" / "test" / "patch_only" / "predictions.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        eval_service.derive_run_id_from_predictions(p, artifacts_dir)

