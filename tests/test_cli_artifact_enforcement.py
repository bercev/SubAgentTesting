import json
from pathlib import Path

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


def _run_once(monkeypatch, tmp_path: Path, raw_artifact: str) -> dict:
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
            "runs_dir": str(tmp_path / "runs"),
            "logs_dir": str(tmp_path / "logs"),
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
    )
    out_files = list((tmp_path / "runs").glob("*/*/*/*_predictions.jsonl"))
    assert len(out_files) == 1
    records = [json.loads(line) for line in out_files[0].read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    return records[0]


def test_cli_writes_empty_patch_for_invalid_patch_output(monkeypatch, tmp_path: Path):
    record = _run_once(monkeypatch, tmp_path, raw_artifact="I'll inspect files first.")
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
    record = _run_once(monkeypatch, tmp_path, raw_artifact=valid_patch)
    assert record["model_patch"] == valid_patch
