from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.swebench_verified.adapter import SWEbenchVerifiedAdapter
from runtime.schemas import BenchmarkTask


def test_local_source_requires_data_root():
    adapter = SWEbenchVerifiedAdapter(data_source="local", data_root=None)
    with pytest.raises(ValueError, match="data_root is required"):
        adapter.load_tasks(split="test", selector=1)


def test_local_source_fails_when_split_file_missing(tmp_path: Path):
    adapter = SWEbenchVerifiedAdapter(data_source="local", data_root=str(tmp_path))
    with pytest.raises(ValueError, match="Missing dataset split file"):
        adapter.load_tasks(split="test", selector=1)


def test_hf_loader_uses_requested_split_without_remap(monkeypatch):
    captured = {"split": None}

    class _FakeDataset:
        def __len__(self):
            return 1

        def select(self, _indices):
            return [
                {
                    "instance_id": "astropy__astropy-12907",
                    "problem_statement": "Fix issue",
                    "repo": "astropy/astropy",
                }
            ]

    def _fake_load_dataset(dataset_name: str, split: str):
        captured["split"] = split
        return _FakeDataset()

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    adapter = SWEbenchVerifiedAdapter(data_source="hf", dataset_name="SWE-bench/SWE-bench_Verified")
    tasks = adapter.load_tasks(split="dev", selector=1)
    assert captured["split"] == "dev"
    assert len(tasks) == 1
    assert tasks[0].task_id == "astropy__astropy-12907"


def test_workspace_context_hf_mode_is_not_tool_ready():
    adapter = SWEbenchVerifiedAdapter(data_source="hf", dataset_name="SWE-bench/SWE-bench_Verified")
    task = BenchmarkTask(
        task_id="astropy__astropy-12907",
        instruction="Fix issue",
        resources={"repo": "astropy/astropy"},
        expected_output_type="patch",
    )

    ctx = adapter.workspace_context_for_task(task)

    assert ctx.workspace_kind == "runner_root"
    assert ctx.tools_ready is False
    assert ctx.workspace_exists is True
    assert ctx.repo == "astropy/astropy"
    assert ctx.reason is not None
    assert "benchmark.data_source=local" in ctx.reason


def test_workspace_context_local_mode_uses_repo_checkout_when_present(tmp_path: Path):
    repo_path = tmp_path / "astropy" / "astropy"
    repo_path.mkdir(parents=True)
    adapter = SWEbenchVerifiedAdapter(data_source="local", data_root=str(tmp_path))
    task = BenchmarkTask(
        task_id="astropy__astropy-12907",
        instruction="Fix issue",
        resources={"repo": "astropy/astropy"},
        expected_output_type="patch",
    )

    ctx = adapter.workspace_context_for_task(task)

    assert ctx.workspace_kind == "repo_checkout"
    assert ctx.tools_ready is True
    assert ctx.workspace_root == repo_path


def test_workspace_context_local_mode_reports_missing_repo_checkout(tmp_path: Path):
    adapter = SWEbenchVerifiedAdapter(data_source="local", data_root=str(tmp_path))
    task = BenchmarkTask(
        task_id="astropy__astropy-12907",
        instruction="Fix issue",
        resources={"repo": "astropy/astropy"},
        expected_output_type="patch",
    )

    ctx = adapter.workspace_context_for_task(task)

    assert ctx.workspace_kind == "dataset_root"
    assert ctx.tools_ready is False
    assert ctx.workspace_root == tmp_path
    assert ctx.reason is not None
    assert "Missing repo checkout" in ctx.reason
