from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.swebench_verified.adapter import SWEbenchVerifiedAdapter


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
