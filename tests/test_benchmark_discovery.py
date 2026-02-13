from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from benchmarks.discovery import _is_adapter_candidate, discover_benchmark_adapters
from benchmarks.registry import BenchmarkRegistry
from runtime.schemas import BenchmarkTask


class _ValidShapeAdapter:
    benchmark_name = "valid"

    @classmethod
    def from_config(cls, config):
        return cls()

    def load_tasks(self, split: str, selector: Optional[int]) -> list[BenchmarkTask]:
        return []

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path:
        return Path(".")

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {}

    def get_evaluator(self, config):
        raise NotImplementedError


class _MissingNameAdapter(_ValidShapeAdapter):
    benchmark_name = None


class _OverrideAdapter(_ValidShapeAdapter):
    benchmark_name = "swebench_verified"


def test_discovery_finds_swebench_verified():
    discovered = discover_benchmark_adapters()
    assert "swebench_verified" in discovered


def test_candidate_check_rejects_missing_benchmark_name():
    assert _is_adapter_candidate(_ValidShapeAdapter)
    assert _is_adapter_candidate(_MissingNameAdapter) is False


def test_registry_allows_explicit_override():
    registry = BenchmarkRegistry(overrides={"swebench_verified": _OverrideAdapter})
    assert registry.get_adapter("swebench_verified") is _OverrideAdapter


def test_unknown_benchmark_error_includes_supported_list():
    registry = BenchmarkRegistry()
    with pytest.raises(KeyError) as exc:
        registry.get_adapter("does_not_exist")
    msg = str(exc.value)
    assert "Supported benchmarks:" in msg
    for name in registry.list_benchmarks():
        assert name in msg
