from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Protocol

from runtime.schemas import BenchmarkTask


class BenchmarkEvaluator(Protocol):
    """Protocol for benchmark harness executors used by eval service."""

    last_summary_report: Optional[Path]
    last_harness_log_root: Optional[Path]

    def run_harness(
        self,
        predictions_path: Path,
        run_id: str,
        config: "RunConfig",
    ) -> subprocess.CompletedProcess[str]: ...


class BenchmarkAdapter(Protocol):
    """Protocol for benchmark-specific task loading and prediction serialization."""

    benchmark_name: ClassVar[str]

    @classmethod
    def from_config(cls, config: "RunConfig") -> "BenchmarkAdapter": ...

    def load_tasks(self, split: str, selector: Optional[int]) -> list[BenchmarkTask]: ...

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path: ...

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    def get_evaluator(self, config: "RunConfig") -> BenchmarkEvaluator: ...


from runtime.config_models import RunConfig
