import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from runtime.config_models import RunConfig
from runtime.schemas import BenchmarkTask


class SWEbenchVerifiedAdapter:
    """SWE-bench task loader and prediction serializer."""

    benchmark_name = "swebench_verified"

    def __init__(
        self,
        data_source: str = "hf",
        data_root: Optional[str] = None,
        dataset_name: str = "SWE-bench/SWE-bench_Verified",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist benchmark source configuration for later task loading."""

        self.data_source = data_source
        self.data_root = Path(data_root) if data_root else None
        self.dataset_name = dataset_name
        self.params = params or {}

    @classmethod
    def from_config(cls, config: RunConfig) -> "SWEbenchVerifiedAdapter":
        """Build adapter state from normalized run config."""

        return cls(
            data_source=config.benchmark.data_source,
            data_root=config.benchmark.data_root,
            dataset_name=config.benchmark.dataset_name,
            params=config.benchmark.params,
        )

    def load_tasks(self, split: str = "test", selector: Optional[int] = None) -> List[BenchmarkTask]:
        """Load tasks from HF or local JSONL source with strict input requirements."""

        if self.data_source == "hf":
            return self._load_tasks_from_hf(split, selector)

        if not self.data_root:
            raise ValueError("data_root is required when data_source=local")

        path = self.data_root / f"{split}.jsonl"
        if not path.exists():
            raise ValueError(f"Missing dataset split file: {path}")

        tasks: List[BenchmarkTask] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Invalid record type in {path}: expected object, got {type(record).__name__}"
                    )
                task = self._record_to_task(cast(Mapping[str, Any], record))
                tasks.append(task)
                if selector and len(tasks) >= selector:
                    break
        return tasks

    def _load_tasks_from_hf(self, split: str, selector: Optional[int]) -> List[BenchmarkTask]:
        """Load a bounded set of tasks from Hugging Face dataset rows."""

        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=split)
        tasks: List[BenchmarkTask] = []
        count = selector or len(ds)
        for record in ds.select(range(min(count, len(ds)))):  # type: ignore[arg-type]
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record type in dataset split '{split}': "
                    f"expected object, got {type(record).__name__}"
                )
            tasks.append(self._record_to_task(cast(Mapping[str, Any], record)))
        return tasks

    def _record_to_task(self, record: Mapping[str, Any]) -> BenchmarkTask:
        """Convert one dataset row to a strict patch-generation benchmark task."""

        instance_id_raw = record.get("instance_id")
        if not isinstance(instance_id_raw, str) or not instance_id_raw.strip():
            raise ValueError(f"Invalid or missing instance_id: {instance_id_raw!r}")
        instance_id = instance_id_raw.strip()

        instruction = ""
        for key in ("problem_statement", "task_description", "prompt", "issue", "title"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                instruction = value.strip()
                break

        if not instruction:
            raise ValueError(f"Empty instruction for instance_id={instance_id}")

        repo_raw = record.get("repo")
        repo = repo_raw if isinstance(repo_raw, str) and repo_raw.strip() else None
        resources: Dict[str, Any] = {"repo": repo} if repo is not None else {}

        return BenchmarkTask(
            task_id=instance_id,
            instruction=instruction,
            resources=resources,
            expected_output_type="patch",
        )

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path:
        """Resolve the workspace root used by tools for a given task."""

        if self.data_root:
            repo = task.resources.get("repo") if task.resources else None
            if isinstance(repo, str) and repo:
                return self.data_root / repo
            return self.data_root
        return Path(".")

    def get_evaluator(self, config: RunConfig):
        """Return the benchmark-specific evaluator implementation."""

        from benchmarks.swebench_verified.evaluator import SWEbenchEvaluator

        return SWEbenchEvaluator()

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serialize predictions into the SWE-bench JSONL schema."""

        repo = None
        if metadata and metadata.get("repo"):
            repo = metadata.get("repo")
        elif task.resources and task.resources.get("repo"):
            repo = task.resources.get("repo")
        return {
            "instance_id": task.task_id,
            "model_patch": artifact,
            "model_name_or_path": model_name_or_path,
            "model_name": model_name,
            "repo": repo,
        }
