import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from runtime.schemas import BenchmarkTask


class SWEbenchVerifiedAdapter:
    def __init__(
        self,
        data_source: str = "hf",
        data_root: Optional[str] = None,
        dataset_name: str = "SWE-bench/SWE-bench_Verified",
        eval_root: Optional[str] = None,
        harness_cmd: Optional[str] = None,
        workdir: Optional[str] = None,
        report_dir: Optional[str] = None,
    ) -> None:
        self.data_source = data_source
        self.data_root = Path(data_root) if data_root else None
        self.dataset_name = dataset_name
        self.eval_root = Path(eval_root) if eval_root else None
        self.harness_cmd = harness_cmd
        self.workdir = Path(workdir).resolve() if workdir else Path.cwd()
        self.report_dir = Path(report_dir) if report_dir else (self.workdir / "logs" / "reports")

    def load_tasks(self, split: str = "dev", selector: Optional[int] = None) -> List[BenchmarkTask]:
        """
        Load tasks from a jsonl file at {root}/{split}.jsonl.
        Each line should contain fields: instance_id, repo, prompt, patch? (ignored)
        """
        if self.data_source == "hf":
            return self._load_tasks_from_hf(split, selector)

        if not self.data_root:
            raise ValueError("data_root is required when data_source=local")

        path = self.data_root / f"{split}.jsonl"
        tasks: List[BenchmarkTask] = []
        if not path.exists():
            # Provide a dummy task for smoke tests
            tasks.append(
                BenchmarkTask(
                    task_id="dummy-0",
                    instruction="Fix the issue described in the prompt (dummy)",
                    resources={"repo": "dummy"},
                    expected_output_type="patch",
                )
            )
            return tasks[: selector or 1]

        if path.suffix == ".json":
            import ijson

            with path.open("r", encoding="utf-8") as f:
                for record in ijson.items(f, "item"):
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Invalid record type in {path}: expected object, got {type(record).__name__}"
                        )
                    task = self._record_to_task(cast(Mapping[str, Any], record))
                    tasks.append(task)
                    if selector and len(tasks) >= selector:
                        break
        else:
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
        from datasets import load_dataset

        # SWE-bench Verified on HF currently exposes only "test".
        hf_split = "test" if split == "dev" else split
        ds = load_dataset(self.dataset_name, split=hf_split)
        tasks: List[BenchmarkTask] = []
        count = selector or len(ds)
        for record in ds.select(range(min(count, len(ds)))):  # type: ignore[arg-type]
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record type in dataset split '{hf_split}': "
                    f"expected object, got {type(record).__name__}"
                )
            tasks.append(self._record_to_task(cast(Mapping[str, Any], record)))
        return tasks

    def _record_to_task(self, record: Mapping[str, Any]) -> BenchmarkTask:
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
        if self.data_root:
            repo = task.resources.get("repo") if task.resources else None
            if isinstance(repo, str) and repo:
                return self.data_root / repo
            return self.data_root
        return Path(".")

    def get_evaluator(self):
        from benchmarks.swebench_verified.evaluator import SWEbenchEvaluator

        return SWEbenchEvaluator(
            eval_root=self.eval_root,
            harness_cmd=self.harness_cmd,
            workdir=self.workdir,
            report_dir=self.report_dir,
        )

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
