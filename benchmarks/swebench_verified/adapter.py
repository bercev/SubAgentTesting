import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                    task = self._record_to_task(record)
                    tasks.append(task)
                    if selector and len(tasks) >= selector:
                        break
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    task = self._record_to_task(record)
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
            tasks.append(self._record_to_task(record))
        return tasks

    def _record_to_task(self, record: Dict[str, Any]) -> BenchmarkTask:
        instruction = (
            record.get("problem_statement")
            or record.get("task_description")
            or record.get("prompt")
            or record.get("issue")
            or record.get("title")
            or ""
        )
        if not instruction:
            raise ValueError(f"Empty instruction for instance_id={record.get('instance_id')}")
        return BenchmarkTask(
            task_id=record.get("instance_id"),
            instruction=instruction,
            resources={"repo": record.get("repo")},
            expected_output_type="patch",
        )

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path:
        if self.data_root:
            repo = task.resources.get("repo") if task.resources else None
            if repo:
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
