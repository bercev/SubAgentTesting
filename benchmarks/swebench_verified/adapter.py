import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from runtime.schemas import BenchmarkTask


class SWEbenchVerifiedAdapter:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or Path(os.getenv("SWE_BENCH_ROOT", "."))

    def load_tasks(self, split: str = "dev", selector: Optional[int] = None) -> List[BenchmarkTask]:
        """
        Load tasks from a jsonl file at {root}/{split}.jsonl.
        Each line should contain fields: instance_id, repo, prompt, patch? (ignored)
        """
        path = self.root / f"{split}.jsonl"
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

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                task = BenchmarkTask(
                    task_id=record.get("instance_id"),
                    instruction=record.get("prompt") or record.get("task_description", ""),
                    resources={"repo": record.get("repo")},
                    expected_output_type="patch",
                )
                tasks.append(task)
                if selector and len(tasks) >= selector:
                    break
        return tasks

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path:
        repo = task.resources.get("repo") if task.resources else None
        if repo:
            return self.root / repo
        return self.root

    def get_evaluator(self):
        from benchmarks.swebench_verified.evaluator import SWEbenchEvaluator

        return SWEbenchEvaluator(self.root)
