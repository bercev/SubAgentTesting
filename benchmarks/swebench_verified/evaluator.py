import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List

from runtime.schemas import AgentResult


class SWEbenchEvaluator:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.harness_cmd = os.getenv("HARNESS_CMD")

    def write_predictions(self, results: Iterable[AgentResult], model_name: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for res in results:
                record = {
                    "instance_id": res.task_id,
                    "model_patch": res.final_artifact,
                    "model_name": model_name,
                    "repo": res.metadata.get("repo") if res.metadata else None,
                }
                f.write(json.dumps(record) + "\n")
        return out_path

    def run_harness(self, predictions_path: Path) -> subprocess.CompletedProcess:
        if not self.harness_cmd:
            raise RuntimeError("HARNESS_CMD not set; cannot run official harness")
        cmd = f"{self.harness_cmd} --predictions {predictions_path}"
        return subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True, text=True)
