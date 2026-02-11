import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List

from runtime.schemas import AgentResult


class SWEbenchEvaluator:
    def __init__(
        self,
        eval_root: "Path | None",
        harness_cmd: "str | None",
        workdir: Path,
        report_dir: Path,
    ) -> None:
        self.root = eval_root
        self.harness_cmd = harness_cmd
        self.workdir = workdir
        self.report_dir = report_dir

    def write_predictions(self, results: Iterable[AgentResult], model_name: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for res in results:
                record = {
                    "instance_id": res.task_id,
                    "model_patch": res.final_artifact,
                    "model_name_or_path": model_name,
                    "model_name": model_name,
                    "repo": res.metadata.get("repo") if res.metadata else None,
                }
                f.write(json.dumps(record) + "\n")
        return out_path

    def run_harness(
        self,
        predictions_path: Path,
        dataset_name: str,
        split: str,
        run_id: str,
    ) -> subprocess.CompletedProcess:
        if not self.harness_cmd:
            raise RuntimeError("harness_cmd not set; cannot run official harness")
        if not self.root:
            raise RuntimeError("eval_root not set; cannot run official harness")
        if not self.root.exists():
            raise RuntimeError(f"eval_root does not exist: {self.root}")
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True, exist_ok=True)
        report_dir_abs = self.report_dir if self.report_dir.is_absolute() else (self.workdir / self.report_dir)
        report_dir_abs.mkdir(parents=True, exist_ok=True)

        abs_predictions = predictions_path.resolve()
        abs_predictions = self._ensure_model_key(abs_predictions)
        cmd = (
            f"{self.harness_cmd} "
            f"-d {dataset_name} "
            f"-s {split} "
            f"-p {abs_predictions} "
            f"-id {run_id} "
            f"--report_dir {report_dir_abs.resolve()}"
        )
        return subprocess.run(cmd, shell=True, cwd=self.workdir, capture_output=True, text=True)

    def _ensure_model_key(self, predictions_path: Path) -> Path:
        if predictions_path.suffix != ".jsonl":
            return predictions_path
        with predictions_path.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        needs_fix = any("model_name_or_path" not in rec for rec in lines)
        if not needs_fix:
            return predictions_path
        for rec in lines:
            if "model_name_or_path" not in rec:
                rec["model_name_or_path"] = rec.get("model_name") or "unknown"
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl")
        with tmp:
            for rec in lines:
                tmp.write(json.dumps(rec) + "\n")
        return Path(tmp.name)
