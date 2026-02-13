import json
import re
import shutil
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
        self.last_summary_report: "Path | None" = None
        self.last_harness_log_root: "Path | None" = None

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
        artifacts_dir: Path,
    ) -> subprocess.CompletedProcess:
        if not self.harness_cmd:
            raise RuntimeError("harness_cmd not set; cannot run official harness")
        if not self.root:
            raise RuntimeError("eval_root not set; cannot run official harness")
        if not self.root.exists():
            raise RuntimeError(f"eval_root does not exist: {self.root}")
        if not self.workdir.exists():
            self.workdir.mkdir(parents=True, exist_ok=True)
        run_root = artifacts_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        abs_predictions = predictions_path.resolve()
        abs_predictions = self._ensure_model_key(abs_predictions)
        cmd = (
            f"{self.harness_cmd} "
            f"-d {dataset_name} "
            f"-s {split} "
            f"-p {abs_predictions} "
            f"-id {run_id} "
            f"--report_dir {run_root.resolve()}"
        )
        proc = subprocess.run(cmd, shell=True, cwd=self.workdir, capture_output=True, text=True)
        self.last_summary_report = self._relocate_summary_report(
            stdout=proc.stdout,
            run_id=run_id,
            run_root=run_root,
        )
        self.last_harness_log_root = self._relocate_harness_logs(run_id=run_id, run_root=run_root)
        if self.last_summary_report:
            proc.stdout = (
                f"{proc.stdout.rstrip()}\n"
                f"Report relocated to {self.last_summary_report}\n"
            )
        if self.last_harness_log_root:
            proc.stdout = (
                f"{proc.stdout.rstrip()}\n"
                f"Harness logs relocated to {self.last_harness_log_root}\n"
            )
        return proc

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

    def _relocate_summary_report(
        self,
        stdout: str,
        run_id: str,
        run_root: Path,
    ) -> "Path | None":
        source = self._resolve_summary_report(stdout=stdout, run_id=run_id, run_root=run_root)
        if not source or not source.exists():
            return None

        destination = run_root / "report.json"
        source_resolved = source.resolve(strict=False)
        destination_resolved = destination.resolve(strict=False)
        if source_resolved == destination_resolved:
            return destination_resolved

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        source.replace(destination)
        return destination.resolve(strict=False)

    def _resolve_summary_report(self, stdout: str, run_id: str, run_root: Path) -> "Path | None":
        matches = re.findall(r"Report written to\s+([^\s]+\.json)", stdout or "")
        for raw_path in reversed(matches):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self.workdir / candidate
            if candidate.exists():
                return candidate

        canonical = run_root / "report.json"
        if canonical.exists():
            return canonical

        in_run_root = sorted(run_root.glob(f"*.{run_id}.json"))
        if in_run_root:
            return in_run_root[-1]

        fallback_matches = sorted(self.workdir.glob(f"*.{run_id}.json"))
        if fallback_matches:
            return fallback_matches[-1]
        return None

    def _relocate_harness_logs(self, run_id: str, run_root: Path) -> "Path | None":
        source = self.workdir / "logs" / "run_evaluation" / run_id
        destination = run_root / "evaluation"

        if not source.exists():
            if destination.exists():
                return destination.resolve(strict=False)
            return None

        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

        for model_dir in sorted(source.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for instance_dir in sorted(model_dir.iterdir()):
                if not instance_dir.is_dir():
                    continue
                target = destination / instance_dir.name
                if target.exists():
                    target = destination / f"{instance_dir.name}__{model_name}"
                    suffix = 2
                    while target.exists():
                        target = destination / f"{instance_dir.name}__{model_name}_{suffix}"
                        suffix += 1
                shutil.move(str(instance_dir), str(target))

        shutil.rmtree(source, ignore_errors=True)
        return destination.resolve(strict=False)
