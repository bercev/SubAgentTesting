from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from runtime.config_models import RunConfig


class BaseHarnessEvaluator:
    def __init__(self) -> None:
        self.last_summary_report: Optional[Path] = None
        self.last_harness_log_root: Optional[Path] = None
        self._workdir: Path = Path.cwd()

    def run_harness(
        self,
        predictions_path: Path,
        run_id: str,
        config: RunConfig,
    ) -> subprocess.CompletedProcess[str]:
        evaluation = config.evaluation
        if not evaluation.harness_cmd:
            raise RuntimeError("harness_cmd not set; cannot run official harness")
        if not evaluation.eval_root:
            raise RuntimeError("eval_root not set; cannot run official harness")

        eval_root = Path(evaluation.eval_root)
        if not eval_root.exists():
            raise RuntimeError(f"eval_root does not exist: {eval_root}")

        workdir = Path(evaluation.workdir).resolve()
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        self._workdir = workdir

        run_root = Path(config.output.artifacts_dir) / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        cmd = self.build_command(predictions_path.resolve(), run_id, config)
        proc = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True, text=True)

        self.last_summary_report = self._relocate_summary_report(stdout=proc.stdout, run_id=run_id, run_root=run_root)
        self.last_harness_log_root = self.relocate_harness_logs(run_id=run_id, run_root=run_root)

        if self.last_summary_report:
            proc.stdout = f"{proc.stdout.rstrip()}\nReport relocated to {self.last_summary_report}\n"
        if self.last_harness_log_root:
            proc.stdout = f"{proc.stdout.rstrip()}\nHarness logs relocated to {self.last_harness_log_root}\n"
        return proc

    def build_command(self, predictions_path: Path, run_id: str, config: RunConfig) -> str:
        raise NotImplementedError

    def resolve_summary_report(self, stdout: str, run_id: str, run_root: Path) -> Optional[Path]:
        matches = re.findall(r"Report written to\s+([^\s]+\.json)", stdout or "")
        for raw_path in reversed(matches):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self._workdir / candidate
            if candidate.exists():
                return candidate

        canonical = run_root / "report.json"
        if canonical.exists():
            return canonical

        in_run_root = sorted(run_root.glob(f"*.{run_id}.json"))
        if in_run_root:
            return in_run_root[-1]

        fallback_matches = sorted(self._workdir.glob(f"*.{run_id}.json"))
        if fallback_matches:
            return fallback_matches[-1]
        return None

    def relocate_harness_logs(self, run_id: str, run_root: Path) -> Optional[Path]:
        return None

    def _relocate_summary_report(self, stdout: str, run_id: str, run_root: Path) -> Optional[Path]:
        source = self.resolve_summary_report(stdout=stdout, run_id=run_id, run_root=run_root)
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

