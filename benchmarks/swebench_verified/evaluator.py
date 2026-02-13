import json
import shutil
import tempfile
from pathlib import Path

from benchmarks.base_evaluator import BaseHarnessEvaluator
from runtime.config_models import RunConfig


class SWEbenchEvaluator(BaseHarnessEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def build_command(self, predictions_path: Path, run_id: str, config: RunConfig) -> str:
        run_root = Path(config.output.artifacts_dir) / run_id
        abs_predictions = self._ensure_model_key(predictions_path.resolve())
        return (
            f"{config.evaluation.harness_cmd} "
            f"-d {config.benchmark.dataset_name} "
            f"-s {config.benchmark.split} "
            f"-p {abs_predictions} "
            f"-id {run_id} "
            f"--report_dir {run_root.resolve()}"
        )

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

    def relocate_harness_logs(self, run_id: str, run_root: Path) -> "Path | None":
        source = self._workdir / "logs" / "run_evaluation" / run_id
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
