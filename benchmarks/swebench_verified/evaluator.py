import json
import shutil
from pathlib import Path

from benchmarks.base_evaluator import BaseHarnessEvaluator
from runtime.config_models import RunConfig


class SWEbenchEvaluator(BaseHarnessEvaluator):
    """SWE-bench harness evaluator with canonical report/log relocation."""

    def __init__(self) -> None:
        """Initialize evaluator state managed by `BaseHarnessEvaluator`."""

        super().__init__()

    def build_command(self, predictions_path: Path, run_id: str, config: RunConfig) -> str:
        """Validate predictions schema and build harness CLI command."""

        run_root = Path(config.output.artifacts_dir) / run_id
        abs_predictions = predictions_path.resolve()
        self._validate_predictions_schema(abs_predictions)
        return (
            f"{config.evaluation.harness_cmd} "
            f"-d {config.benchmark.dataset_name} "
            f"-s {config.benchmark.split} "
            f"-p {abs_predictions} "
            f"-id {run_id} "
            f"--report_dir {run_root.resolve()}"
        )

    def _validate_predictions_schema(self, predictions_path: Path) -> None:
        """Require `model_name_or_path` on every prediction row before harness run."""

        if predictions_path.suffix != ".jsonl":
            raise ValueError(f"Predictions must be a .jsonl file: {predictions_path}")

        with predictions_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in predictions file at line {line_no}: {predictions_path}"
                    ) from exc
                if not isinstance(rec, dict):
                    raise ValueError(
                        f"Invalid prediction row at line {line_no}: expected object, got {type(rec).__name__}"
                    )
                value = rec.get("model_name_or_path")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Missing required `model_name_or_path` at line {line_no} in {predictions_path}"
                    )

    def relocate_harness_logs(self, run_id: str, run_root: Path) -> "Path | None":
        """Move per-instance harness logs under `artifacts/<run_id>/evaluation`."""

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
