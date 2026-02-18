from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkConfig(BaseModel):
    """Dataset/benchmark selection and adapter-specific knobs."""

    model_config = ConfigDict(extra="forbid")

    name: str = "swebench_verified"
    dataset_name: str = "SWE-bench/SWE-bench_Verified"
    split: str = "test"
    data_source: str = "hf"
    data_root: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Harness execution settings shared by benchmark evaluators."""

    model_config = ConfigDict(extra="forbid")

    harness_cmd: str = "python -m swebench.harness.run_evaluation"
    eval_root: str = "./external/SWE-bench"
    workdir: str = "."
    params: Dict[str, Any] = Field(default_factory=dict)


class ToolQualityWeights(BaseModel):
    """Weighted components for generic tool-quality scoring."""

    model_config = ConfigDict(extra="forbid")

    execution_quality: float = 0.45
    policy_quality: float = 0.25
    termination_quality: float = 0.20
    budget_quality: float = 0.10

    @model_validator(mode="after")
    def validate_bounds_and_total(self) -> "ToolQualityWeights":
        """Require non-negative unit-bounded weights with sum approximately 1.0."""

        keys = (
            "execution_quality",
            "policy_quality",
            "termination_quality",
            "budget_quality",
        )
        total = 0.0
        for key in keys:
            value = float(getattr(self, key))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"runtime.tool_quality_weights.{key} must be within [0, 1]")
            total += value
        if abs(total - 1.0) > 1e-9:
            raise ValueError("runtime.tool_quality_weights must sum to 1.0")
        return self


class RuntimeConfig(BaseModel):
    """Per-task runtime limits and execution mode selection."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["patch_only", "tools_enabled"] = "patch_only"
    selector: Optional[int] = 5
    max_tool_calls: int = 20
    max_wall_time_s: int = 600
    tool_quality_enabled: bool = True
    tool_quality_weights: ToolQualityWeights = Field(default_factory=ToolQualityWeights)


class OutputConfig(BaseModel):
    """Artifact output locations for run/eval products."""

    model_config = ConfigDict(extra="forbid")

    artifacts_dir: str = "artifacts"


class RunConfig(BaseModel):
    """Top-level strongly typed run configuration."""

    model_config = ConfigDict(extra="forbid")

    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
