from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "swebench_verified"
    dataset_name: str = "SWE-bench/SWE-bench_Verified"
    split: str = "test"
    data_source: str = "hf"
    data_root: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    harness_cmd: str = "python -m swebench.harness.run_evaluation"
    eval_root: str = "./external/SWE-bench"
    workdir: str = "."
    report_dir: str = "reports"
    params: Dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "patch_only"
    selector: Optional[int] = 5
    max_tool_calls: int = 20
    max_wall_time_s: int = 600


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: str = "artifacts"


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

