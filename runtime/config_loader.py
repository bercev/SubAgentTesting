from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from runtime.config_models import RunConfig


def default_run_config_dict() -> Dict[str, Any]:
    return {
        "benchmark": {
            "name": "swebench_verified",
            "dataset_name": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "data_source": "hf",
            "data_root": None,
            "params": {},
        },
        "evaluation": {
            "enabled": True,
            "harness_cmd": "python -m swebench.harness.run_evaluation",
            "eval_root": "./external/SWE-bench",
            "workdir": ".",
            "report_dir": "reports",
            "params": {},
        },
        "runtime": {
            "mode": "patch_only",
            "selector": 5,
            "max_tool_calls": 20,
            "max_wall_time_s": 600,
        },
        "output": {
            "artifacts_dir": "artifacts",
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def normalize_mode(mode_value: str) -> str:
    aliases = {
        "A": "patch_only",
        "PATCH_ONLY": "patch_only",
        "PATCH-ONLY": "patch_only",
        "SUBMIT_ONLY": "patch_only",
        "B": "tools_enabled",
        "TOOLS_ENABLED": "tools_enabled",
        "TOOLS-ENABLED": "tools_enabled",
        "TOOLS": "tools_enabled",
    }
    normalized = aliases.get(str(mode_value).strip().upper())
    if normalized:
        return normalized
    raise ValueError(
        f"Unsupported mode '{mode_value}'. Use one of: patch_only, tools_enabled (legacy: A, B)."
    )


def normalize_run_config_dict(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    defaults = default_run_config_dict()
    nested_keys = ("benchmark", "evaluation", "runtime", "output")
    if any(isinstance(raw_config.get(key), dict) for key in nested_keys):
        return _deep_merge(defaults, raw_config)

    # Legacy flat config compatibility
    legacy_as_nested = {
        "benchmark": {
            "name": raw_config.get("benchmark"),
            "dataset_name": raw_config.get("dataset_name"),
            "split": raw_config.get("default_split") or raw_config.get("split"),
            "data_source": raw_config.get("data_source"),
            "data_root": raw_config.get("data_root"),
            "params": raw_config.get("benchmark_params"),
        },
        "evaluation": {
            "harness_cmd": raw_config.get("harness_cmd"),
            "eval_root": raw_config.get("eval_root"),
            "workdir": raw_config.get("workdir"),
            "report_dir": raw_config.get("report_dir"),
            "params": raw_config.get("evaluation_params"),
        },
        "runtime": {
            "mode": raw_config.get("mode"),
            "selector": raw_config.get("selector"),
            "max_tool_calls": raw_config.get("max_tool_calls"),
            "max_wall_time_s": raw_config.get("max_wall_time_s"),
        },
        "output": {
            "artifacts_dir": raw_config.get("artifacts_dir"),
        },
    }
    return _deep_merge(defaults, legacy_as_nested)


def normalize_run_config(raw_config: Dict[str, Any]) -> RunConfig:
    normalized_dict = normalize_run_config_dict(raw_config)
    config = RunConfig.model_validate(normalized_dict)
    config.runtime.mode = normalize_mode(config.runtime.mode)
    return config


def load_run_config(run_config_path: Path) -> RunConfig:
    if not run_config_path.exists():
        raise FileNotFoundError(
            "Missing run config: "
            f"{run_config_path}. Create one from `configs/runs/example.swebench_verified.hf.yaml`."
        )
    with run_config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Invalid run config shape in {run_config_path}: expected object at root")
    return normalize_run_config(raw_config)


def apply_run_overrides(
    config: RunConfig,
    *,
    benchmark: Optional[str] = None,
    split: Optional[str] = None,
    selector: Optional[int] = None,
    mode: Optional[str] = None,
) -> RunConfig:
    effective = config.model_copy(deep=True)
    if benchmark:
        effective.benchmark.name = benchmark
    if split:
        effective.benchmark.split = split
    if selector is not None:
        effective.runtime.selector = selector
    if mode:
        effective.runtime.mode = normalize_mode(mode)
    return effective
