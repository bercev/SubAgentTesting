from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from agent_architectures.constants import ARCHITECTURE_NONE, normalize_architecture_id
from runtime.config_models import RunConfig


def default_run_config_dict() -> Dict[str, Any]:
    """Return the canonical nested defaults for all run config sections."""

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
            "harness_cmd": "python -m swebench.harness.run_evaluation",
            "eval_root": "./external/SWE-bench",
            "workdir": ".",
            "params": {},
        },
        "runtime": {
            "mode": "patch_only",
            "selector": 5,
            "max_tool_calls": 20,
            "max_wall_time_s": 600,
            "patch_submit_policy": "reject_retry",
            "max_invalid_submit_attempts": 3,
            "agent_architecture_override": None,
            "tool_quality_enabled": True,
            "tool_quality_weights": {
                "execution_quality": 0.45,
                "policy_quality": 0.25,
                "termination_quality": 0.20,
                "budget_quality": 0.10,
            },
        },
        "output": {
            "artifacts_dir": "artifacts",
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested config values while preserving default sections."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _validate_mode(mode_value: str) -> Literal["patch_only", "tools_enabled"]:
    """Enforce strict mode values with no alias conversions."""

    if mode_value in {"patch_only", "tools_enabled"}:
        return mode_value
    raise ValueError(
        f"Unsupported mode '{mode_value}'. Use one of: patch_only, tools_enabled."
    )


def _validate_agent_architecture_override(value: Optional[str]) -> Optional[str]:
    """Validate optional run-level architecture override id."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("runtime.agent_architecture_override must be a string when provided")
    if not value.strip():
        return None
    normalized = normalize_architecture_id(value)
    # Explicit 'none' is valid and semantically useful as an override.
    if normalized == ARCHITECTURE_NONE:
        return ARCHITECTURE_NONE
    return normalized


def normalize_run_config_dict(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate nested config shape and merge with canonical defaults."""

    required_sections = ("benchmark", "evaluation", "runtime", "output")
    for key in required_sections:
        value = raw_config.get(key)
        if not isinstance(value, dict):
            raise ValueError(
                "Run config must use strict nested sections "
                f"{required_sections}; section '{key}' is missing or not an object."
            )

    defaults = default_run_config_dict()
    return _deep_merge(defaults, raw_config)


def normalize_run_config(raw_config: Dict[str, Any]) -> RunConfig:
    """Parse and strictly validate runtime config values."""

    normalized_dict = normalize_run_config_dict(raw_config)
    config = RunConfig.model_validate(normalized_dict)
    config.runtime.mode = _validate_mode(config.runtime.mode)
    config.runtime.agent_architecture_override = _validate_agent_architecture_override(
        config.runtime.agent_architecture_override
    )
    return config


def load_run_config(run_config_path: Path) -> RunConfig:
    """Load and validate a run config YAML file from disk."""

    if not run_config_path.exists():
        raise FileNotFoundError(
            "Missing run config: "
            f"{run_config_path}. Create one from `profiles/runs/example.swebench_verified.hf.yaml`."
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
    """Apply CLI overrides after strict config parsing."""

    effective = config.model_copy(deep=True)
    if benchmark:
        effective.benchmark.name = benchmark
    if split:
        effective.benchmark.split = split
    if selector is not None:
        effective.runtime.selector = selector
    if mode:
        effective.runtime.mode = _validate_mode(mode)
    return effective
