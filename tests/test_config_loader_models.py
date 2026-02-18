from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config_loader import load_run_config, normalize_run_config


def test_normalize_run_config_keeps_params_maps():
    cfg = normalize_run_config(
        {
            "benchmark": {
                "name": "swebench_verified",
                "dataset_name": "SWE-bench/SWE-bench_Verified",
                "split": "test",
                "data_source": "hf",
                "params": {"subset": "mini"},
            },
            "evaluation": {
                "harness_cmd": "python -m swebench.harness.run_evaluation",
                "eval_root": "./external/SWE-bench",
                "workdir": ".",
                "params": {"timeout_s": 900},
            },
            "runtime": {
                "mode": "patch_only",
                "selector": 1,
                "max_tool_calls": 2,
                "max_wall_time_s": 30,
            },
            "output": {"artifacts_dir": "artifacts"},
        }
    )
    assert cfg.benchmark.params["subset"] == "mini"
    assert cfg.evaluation.params["timeout_s"] == 900
    assert cfg.runtime.tool_quality_enabled is True
    assert cfg.runtime.tool_quality_weights.execution_quality == 0.45
    assert cfg.runtime.tool_quality_weights.policy_quality == 0.25
    assert cfg.runtime.tool_quality_weights.termination_quality == 0.20
    assert cfg.runtime.tool_quality_weights.budget_quality == 0.10


def test_normalize_run_config_rejects_flat_top_level_keys():
    with pytest.raises(ValueError):
        normalize_run_config(
            {
                "benchmark": "swebench_verified",
                "dataset_name": "SWE-bench/SWE-bench_Verified",
                "default_split": "test",
                "data_source": "hf",
                "benchmark_params": {"dataset_revision": "main"},
                "evaluation_params": {"retries": 1},
                "mode": "A",
                "selector": 3,
                "max_tool_calls": 5,
                "max_wall_time_s": 60,
                "artifacts_dir": "artifacts",
            }
        )


def test_normalize_run_config_rejects_mode_aliases():
    with pytest.raises(ValueError):
        normalize_run_config(
            {
                "benchmark": {
                    "name": "swebench_verified",
                    "dataset_name": "SWE-bench/SWE-bench_Verified",
                    "split": "test",
                    "data_source": "hf",
                },
                "evaluation": {
                    "harness_cmd": "python -m swebench.harness.run_evaluation",
                    "eval_root": "./external/SWE-bench",
                    "workdir": ".",
                },
                "runtime": {
                    "mode": "A",
                    "selector": 1,
                    "max_tool_calls": 2,
                    "max_wall_time_s": 30,
                },
                "output": {"artifacts_dir": "artifacts"},
            }
        )


def test_repo_run_configs_parse():
    paths = sorted(Path("profiles/runs").glob("*.yaml"))
    assert paths, "No run config files found under profiles/runs/"
    for path in paths:
        cfg = load_run_config(path)
        assert cfg.benchmark.name
        assert cfg.output.artifacts_dir


def test_invalid_types_fail_validation():
    with pytest.raises(Exception):
        normalize_run_config(
            {
                "runtime": {
                    "mode": "patch_only",
                    "selector": 1,
                    "max_tool_calls": "not-an-int",
                    "max_wall_time_s": 10,
                }
            }
        )


def test_tool_quality_weights_reject_out_of_bounds_values():
    with pytest.raises(Exception):
        normalize_run_config(
            {
                "benchmark": {
                    "name": "swebench_verified",
                    "dataset_name": "SWE-bench/SWE-bench_Verified",
                    "split": "test",
                    "data_source": "hf",
                },
                "evaluation": {
                    "harness_cmd": "python -m swebench.harness.run_evaluation",
                    "eval_root": "./external/SWE-bench",
                    "workdir": ".",
                },
                "runtime": {
                    "mode": "tools_enabled",
                    "selector": 1,
                    "max_tool_calls": 2,
                    "max_wall_time_s": 30,
                    "tool_quality_weights": {
                        "execution_quality": 1.1,
                        "policy_quality": 0.0,
                        "termination_quality": 0.0,
                        "budget_quality": -0.1,
                    },
                },
                "output": {"artifacts_dir": "artifacts"},
            }
        )


def test_tool_quality_weights_reject_invalid_sum():
    with pytest.raises(Exception):
        normalize_run_config(
            {
                "benchmark": {
                    "name": "swebench_verified",
                    "dataset_name": "SWE-bench/SWE-bench_Verified",
                    "split": "test",
                    "data_source": "hf",
                },
                "evaluation": {
                    "harness_cmd": "python -m swebench.harness.run_evaluation",
                    "eval_root": "./external/SWE-bench",
                    "workdir": ".",
                },
                "runtime": {
                    "mode": "tools_enabled",
                    "selector": 1,
                    "max_tool_calls": 2,
                    "max_wall_time_s": 30,
                    "tool_quality_weights": {
                        "execution_quality": 0.5,
                        "policy_quality": 0.3,
                        "termination_quality": 0.3,
                        "budget_quality": 0.1,
                    },
                },
                "output": {"artifacts_dir": "artifacts"},
            }
        )
