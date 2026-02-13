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


def test_normalize_run_config_rejects_legacy_flat_keys():
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
    paths = [
        Path("configs/runs/default.yaml"),
        Path("configs/runs/example.swebench_verified.hf.yaml"),
        Path("configs/runs/openrouter_free_swebench.yaml"),
        Path("configs/runs/qwen3_coder_free_swebench.yaml"),
    ]
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
