# Portable Agent Runner

End-to-end pipeline for benchmarking code-editing agents on SWE-bench: it bootstraps the environment, composes an agent profile with a run config, runs the agent to generate patches, and evaluates them with the SWE-bench harness while saving outputs in a portable layout.

- Agent behavior comes from `agents/*.yaml` (model/backend, prompts, skills).
- Execution policy and dataset wiring come from `configs/runs/*.yaml`.
- `agent run` produces prediction JSONL files; `agent eval` scores them and stores run artifacts under `artifacts/<run_id>/`.

## Quick Start

### 1) Bootstrap everything with one command

```bash
./scripts/bootstrap.sh
```

This command:
- creates `.venv` with `uv`
- installs this project in editable mode
- clones SWE-bench into `external/SWE-bench` if missing
- installs SWE-bench into the same `.venv`

### 2) Configure secrets

Create `.env` and set your API key:

```bash
cp .env.example .env
```

Set:

```bash
OPENROUTER_API_KEY=your_key_here
```

### 3) Choose run config + agent profile

- Run configs: `configs/runs/*.yaml`
- Agent profiles: `agents/*.yaml`

Default run config:
- `configs/runs/default.yaml`

### 4) Run a quick example (from project root)

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run one quick prediction job:

```bash
agent run \
  --agent agents/openrouter_free.yaml \
  --run-config configs/runs/openrouter_free_swebench.yaml \
  --mode patch_only \
  --selector 10
```

Pick the latest predictions file:

```bash
PRED_PATH=$(find artifacts -type f -path "*/predictions.jsonl" | sort | tail -n 1)
echo "$PRED_PATH"
```

Evaluate it:

```bash
agent eval \
  "$PRED_PATH" \
  --run-config configs/runs/openrouter_free_swebench.yaml
```

## Config Model

### Agent profile (`agents/*.yaml`)
Defines model behavior:
- backend type + model id
- prompt template
- skills and tool policy
- decoding defaults

### Run config (`configs/runs/*.yaml`)
Defines operational environment:
- benchmark source and split
- evaluation command and paths
- runtime budgets
- output directories

### How they work together

`agent run` composes both files for each run:
- `agents/*.yaml` controls the model/backend and agent behavior.
- `configs/runs/*.yaml` controls what benchmark to run, how to evaluate, and where to write outputs.
- CLI flags (like `--mode`, `--split`, `--selector`, `--benchmark`) override values from run config for that invocation.

TLDR:
- `agents/*.yaml` = "how the assistant thinks and responds"
- `configs/runs/*.yaml` = "what task environment and runtime policy to execute"

Example (`configs/runs/example.swebench_verified.hf.yaml`):

```yaml
benchmark:
  name: swebench_verified
  dataset_name: SWE-bench/SWE-bench_Verified
  split: test
  data_source: hf
  data_root:
  params: {}

evaluation:
  harness_cmd: python -m swebench.harness.run_evaluation
  eval_root: ./external/SWE-bench
  workdir: .
  params: {}

runtime:
  mode: patch_only
  selector: 10
  max_tool_calls: 20
  max_wall_time_s: 600

output:
  artifacts_dir: artifacts
```

## Architecture

- `scripts/cli.py`: thin Typer entrypoint (`agent run/predict/eval/list`) and terminal output only.
- `runtime/config_loader.py`: config read + normalization + typed validation.
- `runtime/backend_factory.py`: backend construction from agent spec.
- `runtime/run_service.py`: task loop orchestration, prediction writes, diagnostics, manifest/run log writes.
- `runtime/eval_service.py`: evaluator execution, report metrics parse, manifest updates.
- `runtime/metrics.py`: report metric parsing and formatting.
- `runtime/manifest_store.py`: stable manifest and run-log read/write helpers.
- `benchmarks/contracts.py`: adapter/evaluator interfaces.
- `benchmarks/registry.py` + `benchmarks/discovery.py`: benchmark adapter discovery and lookup.
- `benchmarks/base_evaluator.py`: reusable harness evaluator flow with benchmark hooks.

Data flow:
1. CLI loads typed config.
2. Benchmark adapter is discovered by `benchmark.name`.
3. Run service loads tasks and writes `artifacts/<run_id>/predictions.jsonl`.
4. Eval service invokes benchmark evaluator and relocates canonical outputs.
5. Metrics are parsed from `report.json` and persisted to `manifest.json`.

## Commands

List available assets:

```bash
agent list
```

Generate predictions (`patch_only`):

```bash
agent run \
  --agent agents/openrouter_free.yaml \
  --run-config configs/runs/default.yaml \
  --mode patch_only \
  --selector 10 \
  --split test
```

Generate predictions (`tools_enabled`):

```bash
agent run \
  --agent agents/openrouter_free.yaml \
  --run-config configs/runs/default.yaml \
  --mode tools_enabled \
  --selector 10 \
  --split test
```

Evaluate predictions:

```bash
agent eval \
  artifacts/<run_id>/predictions.jsonl \
  --run-config configs/runs/default.yaml
```

## Outputs

Predictions are written to:

```text
artifacts/<run_id>/predictions.jsonl
```

Run logs are written to:

```text
artifacts/<run_id>/run.log
```

Harness logs are relocated to:

```text
artifacts/<run_id>/evaluation/<instance_id>/
```

Harness report is written to:

```text
artifacts/<run_id>/report.json
```

Each run also writes:

```text
artifacts/<run_id>/manifest.json
```

`agent run` / `agent predict` are quiet by default for per-task lines. Use `--verbose` to show per-task output.
`agent eval` remains verbose by default; use `--quiet` to only print harness output on failure.

Patch outputs are pass-through: `model_patch` keeps raw model output even when patch diagnostics mark it invalid.

Quiet mode examples:

```bash
agent run --quiet --agent agents/openrouter_free.yaml --run-config configs/runs/default.yaml --mode patch_only --selector 10
agent eval --quiet artifacts/<run_id>/predictions.jsonl --run-config configs/runs/default.yaml
```

## Accuracy Summary

During `agent eval`, the CLI prints a concise summary line with:
- `resolved/submitted` accuracy (primary)
- resolved/completed and completion(submitted) rates
- unresolved/error/empty patch counts

The same metrics are persisted under:

```text
artifacts/<run_id>/manifest.json -> evaluation.metrics
```

## Portability

Swap providers/models/datasets without code changes:
1. pick an agent profile in `agents/`
2. pick a run config in `configs/runs/`
3. run `agent run` with optional CLI overrides

## Add A New Benchmark

### Step 1: Create benchmark files

Create:
- `benchmarks/<name>/__init__.py`
- `benchmarks/<name>/adapter.py`
- optionally `benchmarks/<name>/evaluator.py`

No manual registry edits are required. `benchmarks/registry.py` auto-discovers `benchmarks/*/adapter.py`.

### Step 2: Implement adapter contract

Your adapter must satisfy `benchmarks/contracts.py` and expose `benchmark_name`.

Minimal adapter shape:

```python
from pathlib import Path
from typing import Any, Dict, Optional

from runtime.config_models import RunConfig
from runtime.schemas import BenchmarkTask


class MyBenchmarkAdapter:
    benchmark_name = "my_benchmark"

    @classmethod
    def from_config(cls, config: RunConfig) -> "MyBenchmarkAdapter":
        return cls()

    def load_tasks(self, split: str, selector: Optional[int]) -> list[BenchmarkTask]:
        tasks = [
            BenchmarkTask(
                task_id="task-1",
                instruction="Fix the bug ...",
                resources={},
                expected_output_type="patch",
            )
        ]
        return tasks[: selector or len(tasks)]

    def workspace_root_for_task(self, task: BenchmarkTask) -> Path:
        return Path(".")

    def to_prediction_record(
        self,
        task: BenchmarkTask,
        artifact: str,
        model_name_or_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "instance_id": task.task_id,
            "model_patch": artifact,
            "model_name_or_path": model_name_or_path,
            "model_name": model_name,
        }

    def get_evaluator(self, config: RunConfig):
        from .evaluator import MyBenchmarkEvaluator

        return MyBenchmarkEvaluator()
```

### Step 3: Implement evaluator (optional)

If your benchmark needs evaluation, add `benchmarks/<name>/evaluator.py`.

Best default is subclassing `benchmarks/base_evaluator.py` and overriding `build_command(...)`. Override `resolve_summary_report(...)` and `relocate_harness_logs(...)` only if your harness output layout differs.

### Step 4: Add run config

Add `configs/runs/<name>.yaml`:

```yaml
benchmark:
  name: my_benchmark
  dataset_name: my-dataset
  split: test
  data_source: local
  data_root: ./data/my-benchmark
  params: {}

evaluation:
  harness_cmd: python -m my_harness.eval
  eval_root: .
  workdir: .
  params: {}

runtime:
  mode: patch_only
  selector: 1
  max_tool_calls: 20
  max_wall_time_s: 600

output:
  artifacts_dir: artifacts
```

### Step 5: Validate discovery and smoke-test

Run:

```bash
agent list
```

Confirm `my_benchmark` appears under Benchmarks, then run a 1-task smoke test:

```bash
agent run \
  --agent agents/openrouter_free.yaml \
  --run-config configs/runs/<name>.yaml \
  --selector 1 \
  --mode patch_only
```

If evaluator is implemented, run:

```bash
agent eval artifacts/<run_id>/predictions.jsonl --run-config configs/runs/<name>.yaml
```

### Step 6: Add tests for the new benchmark

Recommended:
1. Adapter task-loading and `selector` behavior.
2. Prediction record shape.
3. Evaluator relocation/report path behavior (if evaluator exists).

## Presets

- `configs/runs/default.yaml`
- `configs/runs/example.swebench_verified.hf.yaml`
- `configs/runs/openrouter_free_swebench.yaml`
- `configs/runs/local_jsonl_swebench.yaml`
- `configs/runs/qwen3_coder_free_swebench.yaml`

Agent presets include:
- `agents/openrouter_free.yaml`
- `agents/qwen3_coder_free.yaml`

## Tests

```bash
pytest
```

## Reference Tables

### CLI Arguments

#### `agent list`

| Argument | Type | Default | Default On? | Description |
|---|---|---|---|---|
| (none) | - | - | - | Lists available agent profiles, benchmarks, and run configs. |

#### `agent run`

| Argument | Type | Default | Default On? | Description |
|---|---|---|---|---|
| `--agent` | `str` | `agents/qwen2_5_coder.yaml` | Yes | Agent profile YAML path. |
| `--benchmark` | `str \| null` | `null` | No | Override benchmark name from run config. |
| `--split` | `str \| null` | `null` | No | Override dataset split from run config. |
| `--selector` | `int \| null` | `null` | No | Override number of tasks. |
| `--mode` | `str \| null` | `null` | No | Override runtime mode (`patch_only` or `tools_enabled`). |
| `--run-config` | `str` | `configs/runs/default.yaml` | Yes | Run config YAML path. |
| `--verbose / --quiet` | flag | `--quiet` | `--quiet` | Terminal verbosity for per-task lines. Run logs are still written to file. |

#### `agent predict`

| Argument | Type | Default | Default On? | Description |
|---|---|---|---|---|
| `--agent` | `str` | `agents/qwen2_5_coder.yaml` | Yes | Agent profile YAML path. |
| `--split` | `str \| null` | `null` | No | Override dataset split. |
| `--selector` | `int \| null` | `1` | Yes | Number of tasks to run. |
| `--run-config` | `str` | `configs/runs/default.yaml` | Yes | Run config YAML path. |
| `--verbose / --quiet` | flag | `--quiet` | `--quiet` | Terminal verbosity for per-task lines. |

Notes:
1. `agent predict` is a convenience wrapper over `agent run` with `mode="patch_only"` and no benchmark override.

#### `agent eval`

| Argument | Type | Default | Default On? | Description |
|---|---|---|---|---|
| `predictions` (positional) | `str` | `artifacts/<run_id>/predictions.jsonl` | Yes | Predictions file to evaluate. Must follow canonical run layout. |
| `--run-config` | `str` | `configs/runs/default.yaml` | Yes | Run config YAML path. |
| `--benchmark` | `str \| null` | `null` | No | Optional benchmark override. |
| `--verbose / --quiet` | flag | `--verbose` | `--verbose` | Harness output verbosity in terminal. |

### YAML Configuration Schema

#### Run Config Schema (`configs/runs/*.yaml`)

| Key | Type | Default | Required | Description |
|---|---|---|---|---|
| `benchmark.name` | `str` | `swebench_verified` | No | Benchmark adapter name used by discovery/registry. |
| `benchmark.dataset_name` | `str` | `SWE-bench/SWE-bench_Verified` | No | Dataset identifier passed to adapter/evaluator. |
| `benchmark.split` | `str` | `test` | No | Dataset split to load. |
| `benchmark.data_source` | `str` | `hf` | No | Task source type (`hf` or `local`, adapter-dependent). |
| `benchmark.data_root` | `str \| null` | `null` | No | Local data root when `data_source=local`. |
| `benchmark.params` | `dict[str, Any]` | `{}` | No | Benchmark-specific extension map. |
| `evaluation.harness_cmd` | `str` | `python -m swebench.harness.run_evaluation` | No | Harness command used by evaluator. |
| `evaluation.eval_root` | `str` | `./external/SWE-bench` | No | Required path for evaluator preflight checks. |
| `evaluation.workdir` | `str` | `.` | No | Working directory for harness subprocess. |
| `evaluation.params` | `dict[str, Any]` | `{}` | No | Evaluator-specific extension map. |
| `runtime.mode` | `str` | `patch_only` | No | Execution mode (`patch_only` or `tools_enabled`) with strict values only. |
| `runtime.selector` | `int \| null` | `5` | No | Number of tasks to run (`null` means adapter default/all). |
| `runtime.max_tool_calls` | `int` | `20` | No | Per-task tool-call budget. |
| `runtime.max_wall_time_s` | `int` | `600` | No | Per-task wall-time budget in seconds. |
| `output.artifacts_dir` | `str` | `artifacts` | No | Root directory for run artifacts. |

#### Agent Profile Schema (`agents/*.yaml`)

| Key | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | `str` | none | Yes | Agent/profile name stored in predictions/manifest. |
| `backend` | `dict` | none | Yes | Backend configuration object. |
| `backend.type` | `str` | `openrouter` | No | Backend type consumed by `runtime/backend_factory.py`. |
| `backend.model` | `str` | none | Yes (OpenRouter) | Model id (for OpenRouter backend). |
| `backend.base_url` | `str` | `https://openrouter.ai/api/v1` | No | OpenRouter-compatible API base URL. |
| `backend.max_retries` | `int` | `8` | No | Request retry attempts for transient backend errors. |
| `backend.initial_backoff_s` | `float` | `1.0` | No | Initial exponential backoff delay. |
| `backend.max_backoff_s` | `float` | `10.0` | No | Max retry backoff delay. |
| `prompt_template` | `str` | none | Yes | System prompt template (`{skills}` placeholder supported). |
| `tools` | `list[dict]` | `[]` | No | Reserved for agent/tool policy metadata. |
| `skills` | `list[str]` | `[]` | No | Skill folder names loaded from `skills/<name>/SKILL.md`. |
| `tool_to_skill_map` | `dict[str, list[str]]` | `{}` | No | Optional tool-to-skill mapping metadata. |
| `termination` | `dict` | `{}` | No | Termination metadata (`tool`, `output_type`) for profile consistency. |
| `decoding_defaults` | `dict[str, Any]` | `{}` | No | Generation defaults passed to backend (e.g., `temperature`, `top_p`, `max_tokens`). |

### Included YAML Presets

#### Run Config Presets (`configs/runs/*.yaml`)

| File | Benchmark | Data Source | Mode | Selector | Purpose |
|---|---|---|---|---|---|
| `configs/runs/default.yaml` | `swebench_verified` | `hf` | `patch_only` | `10` | General default SWE-bench run. |
| `configs/runs/example.swebench_verified.hf.yaml` | `swebench_verified` | `hf` | `patch_only` | `10` | Schema example showing `benchmark.params` and `evaluation.params`. |
| `configs/runs/local_jsonl_swebench.yaml` | `swebench_verified` | `local` | `patch_only` | `5` | Local JSONL task loading template. |
| `configs/runs/openrouter_free_swebench.yaml` | `swebench_verified` | `hf` | `patch_only` | `5` | Default run settings paired with `openrouter_free` agent. |
| `configs/runs/qwen3_coder_free_swebench.yaml` | `swebench_verified` | `hf` | `patch_only` | `10` | Default run settings paired with Qwen3 coder agent. |

#### Agent Presets (`agents/*.yaml`)

| File | `name` | Model | Retry/Backoff Overrides | Purpose |
|---|---|---|---|---|
| `agents/openrouter_free.yaml` | `openrouter-free` | `openrouter/free` | `max_retries=30`, `initial_backoff_s=0.5`, `max_backoff_s=4.0` | Dynamic free-model routing profile. |
| `agents/qwen2_5_coder.yaml` | `qwen2.5-coder-0.5b` | `qwen2.5-coder-0.5b-instruct` | backend defaults | Small Qwen coder profile. |
| `agents/qwen3_coder_free.yaml` | `qwen3-coder-free` | `qwen/qwen3-coder:free` | backend defaults | Qwen3 coder free profile. |
| `agents/qwen3_next_80b_free.yaml` | `qwen3-next-80b-a3b-instruct-free` | `qwen/qwen3-next-80b-a3b-instruct:free` | backend defaults | Larger Qwen3 Next profile. |
