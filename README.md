# Portable Agent Runner

Portable benchmark runner with pluggable agent profiles (`agents/*.yaml`) and run configurations (`configs/runs/*.yaml`).

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

Practical mental model:
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

evaluation:
  enabled: true
  harness_cmd: python -m swebench.harness.run_evaluation
  eval_root: ./external/SWE-bench
  workdir: .
  report_dir: logs/reports

runtime:
  mode: patch_only
  selector: 10
  max_tool_calls: 20
  max_wall_time_s: 600

output:
  runs_dir: runs
  logs_dir: logs
```

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
  runs/<model>/<split>/<mode>/<timestamp>_predictions.jsonl \
  --run-config configs/runs/default.yaml
```

## Outputs

Predictions are written to:

```text
runs/<model>/<split>/<mode>/<YYYY-MM-DD_HHMM>_predictions.jsonl
```

Harness logs are written to this repo root (not `external/SWE-bench`):

```text
logs/run_evaluation/<run_id>/<model>/<instance_id>/
```

Harness reports are written to:

```text
logs/reports/
```

## Portability

Swap providers/models/datasets without code changes:
1. pick an agent profile in `agents/`
2. pick a run config in `configs/runs/`
3. run `agent run` with optional CLI overrides

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
