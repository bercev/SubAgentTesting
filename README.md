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
  report_dir: reports

runtime:
  mode: patch_only
  selector: 10
  max_tool_calls: 20
  max_wall_time_s: 600

output:
  artifacts_dir: artifacts
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

`agent run` / `agent eval` are quiet by default. Use `--verbose` to print per-task validation details and full harness output.

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
