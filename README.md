# Sub Agent Testing (Agent + Benchmark pipeline)

End-to-end pipeline for benchmarking code-editing agents on SWE-bench: it bootstraps the environment, composes an agent profile with a run config, runs the agent to generate patches, and evaluates them with the SWE-bench harness while saving outputs in a portable layout.

- Agent behavior comes from `profiles/agents/*.yaml` (model/backend, prompts, skills).
- Prompt text files live in `profiles/prompts/*.txt`.
- Execution policy and dataset wiring come from `profiles/runs/*.yaml`.
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

- Run configs: `profiles/runs/*.yaml`
- Agent profiles: `profiles/agents/*.yaml`

Default run config:
- `profiles/runs/default.yaml`

### 4) Run a quick example (from project root)

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run one quick prediction job:

```bash
agent run \
  --agent profiles/agents/openrouter_free.yaml \
  --run-config profiles/runs/swebench.yaml \
  --mode patch_only \
  --selector 10
```

`tools_enabled` runs require a local tool-ready repository workspace for the task dataset (for SWE-bench: use `benchmark.data_source=local` and set `benchmark.data_root` with repo checkouts under `<data_root>/<repo>`). HF-only task loading is suitable for `patch_only` runs but will now fail fast in `tools_enabled`.

Pick the latest predictions file:

```bash
PRED_PATH=$(find artifacts -type f -path "*/predictions.jsonl" | sort | tail -n 1)
echo "$PRED_PATH"
```

Evaluate it:

```bash
agent eval \
  "$PRED_PATH" \
  --run-config profiles/runs/swebench.yaml
```
