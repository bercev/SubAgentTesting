# Portable Agent Runner

Python + uv-based agent runtime supporting model backends and benchmark adapters. First target: Qwen2.5-Coder-0.5B via OpenRouter and SWE-bench Verified.

## Setup

```
pip install -e .
cp .env.example .env  # fill OPENROUTER_API_KEY, paths
```

## Commands

- `agent list` — list agents/benchmarks
- `agent run --mode A` — patch-only smoke (no tools)
- `agent run --mode B` — tools-enabled run
- `agent predict` — batch Mode A to predictions.jsonl
- `agent eval runs/predictions.jsonl` — run official harness (set `HARNESS_CMD`)

## Tests

```
pytest
```
