#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating .venv with uv..."
  uv venv .venv
else
  echo ".venv already exists; reusing it."
fi

echo "Installing runner dependencies..."
uv pip install --python .venv/bin/python -e .

if [ ! -d "external/SWE-bench/.git" ]; then
  echo "Cloning SWE-bench into external/SWE-bench..."
  mkdir -p external
  git clone https://github.com/swe-bench/SWE-bench.git external/SWE-bench
else
  echo "SWE-bench already present at external/SWE-bench; skipping clone."
fi

echo "Installing SWE-bench in the same environment..."
uv pip install --python .venv/bin/python -e ./external/SWE-bench

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  agent run --agent agents/openrouter_free.yaml --run-config configs/runs/default.yaml --mode patch_only --selector 10"
echo "  agent eval runs/<model>/<split>/<mode>/<timestamp>_predictions.jsonl --run-config configs/runs/default.yaml"
