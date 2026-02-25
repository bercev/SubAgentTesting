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

if [ ! -d "data/swebench_repos/astropy/astropy/.git" ]; then
  echo "Cloning Astropy repo into data/swebench_repos/astropy/astropy for tools-enabled runs..."
  mkdir -p data/swebench_repos/astropy
  git clone https://github.com/astropy/astropy.git data/swebench_repos/astropy/astropy
else
  echo "Astropy repo already present at data/swebench_repos/astropy/astropy; skipping clone."
fi

echo "Installing SWE-bench in the same environment..."
uv pip install --python .venv/bin/python -e ./external/SWE-bench

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  agent run --agent profiles/agents/openrouter_free.yaml --run-config profiles/runs/default.yaml --mode patch_only --selector 10"
echo "  agent eval artifacts/<run_id>/predictions.jsonl --run-config profiles/runs/default.yaml"
echo "  # tools_enabled (HF tasks + local repos, astropy-only)"
echo "  agent run --agent profiles/agents/gemini_2.5_flash_tools.yaml --run-config profiles/runs/swebench_tools_hf_astropy.yaml --mode tools_enabled --selector 3 --full-log-previews"
