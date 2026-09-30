#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; CPU setup remains supported."
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
  python -m pip install -e . --no-deps
  python -m pip install PyYAML matplotlib pytest ruff
else
  python -m pip install -e .
  python -m pip install -r requirements-dev.txt
fi
python scripts/collect_environment.py
