#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f data/raw/tiny_shakespeare.txt ]]; then
  echo "Dataset missing. Run: python scripts/download_data.py" >&2
  exit 2
fi

python -m mini_transformer.experiment \
  --matrix configs/experiment_matrix.yaml \
  --training configs/training.yaml \
  --data data/raw/tiny_shakespeare.txt \
  --output-root outputs
