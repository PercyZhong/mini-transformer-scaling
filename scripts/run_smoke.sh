#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f data/raw/tiny_shakespeare.txt ]]; then
  python scripts/download_data.py
fi

python -m mini_transformer.train \
  --model-size tiny \
  --data-fraction 0.1 \
  --data data/raw/tiny_shakespeare.txt \
  --output outputs/smoke-tiny-010pct \
  --device cpu \
  --no-mixed-precision \
  --context-length 32 \
  --batch-size 4 \
  --max-steps 10 \
  --eval-interval 5 \
  --eval-batches 2
