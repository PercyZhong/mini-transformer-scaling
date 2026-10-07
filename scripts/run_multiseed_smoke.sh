#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/run_multiseed_20k.py \
  --config configs/experiment_multiseed_20k.yaml \
  --output-root outputs_multiseed_20k_smoke \
  --smoke \
  "$@"
