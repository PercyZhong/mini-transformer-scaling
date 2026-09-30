# Mini Transformer Scaling

A from-scratch, character-level decoder-only Transformer and reproducible 3-model ×
3-data-scale teaching experiment. The formal experiment is intended for Linux Python 3.11;
Windows is used only for authoring, syntax checks, unit tests, and Git operations.

This study examines small-scale trends. It does **not** establish a general scaling law.

## What is implemented

- Learned token and positional embeddings
- Explicit Q/K/V projections, scaled dot-product causal multi-head attention, and head merge
- Pre-LayerNorm residual Transformer blocks and GELU feed-forward networks
- Optional tied token/output weights, dropout, initialization, parameter counting
- AdamW, warmup/cosine schedule, gradient clipping/accumulation, CUDA AMP
- Deterministic data audit, fixed splits, nested 10%/30%/100% train-window subsets
- Evaluation, perplexity, temperature/top-k generation, resumable checkpoints
- Sequential 3×3 matrix, summaries, plots, report scaffold, and artifact packaging

## Fresh Linux setup through smoke test

Choose a PyTorch wheel index compatible with the server driver when needed. Do not copy a
CUDA minor version blindly; consult the current PyTorch installation selector.

```bash
git clone <REPOSITORY_URL>
cd mini-transformer-scaling
python3 --version
nvidia-smi

# Optional, for a server-specific official PyTorch wheel index:
# export TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX

bash scripts/bootstrap_linux.sh
source .venv/bin/activate
python scripts/download_data.py
pytest -q
bash scripts/run_smoke.sh
```

Only after both tests and smoke pass, run the formal matrix:

```bash
bash scripts/run_matrix.sh
python scripts/summarize_results.py
bash scripts/package_results.sh
```

The 9 formal runs use the same optimizer settings, effective batch/token budget, context
length, evaluation cadence, and seed. If memory requires a smaller physical batch, increase
`gradient_accumulation_steps` so
`batch_size × gradient_accumulation_steps × context_length` remains unchanged.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m ruff check .
python -m pytest -q
```

Generated datasets, environments, caches, outputs, result packages, checkpoints, and secrets
are excluded from Git. See [Linux runbook](docs/LINUX_RUNBOOK.md) and
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md).
