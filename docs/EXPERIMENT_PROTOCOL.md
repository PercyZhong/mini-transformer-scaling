# Experiment protocol

## Objective

Compare teaching-scale model and data trends for a character-level language model without
claiming a general scaling law.

## Controlled design

The model presets are tiny (64 dimensions, 2 layers, 2 heads, FFN 256), small (128, 4, 4,
512), and medium (256, 6, 8, 1024). Each is trained on deterministic nested subsets containing
10%, 30%, or 100% of the available training windows. The ordered corpus split is fixed at
80%/10%/10%; validation and test never shrink. Seed 42 generates one window permutation, so
10% is a subset of 30%, which is a subset of 100%.

All nine runs use batch size 32, context 128, 2,000 optimizer steps, AdamW, a 3e-4 peak
learning rate, warmup/cosine decay, weight decay 0.01, dropout 0.1, clipping at 1.0, and the
same evaluation schedule. With memory-driven adjustment, preserve effective tokens per update.

## Audit and acceptance

Every run records its resolved configuration, config hash, environment, Git commit, dataset
SHA256, selected-window hash, JSONL metrics, status, checkpoints, summary, and fixed-prompt
sample. A completed matrix has 9/9 completed runs, no non-finite loss, five plots, an empty
failed-runs list, and a package checksum. Formal claims must be traceable to these artifacts.
