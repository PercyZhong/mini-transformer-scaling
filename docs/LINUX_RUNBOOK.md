# Linux runbook

## Install and smoke test

```bash
git clone <REPOSITORY_URL>
cd mini-transformer-scaling
python3 --version
nvidia-smi
df -h .
bash scripts/bootstrap_linux.sh
source .venv/bin/activate
python scripts/download_data.py
pytest -q
bash scripts/run_smoke.sh
```

For a CPU-only smoke test, `nvidia-smi` may be absent; continue with the same bootstrap and
smoke commands. `run_smoke.sh` explicitly selects CPU.

If PyTorch and the NVIDIA driver are incompatible, remove only this project's `.venv`, select
the official PyTorch wheel index compatible with the installed driver, and rerun:

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX
bash scripts/bootstrap_linux.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Formal matrix, monitoring, and recovery

```bash
source .venv/bin/activate
tmux new -s transformer-stage1
bash scripts/run_matrix.sh 2>&1 | tee outputs/matrix_console.log
```

In another terminal, use `watch -n 2 nvidia-smi`, `df -h .`, and inspect
`outputs/matrix_manifest.json` plus each `status.json`. Reattach with
`tmux attach -t transformer-stage1`. Rerunning the matrix resumes `latest.pt`; it skips a
completed run only when the config hash matches, and rejects reuse with a different config.

For out-of-memory errors, reduce `batch_size` and increase `gradient_accumulation_steps` by the
inverse factor. Preserve effective tokens per optimizer update and record the change in the
report. Runs remain sequential to avoid GPU-memory competition.

## Summarize and package

```bash
python scripts/summarize_results.py
bash scripts/package_results.sh
# Optional (shows estimated size before adding best checkpoints):
bash scripts/package_results.sh --include-best-checkpoints
sha256sum -c results_packages/SHA256SUMS.txt
```

Commit source, configuration, tests, scripts, and documentation. Never commit `.venv`, raw or
processed data, `outputs`, `results_packages`, caches, checkpoints, credentials, or private
keys. Copy result packages through an approved artifact channel instead.
