# Stage 1 multi-seed 20k Linux runbook

This extension is isolated from the closed Stage 1 outputs. It uses physical GPUs 0, 1, and 2
for seeds 42, 43, and 44 respectively. GPU 3 is forbidden. Run every command from the repository
root with the Python 3.11 environment activated.

## Preflight

```bash
git status --short --branch
git switch experiment/stage1-multiseed-20k
git pull --ff-only
python --version
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python scripts/run_multiseed_20k.py --dry-run
nvidia-smi
```

The scheduler independently checks for compute processes on physical GPUs 0–2 and refuses to
start if any are occupied. It never selects GPU 3.

## Three-GPU smoke test

```bash
bash scripts/run_multiseed_smoke.sh
```

Inspect `outputs_multiseed_20k_smoke/scheduler_manifest.json` and the three files under
`scheduler_logs/`. Re-run `nvidia-smi`; formal training may start only if GPUs 0–2 are idle.

## Formal matrix

Use tmux so SSH loss does not terminate the scheduler:

```bash
tmux new -s stage1-ms20k
bash scripts/run_multiseed_formal.sh
```

Detach with `Ctrl-b`, then `d`. Reattach with `tmux attach -t stage1-ms20k`. To safely continue
an interrupted matrix after inspecting its status, run:

```bash
python scripts/run_multiseed_20k.py --resume
```

Only a run whose completed status, summary, token budget, and both loadable checkpoints pass is
skipped. Other runs resume from their own `latest.pt`; no original Stage 1 checkpoint is used.

## Summarize, validate, and archive

```bash
python scripts/summarize_multiseed_20k.py
python scripts/validate_multiseed_20k.py
python scripts/package_multiseed_20k.py create
```

Use the two exact ZIP paths printed by the create command:

```bash
python scripts/package_multiseed_20k.py verify results_packages/<analysis.zip> \
  --extract-dir results_packages/verify-analysis-new
python scripts/package_multiseed_20k.py verify results_packages/<full.zip> \
  --extract-dir results_packages/verify-full-new
```

The analysis ZIP excludes checkpoint bodies. The full ZIP contains all 54 best/final
checkpoints and remains server-local. Both ZIPs have `.zip.sha256` sidecars.
