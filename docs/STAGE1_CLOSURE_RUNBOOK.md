# Stage 1 checkpoint closure runbook

Run closure only in the original Linux repository containing the ignored `outputs/` tree.
The audit is read-only with respect to existing metrics and checkpoints. It never trains.

```bash
python -m pytest -q
python -m ruff check .
python scripts/closure_checkpoints.py audit
python scripts/package_closure.py create --include-finals
```

Use the archive name printed by the create command:

```bash
unzip -t results_packages/stage1-closure-<closure-short-commit>.zip
sha256sum -c results_packages/stage1-closure-<closure-short-commit>.zip.sha256
python scripts/package_closure.py verify \
  results_packages/stage1-closure-<closure-short-commit>.zip \
  --extract-dir results_packages/verify-stage1-closure-<closure-short-commit>
```

The audit requires all nine formal runs to be completed and all nine `best.pt` files to be
present and non-empty. Missing required checkpoints cause an immediate failure; this script
does not rerun training. It also inventories `final.pt` when present, hashes every archived
checkpoint, and spawns a fresh Python process for each strict best-checkpoint reload and
`ROMEO:` generation test.

Checkpoints and `results_packages/` remain ignored by ordinary Git. Do not force-add them.
