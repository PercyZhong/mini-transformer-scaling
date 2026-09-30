#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_transformer.plotting import (  # noqa: E402
    plot_compute_tradeoff,
    plot_grouped,
    plot_loss_curves,
)

FIELDS = [
    "run_id", "model_size", "data_fraction", "seed", "parameter_count",
    "unique_train_windows", "tokens_processed", "best_step", "best_val_loss",
    "final_train_loss", "final_val_loss", "final_val_perplexity", "training_seconds",
    "tokens_per_second", "peak_gpu_memory_mb", "best_checkpoint", "final_checkpoint",
    "status", "git_commit", "config_hash", "dataset_hash",
]


def load_runs(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    run_dirs = (
        path for path in output_root.iterdir() if path.is_dir() and path.name != "summary"
    )
    for run_dir in sorted(run_dirs):
        status_path = run_dir / "status.json"
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics_path = run_dir / "metrics.jsonl"
            row["metrics"] = [
                json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ] if metrics_path.exists() else []
            completed.append(row)
        elif status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            failed.append({"run_id": run_dir.name, **status})
    return completed, failed


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No completed runs.\n"
    columns = [
        "run_id", "parameter_count", "data_fraction", "best_val_loss",
        "final_val_perplexity", "training_seconds", "status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def make_report(rows: list[dict[str, Any]], failed: list[dict[str, Any]]) -> str:
    completion = f"{len(rows)}/9"
    state = "Complete" if len(rows) == 9 and not failed else "INCOMPLETE"
    return f"""# Stage 1 Experiment Report

> Status: **{state}** ({completion} formal runs completed). Conclusions below must only be
> added after reviewing traceable metrics. This is a teaching-scale trend experiment, not a
> claim of a general scaling law.

## Objective and architecture

Character-level decoder-only Transformer with learned token/position embeddings, explicit
causal multi-head self-attention, Pre-LN residual blocks, and a feed-forward network.

## Data and split

Tiny Shakespeare, sequential 80/10/10 train/validation/test split. Vocabulary is built from
training text only; 10%, 30%, and 100% training-window subsets are deterministic and nested.

## Experiment matrix and results

{markdown_table(rows)}

## Software and hardware environment

See each run's `environment.json`, resolved config, dataset manifest, and Git commit.

## Figures

- `loss_curves.png`
- `model_scale_comparison.png`
- `data_scale_comparison.png`
- `perplexity_comparison.png`
- `compute_tradeoff.png`

## Fixed-prompt samples

See each completed run's `samples.txt`.

## Model-scale, data-scale, time, throughput, and memory analysis

**Not completed automatically.** Interpret only the measured rows and figures above.

## Anomalies and failures

Failed or unfinished runs: {len(failed)}. See `failed_runs.json`.

## Limitations

Single seed, character-level corpus, small models, fixed optimization budget, and no basis for
general large-scale scaling-law claims.

## Next stage

After 9/9 completion, audit artifacts and metrics, add evidence-based observations, and test
multiple seeds as an optional extension.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real experiment artifacts")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not args.output_root.exists():
        raise SystemExit(f"Output root does not exist: {args.output_root}")
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = load_runs(args.output_root)
    with (summary_dir / "experiment_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (summary_dir / "experiment_summary.md").write_text(markdown_table(rows), encoding="utf-8")
    (summary_dir / "failed_runs.json").write_text(
        json.dumps(failed, indent=2) + "\n", encoding="utf-8"
    )
    plot_loss_curves(rows, summary_dir / "loss_curves.png")
    plot_grouped(
        rows, "data_fraction", "parameter_count", "best_val_loss",
        "Model scale comparison", "Trainable parameters", "Best validation loss",
        summary_dir / "model_scale_comparison.png",
    )
    plot_grouped(
        rows, "model_size", "data_fraction", "best_val_loss",
        "Data scale comparison", "Training data fraction", "Best validation loss",
        summary_dir / "data_scale_comparison.png",
    )
    plot_grouped(
        rows, "model_size", "data_fraction", "final_val_perplexity",
        "Validation perplexity", "Training data fraction", "Validation perplexity",
        summary_dir / "perplexity_comparison.png",
    )
    plot_compute_tradeoff(rows, summary_dir / "compute_tradeoff.png")
    (summary_dir / "STAGE1_REPORT.md").write_text(make_report(rows, failed), encoding="utf-8")
    print(f"Completed runs: {len(rows)}/9; unfinished/failed: {len(failed)}")


if __name__ == "__main__":
    main()
