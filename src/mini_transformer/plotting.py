from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _finish(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_loss_curves(runs: list[dict[str, Any]], output: Path) -> None:
    for run in runs:
        metrics = run.get("metrics", [])
        if metrics:
            steps = [row["step"] for row in metrics]
            plt.plot(steps, [row["train_loss"] for row in metrics], "--", alpha=0.7)
            plt.plot(steps, [row["val_loss"] for row in metrics], label=run["run_id"])
    plt.title("Training and validation loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Cross-entropy loss")
    if runs:
        plt.legend(fontsize=6, ncol=2)
    _finish(output)


def plot_grouped(
    rows: list[dict[str, Any]], group: str, x: str, y: str, title: str, xlabel: str,
    ylabel: str, output: Path,
) -> None:
    groups = sorted({str(row[group]) for row in rows})
    for value in groups:
        selected = sorted((row for row in rows if str(row[group]) == value), key=lambda row: row[x])
        plt.plot(
            [row[x] for row in selected],
            [row[y] for row in selected],
            marker="o",
            label=value,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if groups:
        plt.legend(title=group)
    _finish(output)


def plot_compute_tradeoff(rows: list[dict[str, Any]], output: Path) -> None:
    if rows:
        points = plt.scatter(
            [row["training_seconds"] for row in rows],
            [row["best_val_loss"] for row in rows],
            s=[max(20, row["parameter_count"] / 10000) for row in rows],
            c=[row["data_fraction"] for row in rows],
            cmap="viridis",
        )
        plt.colorbar(points, label="Training data fraction")
    plt.title("Compute/effect tradeoff (marker size = parameters)")
    plt.xlabel("Training time (seconds)")
    plt.ylabel("Best validation loss")
    _finish(output)
