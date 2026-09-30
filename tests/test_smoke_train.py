import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from mini_transformer.config import TrainingConfig
from mini_transformer.train import run_training


def test_smoke_training_produces_minimum_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = tmp_path / "tiny.txt"
    data_path.write_text(("To be, or not to be.\n" * 80), encoding="utf-8")
    output = tmp_path / "outputs/smoke"
    config = TrainingConfig(
        batch_size=2,
        context_length=8,
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        warmup_steps=1,
        mixed_precision=False,
        device="cpu",
        checkpoint_interval=1,
    )
    summary = run_training(
        root=root,
        data_path=data_path,
        output_dir=output,
        model_size="tiny",
        data_fraction=0.1,
        training=config,
    )
    expected = [
        "config.resolved.yaml", "environment.json", "dataset_manifest.json",
        "metrics.jsonl", "summary.json", "samples.txt", "status.json",
        "checkpoints/best.pt", "checkpoints/latest.pt", "checkpoints/final.pt",
    ]
    assert all((output / item).exists() for item in expected)
    assert summary["status"] == "completed"
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["state"] == "completed"
