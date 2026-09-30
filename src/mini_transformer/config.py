from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.context_length, self.n_layers, self.n_heads) <= 0:
            raise ValueError("model dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 32
    context_length: int = 128
    max_steps: int = 2000
    eval_interval: int = 100
    eval_batches: int = 30
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    dropout: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    device: str = "auto"
    num_workers: int = 0
    checkpoint_interval: int = 100
    fixed_prompt: str = "ROMEO:"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def load_model_preset(path: Path, name: str, vocab_size: int, context_length: int) -> ModelConfig:
    raw = load_yaml(path)
    if name not in raw["models"]:
        raise KeyError(f"unknown model preset: {name}")
    values = {**raw.get("defaults", {}), **raw["models"][name]}
    return ModelConfig(vocab_size=vocab_size, context_length=context_length, **values)


def load_training_config(path: Path, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    values = load_yaml(path)
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return TrainingConfig(**values)
