from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml

EXPECTED_MODELS = ("tiny", "small", "medium")
EXPECTED_FRACTIONS = (0.1, 0.3, 1.0)
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_GPU_MAP = {42: 0, 43: 1, 44: 2}
FORMAL_STEPS = tuple(range(2000, 20001, 2000))


@dataclass(frozen=True)
class MultiseedRun:
    model_size: str
    data_fraction: float
    seed: int
    physical_gpu_id: int

    @property
    def run_id(self) -> str:
        fraction = int(round(self.data_fraction * 100))
        return f"{self.model_size}-{fraction:03d}pct-seed{self.seed}"


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    gpu_map = {int(seed): int(gpu) for seed, gpu in config["physical_gpu_by_seed"].items()}
    normalized = {
        **config,
        "seeds": [int(value) for value in config["seeds"]],
        "models": [str(value) for value in config["models"]],
        "data_fractions": [float(value) for value in config["data_fractions"]],
        "physical_gpu_by_seed": gpu_map,
    }
    validate_experiment_config(normalized)
    return normalized


def validate_experiment_config(config: dict[str, Any]) -> None:
    if tuple(config["seeds"]) != EXPECTED_SEEDS:
        raise ValueError(f"formal seeds must be {EXPECTED_SEEDS}")
    if tuple(config["models"]) != EXPECTED_MODELS:
        raise ValueError(f"formal models must be {EXPECTED_MODELS}")
    if tuple(config["data_fractions"]) != EXPECTED_FRACTIONS:
        raise ValueError(f"formal data fractions must be {EXPECTED_FRACTIONS}")
    required = {
        "max_steps": 20000,
        "eval_interval": 2000,
        "eval_batches": 50,
        "checkpoint_interval": 2000,
        "output_root": "outputs_multiseed_20k",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}, got {config.get(key)!r}")
    if config["physical_gpu_by_seed"] != EXPECTED_GPU_MAP:
        raise ValueError(f"physical GPU map must be {EXPECTED_GPU_MAP}")
    if 3 in config["physical_gpu_by_seed"].values():
        raise ValueError("physical GPU 3 is forbidden")


def expand_runs(config: dict[str, Any]) -> list[MultiseedRun]:
    runs = [
        MultiseedRun(model, fraction, seed, config["physical_gpu_by_seed"][seed])
        for seed in config["seeds"]
        for model in config["models"]
        for fraction in config["data_fractions"]
    ]
    if len(runs) != 27 or len({run.run_id for run in runs}) != 27:
        raise ValueError("the formal matrix must contain exactly 27 unique runs")
    return runs
