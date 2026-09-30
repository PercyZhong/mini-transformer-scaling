from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import ModelConfig
from .data import InfiniteBatchIterator, prepare_data
from .model import MiniTransformerLM
from .utils import resolve_device


@torch.no_grad()
def evaluate_loss(
    model: MiniTransformerLM,
    batches: InfiniteBatchIterator,
    device: torch.device,
    num_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        inputs, targets = next(batches)
        _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/raw/tiny_shakespeare.txt"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    prepared = prepare_data(args.data, model_config.context_length, checkpoint["data_fraction"])
    model = MiniTransformerLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    batches = InfiniteBatchIterator(prepared.datasets[args.split], 32, 1042)
    loss = evaluate_loss(model, batches, device, args.batches)
    print(f"{args.split}_loss={loss:.6f} perplexity={math.exp(loss):.6f}")


if __name__ == "__main__":
    main()
