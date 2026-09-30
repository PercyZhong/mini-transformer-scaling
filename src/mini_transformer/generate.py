from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .data import CharacterVocabulary
from .model import MiniTransformerLM
from .utils import resolve_device


def generate_text(
    model: MiniTransformerLM,
    vocabulary: CharacterVocabulary,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    encoded = vocabulary.encode(prompt)
    if not encoded:
        encoded = [vocabulary.stoi["<unk>"]]
    inputs = torch.tensor([encoded], dtype=torch.long, device=device)
    model.eval()
    result = model.generate(inputs, max_new_tokens, temperature, top_k)
    return vocabulary.decode(result[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocabulary = CharacterVocabulary(checkpoint["vocabulary"][1:])
    model = MiniTransformerLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(
        generate_text(
            model, vocabulary, args.prompt, device, args.tokens, args.temperature, args.top_k
        )
    )


if __name__ == "__main__":
    main()
