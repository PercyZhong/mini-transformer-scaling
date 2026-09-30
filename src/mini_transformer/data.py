from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .utils import sha256_file, stable_hash

UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TextSplits:
    train: str
    validation: str
    test: str
    train_range: tuple[int, int]
    validation_range: tuple[int, int]
    test_range: tuple[int, int]


def split_text(text: str, train_fraction: float = 0.8, val_fraction: float = 0.1) -> TextSplits:
    if not text:
        raise ValueError("dataset is empty")
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("invalid split fractions")
    train_end = int(len(text) * train_fraction)
    val_end = train_end + int(len(text) * val_fraction)
    if train_end == 0 or val_end <= train_end or val_end >= len(text):
        raise ValueError("dataset is too short for non-empty 80/10/10 splits")
    return TextSplits(
        train=text[:train_end],
        validation=text[train_end:val_end],
        test=text[val_end:],
        train_range=(0, train_end),
        validation_range=(train_end, val_end),
        test_range=(val_end, len(text)),
    )


class CharacterVocabulary:
    def __init__(self, characters: list[str]) -> None:
        unique = sorted(set(characters))
        self.itos = [UNK_TOKEN, *[char for char in unique if char != UNK_TOKEN]]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def from_training_text(cls, text: str) -> CharacterVocabulary:
        return cls(list(text))

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi[UNK_TOKEN]
        return [self.stoi.get(char, unknown) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids)

    def __len__(self) -> int:
        return len(self.itos)


def all_window_starts(token_count: int, context_length: int) -> list[int]:
    count = token_count - context_length
    if count <= 0:
        raise ValueError("split must contain more tokens than context_length")
    return list(range(count))


def nested_window_subsets(
    token_count: int,
    context_length: int,
    fractions: tuple[float, ...] = (0.1, 0.3, 1.0),
    seed: int = 42,
) -> dict[float, list[int]]:
    starts = torch.tensor(all_window_starts(token_count, context_length), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    permutation = starts[torch.randperm(len(starts), generator=generator)].tolist()
    result: dict[float, list[int]] = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("data fractions must be in (0, 1]")
        count = len(permutation) if fraction == 1.0 else max(1, int(len(permutation) * fraction))
        result[fraction] = permutation[:count]
    return result


class WindowDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, tokens: list[int], context_length: int, starts: list[int] | None = None
    ) -> None:
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        self.starts = (
            starts if starts is not None else all_window_starts(len(tokens), context_length)
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        stop = start + self.context_length
        return self.tokens[start:stop], self.tokens[start + 1 : stop + 1]


class InfiniteBatchIterator:
    """Deterministic replacement sampler with serializable generator state."""

    def __init__(self, dataset: WindowDataset, batch_size: int, seed: int) -> None:
        if len(dataset) == 0:
            raise ValueError("dataset contains no windows")
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self.dataset), (self.batch_size,), generator=self.generator)
        pairs = [self.dataset[int(index)] for index in indices]
        return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])


@dataclass
class PreparedData:
    vocabulary: CharacterVocabulary
    splits: TextSplits
    datasets: dict[str, WindowDataset]
    manifest: dict[str, object]


def prepare_data(
    path: Path, context_length: int, data_fraction: float, seed: int = 42
) -> PreparedData:
    text = path.read_text(encoding="utf-8")
    splits = split_text(text)
    vocabulary = CharacterVocabulary.from_training_text(splits.train)
    encoded = {
        "train": vocabulary.encode(splits.train),
        "validation": vocabulary.encode(splits.validation),
        "test": vocabulary.encode(splits.test),
    }
    subsets = nested_window_subsets(len(encoded["train"]), context_length, seed=seed)
    matching = [fraction for fraction in subsets if abs(fraction - data_fraction) < 1e-9]
    if not matching:
        raise ValueError(f"data_fraction must be one of {list(subsets)}")
    selected = subsets[matching[0]]
    datasets = {
        "train": WindowDataset(encoded["train"], context_length, selected),
        "validation": WindowDataset(encoded["validation"], context_length),
        "test": WindowDataset(encoded["test"], context_length),
    }
    manifest: dict[str, object] = {
        "dataset_path": str(path),
        "dataset_sha256": sha256_file(path),
        "split_character_counts": {key: len(value) for key, value in encoded.items()},
        "split_ranges": {
            "train": list(splits.train_range),
            "validation": list(splits.validation_range),
            "test": list(splits.test_range),
        },
        "effective_windows": {key: len(value) for key, value in datasets.items()},
        "all_train_windows": len(all_window_starts(len(encoded["train"]), context_length)),
        "selected_train_windows": len(selected),
        "data_fraction": data_fraction,
        "seed": seed,
        "context_length": context_length,
        "vocabulary": vocabulary.itos,
        "vocabulary_source": "train_only",
        "selected_window_hash": stable_hash(selected),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return PreparedData(vocabulary, splits, datasets, manifest)


def load_download_manifest(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
