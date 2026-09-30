#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_transformer.data import CharacterVocabulary, split_text  # noqa: E402
from mini_transformer.utils import sha256_file  # noqa: E402

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and audit Tiny Shakespeare")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw_path = PROJECT_ROOT / "data/raw/tiny_shakespeare.txt"
    manifest_path = PROJECT_ROOT / "data/processed/dataset_manifest.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if not raw_path.exists() or args.force:
        try:
            with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
                content = response.read()
            raw_path.write_bytes(content)
            downloaded = True
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise SystemExit(
                f"Download failed: {error}\n"
                f"Manually place the canonical file at {raw_path} and rerun."
            ) from error
    text = raw_path.read_text(encoding="utf-8")
    splits = split_text(text)
    vocabulary = CharacterVocabulary.from_training_text(splits.train)
    manifest = {
        "source_url": DATASET_URL,
        "downloaded_this_run": downloaded,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(raw_path.relative_to(PROJECT_ROOT)),
        "bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
        "split_characters": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "split_ranges": {
            "train": list(splits.train_range),
            "validation": list(splits.validation_range),
            "test": list(splits.test_range),
        },
        "vocabulary": vocabulary.itos,
        "vocabulary_source": "train_only",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
