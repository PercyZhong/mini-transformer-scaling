#!/usr/bin/env python3
"""Create and independently verify the final Stage 1 closure ZIP."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from closure_checkpoints import FORMAL_RUNS, read_json, sha256_file, write_json
except ModuleNotFoundError:
    from scripts.closure_checkpoints import FORMAL_RUNS, read_json, sha256_file, write_json

RUN_FILES = (
    "config.resolved.yaml",
    "dataset_manifest.json",
    "environment.json",
    "metrics.jsonl",
    "samples.txt",
    "status.json",
    "summary.json",
    "checkpoint_reload_sample.txt",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_members(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe ZIP member: {name}")


def required_files(root: Path, include_finals: bool) -> list[Path]:
    files: list[Path] = []
    for directory in ("configs", "src"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file())
    for name in (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "scripts/closure_checkpoints.py",
        "scripts/package_closure.py",
    ):
        files.append(root / name)
    for run in FORMAL_RUNS:
        run_dir = root / "outputs" / run.run_id
        files.extend(run_dir / name for name in RUN_FILES)
        files.append(run_dir / "checkpoints" / "best.pt")
        if include_finals:
            files.append(run_dir / "checkpoints" / "final.pt")
    summary_dir = root / "outputs" / "summary"
    files.extend(
        path
        for path in summary_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".png", ".txt"}
    )
    unique = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    missing = [path.relative_to(root).as_posix() for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required closure files are missing: {missing}")
    return unique


def create_archive(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    checkpoint_manifest = read_json(root / "outputs/summary/checkpoint_manifest.json")
    if checkpoint_manifest.get("best_checkpoint_count") != 9:
        raise RuntimeError("checkpoint audit must pass before packaging")
    reload_tests = read_json(root / "outputs/summary/checkpoint_reload_tests.json")
    if len(reload_tests) != 9 or not all(
        test.get("strict_load") and test.get("generation_ok") for test in reload_tests
    ):
        raise RuntimeError("all nine independent reload tests must pass before packaging")
    files = required_files(root, args.include_finals)
    manifest_path = root / "outputs/summary/archive_file_manifest.json"
    archive_manifest: dict[str, Any] = {
        "schema_version": 1,
        "closure_commit": checkpoint_manifest["closure_commit"],
        "experiment_commit": checkpoint_manifest["experiment_commit"],
        "includes_best_checkpoints": True,
        "includes_final_checkpoints": args.include_finals,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "self_note": "This manifest excludes itself; the ZIP hash is stored in its sidecar.",
    }
    write_json(manifest_path, archive_manifest)
    files.append(manifest_path)
    closure_short = str(checkpoint_manifest["closure_commit"])[:7]
    output_dir = root / "results_packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"stage1-closure-{closure_short}.zip"
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive_path}")
    estimated_size = sum(path.stat().st_size for path in files)
    print(f"Estimated uncompressed size: {estimated_size / 1024 / 1024:.2f} MiB")
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            compression = zipfile.ZIP_STORED if path.suffix == ".pt" else zipfile.ZIP_DEFLATED
            archive.write(path, relative, compress_type=compression, compresslevel=6)
    digest = sha256_file(archive_path)
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(archive_path.relative_to(root)),
                "size_bytes": archive_path.stat().st_size,
                "sha256": digest,
                "sha256_sidecar": str(sidecar.relative_to(root)),
            },
            indent=2,
        )
    )
    return 0


def verify_archive(args: argparse.Namespace) -> int:
    archive_path = args.archive.resolve()
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(archive_path)
    if actual != expected:
        raise ValueError(f"ZIP SHA256 mismatch: expected {expected}, got {actual}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        safe_members(names)
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC failed for: {corrupt}")
        if args.extract_dir.exists():
            raise FileExistsError(f"refusing to reuse extraction directory: {args.extract_dir}")
        args.extract_dir.mkdir(parents=True)
        archive.extractall(args.extract_dir)
    extracted = args.extract_dir.resolve()
    manifest = read_json(extracted / "outputs/summary/archive_file_manifest.json")
    for entry in manifest["files"]:
        path = extracted / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"archived file is missing after extraction: {entry['path']}")
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"archived file verification failed: {entry['path']}")
    run = FORMAL_RUNS[0]
    result_path = extracted / "independent_archive_reload.json"
    sample_path = extracted / "independent_archive_sample.txt"
    command = [
        sys.executable,
        str(extracted / "scripts/closure_checkpoints.py"),
        "verify-one",
        "--checkpoint",
        str(extracted / f"outputs/{run.run_id}/checkpoints/best.pt"),
        "--sample-output",
        str(sample_path),
        "--result-output",
        str(result_path),
        "--run-id",
        run.run_id,
        "--seed",
        str(run.seed),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extracted / "src")
    subprocess.run(command, cwd=extracted, env=environment, check=True)
    result = read_json(result_path)
    if not (result["strict_load"] and result["generation_ok"]):
        raise RuntimeError("independent checkpoint verification from extracted ZIP failed")
    print(
        json.dumps(
            {
                "zip_sha256_ok": True,
                "zip_crc_ok": True,
                "manifest_files_verified": len(manifest["files"]),
                "extracted_to_new_directory": str(extracted),
                "independent_reload_generation_ok": True,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, default=project_root())
    create.add_argument("--include-finals", action="store_true")
    create.set_defaults(handler=create_archive)
    verify = subparsers.add_parser("verify")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--root", type=Path, default=project_root())
    verify.add_argument("--extract-dir", type=Path, required=True)
    verify.set_defaults(handler=verify_archive)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
