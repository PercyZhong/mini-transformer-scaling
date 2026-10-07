#!/usr/bin/env python3
"""Create, checksum, extract, and independently verify analysis/full result ZIPs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_transformer.multiseed import expand_runs, load_experiment_config  # noqa: E402
from mini_transformer.utils import sha256_file  # noqa: E402

RUN_FILES = (
    "config.resolved.yaml", "environment.json", "dataset_manifest.json", "metrics.jsonl",
    "samples.txt", "status.json", "summary.json",
)


def git_short(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()


def archive_files(root: Path, output_root: Path, config: Path, full: bool) -> list[Path]:
    runs = expand_runs(load_experiment_config(config))
    files: list[Path] = []
    for directory in ("configs", "src"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file())
    for path in (root / "scripts").glob("*multiseed_20k.py"):
        files.append(path)
    files.extend([root / "pyproject.toml", root / "requirements.txt", root / "README.md"])
    for run in runs:
        run_dir = output_root / run.run_id
        files.extend(run_dir / name for name in RUN_FILES)
        if full:
            files.extend(run_dir / "checkpoints" / f"{kind}.pt" for kind in ("best", "final"))
    files.extend(path for path in (output_root / "summary").iterdir() if path.is_file())
    files.extend(path for path in (output_root / "scheduler_logs").iterdir() if path.is_file())
    files.extend([output_root / "scheduler_manifest.json", output_root / "original_stage1_manifest.before.json"])
    unique = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    missing = [path.relative_to(root).as_posix() for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"archive inputs are missing: {missing}")
    return unique


def create_one(root: Path, output_root: Path, config: Path, package_dir: Path, full: bool) -> dict[str, Any]:
    kind = "full" if full else "analysis"
    files = archive_files(root, output_root, config, full)
    manifest = {
        "kind": kind,
        "checkpoint_bodies_included": full,
        "files": [{"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }
    manifest_path = output_root / "summary" / f"{kind}_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    files.append(manifest_path)
    archive = package_dir / f"stage1-multiseed-20k-{kind}-{git_short(root)}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite: {archive}")
    with zipfile.ZipFile(archive, "w", allowZip64=True) as handle:
        for path in files:
            compression = zipfile.ZIP_STORED if path.suffix == ".pt" else zipfile.ZIP_DEFLATED
            handle.write(path, path.relative_to(root).as_posix(), compress_type=compression, compresslevel=6)
    digest = sha256_file(archive)
    sidecar = archive.with_suffix(".zip.sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return {"kind": kind, "archive": str(archive), "size_bytes": archive.stat().st_size, "sha256": digest, "sidecar": str(sidecar)}


def safe_names(names: list[str]) -> None:
    for name in names:
        member = PurePosixPath(name.replace("\\", "/"))
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe ZIP member: {name}")


def verify_archive(archive: Path, extract_dir: Path) -> dict[str, Any]:
    sidecar = archive.with_suffix(".zip.sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(archive)
    if actual != expected:
        raise ValueError(f"ZIP SHA256 mismatch: {archive}")
    if extract_dir.exists():
        raise FileExistsError(f"refusing to reuse extraction directory: {extract_dir}")
    with zipfile.ZipFile(archive) as handle:
        safe_names(handle.namelist())
        corrupt = handle.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC failure: {corrupt}")
        extract_dir.mkdir(parents=True)
        handle.extractall(extract_dir)
    kind = "full" if "-full-" in archive.name else "analysis"
    manifest_path = extract_dir / "outputs_multiseed_20k" / "summary" / f"{kind}_archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = extract_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"extracted file verification failed: {entry['path']}")
    archived_paths = {entry["path"] for entry in manifest["files"]}
    if kind == "analysis" and any(path.endswith(".pt") for path in archived_paths):
        raise ValueError("analysis archive unexpectedly contains checkpoint bodies")
    summary_dir = extract_dir / "outputs_multiseed_20k" / "summary"
    validation = json.loads((summary_dir / "validation_report.json").read_text(encoding="utf-8"))
    if not validation.get("passed") or validation.get("completed") != 27:
        raise ValueError("archived validation report is not a 27/27 pass")
    expected_rows = {
        "experiment_summary_27runs.csv": 27,
        "condition_summary.csv": 9,
        "paired_differences.csv": 18,
    }
    for filename, expected_count in expected_rows.items():
        with (summary_dir / filename).open(encoding="utf-8", newline="") as handle:
            if len(list(csv.DictReader(handle))) != expected_count:
                raise ValueError(f"archived table row count is invalid: {filename}")
    config = extract_dir / "configs/experiment_multiseed_20k.yaml"
    runs = expand_runs(load_experiment_config(config))
    for run in runs:
        metrics_path = extract_dir / "outputs_multiseed_20k" / run.run_id / "metrics.jsonl"
        steps = [
            int(json.loads(line)["step"])
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if steps != list(range(2000, 20001, 2000)):
            raise ValueError(f"archived formal evaluation points are invalid: {run.run_id}")
    strict_loads = 0
    generation_tests = 0
    if kind == "full":
        for run in runs:
            for checkpoint_type in ("best", "final"):
                checkpoint = extract_dir / "outputs_multiseed_20k" / run.run_id / "checkpoints" / f"{checkpoint_type}.pt"
                command = [sys.executable, str(extract_dir / "scripts/validate_multiseed_20k.py"), "verify-one", str(checkpoint)]
                if checkpoint_type == "best":
                    command.append("--generate")
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(extract_dir / "src")
                subprocess.run(command, cwd=extract_dir, env=environment, check=True, capture_output=True, text=True)
                strict_loads += 1
                generation_tests += checkpoint_type == "best"
    return {
        "archive": str(archive),
        "sha256_ok": True,
        "zip_crc_ok": True,
        "manifest_files_verified": len(manifest["files"]),
        "metadata_acceptance_passed": True,
        "extracted_to": str(extract_dir),
        "strict_checkpoint_loads": strict_loads,
        "best_generation_tests": generation_tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--config", type=Path, default=Path("configs/experiment_multiseed_20k.yaml"))
    create.add_argument("--output-root", type=Path, default=Path("outputs_multiseed_20k"))
    create.add_argument("--package-dir", type=Path, default=Path("results_packages"))
    verify = subparsers.add_parser("verify")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--extract-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        args.config = args.config.resolve()
        args.output_root = args.output_root.resolve()
        args.package_dir = args.package_dir.resolve()
        validation = json.loads((args.output_root / "summary/validation_report.json").read_text(encoding="utf-8"))
        if not validation.get("passed"):
            raise RuntimeError("automatic validation must pass before packaging")
        args.package_dir.mkdir(parents=True, exist_ok=True)
        result = [create_one(PROJECT_ROOT, args.output_root, args.config, args.package_dir, full) for full in (False, True)]
    else:
        result = verify_archive(args.archive.resolve(), args.extract_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
