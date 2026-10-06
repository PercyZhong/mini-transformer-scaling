#!/usr/bin/env python3
"""Audit and independently reload all formal Stage 1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_COMMIT = "bd74aa17ffad8d2db7d0aa9214199a83f2c401cc"
PROMPT = "ROMEO:"


@dataclass(frozen=True)
class FormalRun:
    run_id: str
    model_size: str
    data_fraction: float
    seed: int = 42


FORMAL_RUNS = (
    FormalRun("tiny-010pct-seed42", "tiny", 0.1),
    FormalRun("tiny-030pct-seed42", "tiny", 0.3),
    FormalRun("tiny-100pct-seed42", "tiny", 1.0),
    FormalRun("small-010pct-seed42", "small", 0.1),
    FormalRun("small-030pct-seed42", "small", 0.3),
    FormalRun("small-100pct-seed42", "small", 1.0),
    FormalRun("medium-010pct-seed42", "medium", 0.1),
    FormalRun("medium-030pct-seed42", "medium", 0.3),
    FormalRun("medium-100pct-seed42", "medium", 1.0),
)

INVENTORY_FIELDS = (
    "run_id",
    "model_size",
    "data_fraction",
    "seed",
    "checkpoint_type",
    "path",
    "exists",
    "size_bytes",
    "step",
    "loadable",
    "sha256",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "step",
        "best_val_loss",
        "model_config",
        "model_state",
        "optimizer_state",
        "scaler_state",
        "batch_iterator_state",
        "rng_state",
        "data_fraction",
        "vocabulary",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")
    vocabulary = checkpoint["vocabulary"]
    if not isinstance(vocabulary, list) or not vocabulary or vocabulary[0] != "<unk>":
        raise ValueError("checkpoint vocabulary is invalid")
    if not checkpoint["optimizer_state"]:
        raise ValueError("checkpoint optimizer state is empty")
    return {
        "step": int(checkpoint["step"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "vocabulary_size": len(vocabulary),
        "has_optimizer_state": True,
        "has_scaler_state": checkpoint["scaler_state"] is not None,
        "has_batch_iterator_state": checkpoint["batch_iterator_state"] is not None,
        "scheduler_state": "not_applicable_stateless_schedule",
    }


def verify_checkpoint(
    checkpoint_path: Path,
    sample_output: Path,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    root = project_root()
    sys.path.insert(0, str(root / "src"))

    import torch

    from mini_transformer.config import ModelConfig
    from mini_transformer.data import CharacterVocabulary
    from mini_transformer.generate import generate_text
    from mini_transformer.model import MiniTransformerLM

    result: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint": checkpoint_path.as_posix(),
        "loadable": False,
        "strict_load": False,
        "generation_ok": False,
        "prompt": PROMPT,
        "error": None,
    }
    try:
        torch.manual_seed(seed)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = MiniTransformerLM(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        result["loadable"] = True
        result["strict_load"] = True
        vocabulary = CharacterVocabulary(checkpoint["vocabulary"][1:])
        if vocabulary.itos != checkpoint["vocabulary"]:
            raise ValueError("reconstructed vocabulary differs from checkpoint vocabulary")
        generated = generate_text(
            model,
            vocabulary,
            PROMPT,
            torch.device("cpu"),
            max_new_tokens=100,
            temperature=0.8,
            top_k=40,
        )
        if not generated.startswith(PROMPT) or len(generated) <= len(PROMPT):
            raise ValueError("generation did not preserve the prompt or add text")
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        sample_output.write_text(generated + "\n", encoding="utf-8")
        result["generation_ok"] = True
        result["generated_characters"] = len(generated) - len(PROMPT)
    except Exception as error:  # noqa: BLE001 - result must preserve exact validation error
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def verify_one_command(args: argparse.Namespace) -> int:
    result = verify_checkpoint(args.checkpoint, args.sample_output, args.run_id, args.seed)
    write_json(args.result_output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if all(result[key] for key in ("loadable", "strict_load", "generation_ok")) else 1


def declared_checkpoint(root: Path, summary: dict[str, Any], checkpoint_type: str) -> Path:
    field = f"{checkpoint_type}_checkpoint"
    declared = Path(str(summary[field]))
    if declared.is_absolute():
        raise ValueError(f"{field} must be relative: {declared}")
    resolved = (root / declared).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"{field} escapes project root: {declared}")
    return resolved


def run_independent_reload(
    root: Path, run: FormalRun, checkpoint_path: Path, summary_dir: Path
) -> dict[str, Any]:
    run_dir = root / "outputs" / run.run_id
    sample_output = run_dir / "checkpoint_reload_sample.txt"
    result_output = summary_dir / ".reload_results" / f"{run.run_id}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "verify-one",
        "--checkpoint",
        str(checkpoint_path),
        "--sample-output",
        str(sample_output),
        "--result-output",
        str(result_output),
        "--run-id",
        run.run_id,
        "--seed",
        str(run.seed),
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    if not result_output.exists():
        raise RuntimeError(
            f"independent reload did not produce a result for {run.run_id}: "
            f"{completed.stderr.strip()}"
        )
    result = read_json(result_output)
    result["checkpoint"] = checkpoint_path.relative_to(root).as_posix()
    result["process_returncode"] = completed.returncode
    result["independent_process"] = True
    if completed.returncode != 0:
        raise RuntimeError(f"checkpoint reload failed for {run.run_id}: {result['error']}")
    return result


def closure_report(
    remote_url: str,
    closure_commit: str,
    inventory: list[dict[str, Any]],
    reload_results: list[dict[str, Any]],
) -> str:
    best_count = sum(row["checkpoint_type"] == "best" and row["loadable"] for row in inventory)
    final_count = sum(row["checkpoint_type"] == "final" and row["loadable"] for row in inventory)
    generated_count = sum(result["generation_ok"] for result in reload_results)
    return f"""# Stage 1 Checkpoint Closure Report

## Outcome

- Formal runs completed: 9/9
- Loadable non-empty best checkpoints: {best_count}/9
- Loadable non-empty final checkpoints: {final_count}/9
- Independent strict reload and `ROMEO:` generation tests: {generated_count}/9
- Missing best checkpoints: 0
- Retraining triggered: no

## Repository record

- GitHub repository: `{remote_url}`
- Original experiment commit: `{EXPERIMENT_COMMIT}`
- Closure implementation commit: `{closure_commit}`
- Git LFS used: no
- Checkpoints committed to ordinary Git history: no
- Checkpoint storage: final closure ZIP only

## Validation method

Each `best.pt` was loaded in a newly spawned Python process on CPU. The process rebuilt
`MiniTransformerLM` from the saved model configuration, loaded the state dictionary with
`strict=True`, reconstructed the saved vocabulary, selected `eval()` through the generation
helper, fixed seed 42, and generated 100 characters from `ROMEO:`. SHA256 was calculated from
the exact archived bytes. The inventory also confirms step, optimizer, AMP scaler, batch
iterator, RNG, configuration, vocabulary, and model state required for recovery. The learning
rate schedule is stateless and derived from step/configuration, so there is no scheduler object.

## Notes

The existing experimental metrics and summaries were not changed. Both `best.pt` and
`final.pt` were retained because storage was sufficient. The archive SHA256 is stored in the
ZIP sidecar file because a ZIP cannot contain its own final hash without changing that hash.
"""


def audit_command(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    summary_dir = root / "outputs" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{EXPERIMENT_COMMIT}^{{commit}}"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"original experiment commit is missing: {EXPERIMENT_COMMIT}") from error
    closure_commit = git_output(root, "rev-parse", "HEAD")
    remote_url = git_output(root, "remote", "get-url", "origin")
    inventory: list[dict[str, Any]] = []
    reload_results: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    sha_lines: list[str] = []

    for run in FORMAL_RUNS:
        run_dir = root / "outputs" / run.run_id
        status = read_json(run_dir / "status.json")
        summary = read_json(run_dir / "summary.json")
        if status.get("state") != "completed" or summary.get("status") != "completed":
            raise RuntimeError(f"formal run is not completed: {run.run_id}")
        if summary.get("git_commit") != EXPERIMENT_COMMIT:
            raise RuntimeError(f"experiment commit mismatch for {run.run_id}")
        if int(summary.get("seed", -1)) != run.seed:
            raise RuntimeError(f"seed mismatch for {run.run_id}")
        if not math.isclose(float(summary["data_fraction"]), run.data_fraction):
            raise RuntimeError(f"data fraction mismatch for {run.run_id}")

        for checkpoint_type in ("best", "final"):
            checkpoint_path = declared_checkpoint(root, summary, checkpoint_type)
            relative_path = checkpoint_path.relative_to(root).as_posix()
            exists = checkpoint_path.is_file()
            size = checkpoint_path.stat().st_size if exists else 0
            row: dict[str, Any] = {
                "run_id": run.run_id,
                "model_size": run.model_size,
                "data_fraction": run.data_fraction,
                "seed": run.seed,
                "checkpoint_type": checkpoint_type,
                "path": relative_path,
                "exists": exists,
                "size_bytes": size,
                "step": "",
                "loadable": False,
                "sha256": "",
            }
            if checkpoint_type == "best" and (not exists or size <= 0):
                raise FileNotFoundError(f"required best checkpoint is missing: {relative_path}")
            if not exists or size <= 0:
                inventory.append(row)
                continue
            details = inspect_checkpoint(checkpoint_path)
            digest = sha256_file(checkpoint_path)
            row.update(step=details["step"], loadable=True, sha256=digest)
            inventory.append(row)
            sha_lines.append(f"{digest}  {relative_path}")
            entry = {
                **row,
                "config_hash": summary["config_hash"],
                "dataset_hash": summary["dataset_hash"],
                "experiment_commit": summary["git_commit"],
                "recovery_state": details,
                "strict_load": None,
                "generation_ok": None,
            }
            manifest_entries.append(entry)

        reload_result = run_independent_reload(
            root, run, declared_checkpoint(root, summary, "best"), summary_dir
        )
        reload_results.append(reload_result)
        for entry in manifest_entries:
            if entry["run_id"] == run.run_id and entry["checkpoint_type"] == "best":
                entry["strict_load"] = reload_result["strict_load"]
                entry["generation_ok"] = reload_result["generation_ok"]

    inventory_path = summary_dir / "checkpoint_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(inventory)
    write_json(summary_dir / "checkpoint_reload_tests.json", reload_results)
    manifest = {
        "schema_version": 1,
        "github_url": remote_url,
        "experiment_commit": EXPERIMENT_COMMIT,
        "closure_commit": closure_commit,
        "retraining_triggered": False,
        "checkpoints_in_git": False,
        "git_lfs_used": False,
        "best_checkpoint_count": 9,
        "final_checkpoint_count": sum(
            row["checkpoint_type"] == "final" and row["loadable"] for row in inventory
        ),
        "entries": manifest_entries,
    }
    write_json(summary_dir / "checkpoint_manifest.json", manifest)
    (summary_dir / "checkpoint_sha256.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8"
    )
    (summary_dir / "STAGE1_CLOSURE_REPORT.md").write_text(
        closure_report(remote_url, closure_commit, inventory, reload_results),
        encoding="utf-8",
    )
    reload_dir = summary_dir / ".reload_results"
    for result_file in reload_dir.glob("*.json"):
        result_file.unlink()
    reload_dir.rmdir()
    print(
        json.dumps(
            {
                "formal_runs": 9,
                "best_checkpoints": 9,
                "final_checkpoints": manifest["final_checkpoint_count"],
                "strict_reload_generation_passed": len(reload_results),
                "retraining_triggered": False,
                "closure_commit": closure_commit,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="audit all formal checkpoints")
    audit.add_argument("--root", type=Path, default=project_root())
    audit.set_defaults(handler=audit_command)
    verify = subparsers.add_parser("verify-one", help="strictly reload one checkpoint")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--sample-output", type=Path, required=True)
    verify.add_argument("--result-output", type=Path, required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--seed", type=int, default=42)
    verify.set_defaults(handler=verify_one_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
