#!/usr/bin/env python3
"""Independently validate all formal outputs and all 54 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_transformer.multiseed import (  # noqa: E402
    FORMAL_STEPS,
    expand_runs,
    load_experiment_config,
)
from mini_transformer.utils import atomic_json, sha256_file, stable_hash, utc_now  # noqa: E402

TOKENS_PER_RUN = 81_920_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_one(checkpoint: Path, generate: bool) -> dict[str, Any]:
    import torch

    from mini_transformer.config import ModelConfig
    from mini_transformer.data import CharacterVocabulary
    from mini_transformer.generate import generate_text
    from mini_transformer.model import MiniTransformerLM

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
        "val_batch_iterator_state",
    }
    missing = sorted(required - saved.keys())
    if missing:
        raise ValueError(f"checkpoint recovery state is incomplete: {missing}")
    model = MiniTransformerLM(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model_state"], strict=True)
    result = {"strict_load": True, "step": int(saved["step"]), "generation_ok": None}
    if generate:
        torch.manual_seed(42)
        vocabulary = CharacterVocabulary(saved["vocabulary"][1:])
        text = generate_text(model, vocabulary, "ROMEO:", torch.device("cpu"), max_new_tokens=20)
        result["generation_ok"] = text.startswith("ROMEO:") and len(text) > len("ROMEO:")
        if not result["generation_ok"]:
            raise ValueError("ROMEO: generation test failed")
    return result


def verify_one_subprocess(checkpoint: Path, generate: bool) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "verify-one", str(checkpoint)]
    if generate:
        command.append("--generate")
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"independent checkpoint load failed: {checkpoint}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def current_original_manifest(root: Path) -> list[dict[str, Any]]:
    old_root = root / "outputs"
    if not old_root.exists():
        raise FileNotFoundError("original Stage 1 outputs directory is missing")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in old_root.rglob("*") if item.is_file())
    ]


def finite_fields(row: dict[str, Any]) -> None:
    for key in (
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "final_val_perplexity",
        "training_seconds",
        "tokens_per_second",
        "peak_gpu_memory_mb",
    ):
        if not math.isfinite(float(row[key])):
            raise ValueError(f"non-finite {key} in {row['run_id']}")


def validate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    expected = expand_runs(config)
    output_root = args.output_root.resolve()
    baseline = read_json(output_root / "original_stage1_manifest.before.json")
    original_unchanged = baseline["files"] == current_original_manifest(PROJECT_ROOT)
    if not original_unchanged:
        raise RuntimeError("original Stage 1 outputs changed after the baseline snapshot")
    inventory: list[dict[str, Any]] = []
    sha_lines: list[str] = []
    dataset_hashes: set[str] = set()
    commits: set[str] = set()
    scheduler = read_json(output_root / "scheduler_manifest.json")
    workers = scheduler.get("workers", [])
    if scheduler.get("mode") != "formal" or len(workers) != 3:
        raise ValueError("formal scheduler manifest must contain exactly three workers")
    worker_mapping = {int(worker["seed"]): int(worker["physical_gpu_id"]) for worker in workers}
    if worker_mapping != config["physical_gpu_by_seed"]:
        raise ValueError("scheduler seed/GPU mapping is invalid")
    if any(int(worker.get("returncode", -1)) != 0 for worker in workers):
        raise ValueError("one or more scheduler workers failed")
    for run in expected:
        run_dir = output_root / run.run_id
        status = read_json(run_dir / "status.json")
        summary = read_json(run_dir / "summary.json")
        resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        environment = read_json(run_dir / "environment.json")
        dataset = read_json(run_dir / "dataset_manifest.json")
        metrics = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if status.get("state") != "completed" or summary.get("status") != "completed":
            raise RuntimeError(f"run not completed: {run.run_id}")
        if (
            int(status.get("step", -1)) != 20000
            or int(summary["tokens_processed"]) != TOKENS_PER_RUN
        ):
            raise ValueError(f"step/token budget mismatch: {run.run_id}")
        if (summary["model_size"], float(summary["data_fraction"]), int(summary["seed"])) != (
            run.model_size,
            run.data_fraction,
            run.seed,
        ):
            raise ValueError(f"summary combination mismatch: {run.run_id}")
        training = resolved["training"]
        if (
            int(training["max_steps"]),
            int(training["eval_interval"]),
            int(training["eval_batches"]),
            int(training["checkpoint_interval"]),
        ) != (20000, 2000, 50, 2000):
            raise ValueError(f"resolved formal config mismatch: {run.run_id}")
        if (
            int(training["seed"]) != run.seed
            or int(training["physical_gpu_id"]) != run.physical_gpu_id
        ):
            raise ValueError(f"resolved seed/GPU mismatch: {run.run_id}")
        if (
            environment.get("physical_gpu_id") != run.physical_gpu_id
            or environment.get("logical_gpu_id") != 0
        ):
            raise ValueError(f"environment GPU mapping mismatch: {run.run_id}")
        if not environment.get("cuda_available") or environment.get("cuda_visible_devices") != str(
            run.physical_gpu_id
        ):
            raise ValueError(f"CUDA visibility mismatch: {run.run_id}")
        for backend_key in (
            "cudnn_deterministic",
            "cudnn_benchmark",
            "deterministic_algorithms",
        ):
            if backend_key not in environment:
                raise ValueError(f"missing backend flag {backend_key}: {run.run_id}")
        if summary["config_hash"] != stable_hash(resolved):
            raise ValueError(f"config hash mismatch: {run.run_id}")
        if summary["dataset_hash"] != dataset["dataset_sha256"]:
            raise ValueError(f"dataset hash mismatch: {run.run_id}")
        if environment["git_commit"] != summary["git_commit"]:
            raise ValueError(f"environment Git commit mismatch: {run.run_id}")
        official = [row for row in metrics if int(row["step"]) in FORMAL_STEPS]
        if [int(row["step"]) for row in official] != list(FORMAL_STEPS) or len(official) != 10:
            raise ValueError(f"formal evaluation points mismatch: {run.run_id}")
        for row in metrics:
            if (
                row["run_id"] != run.run_id
                or int(row["seed"]) != run.seed
                or int(row["physical_gpu_id"]) != run.physical_gpu_id
            ):
                raise ValueError(f"metric metadata mismatch: {run.run_id}")
            for key in (
                "train_loss",
                "val_loss",
                "val_perplexity",
                "learning_rate",
                "elapsed_seconds",
            ):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"non-finite metric {key}: {run.run_id}")
        if int(summary["best_step"]) not in FORMAL_STEPS:
            raise ValueError(f"best_step is not an official evaluation point: {run.run_id}")
        finite_fields(summary)
        dataset_hashes.add(str(summary["dataset_hash"]))
        commits.add(str(summary["git_commit"]))
        for checkpoint_type in ("best", "final"):
            checkpoint = run_dir / "checkpoints" / f"{checkpoint_type}.pt"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            digest = sha256_file(checkpoint)
            result = verify_one_subprocess(checkpoint, checkpoint_type == "best")
            expected_checkpoint_step = (
                int(summary["best_step"]) if checkpoint_type == "best" else 20000
            )
            if int(result["step"]) != expected_checkpoint_step:
                raise ValueError(f"checkpoint step mismatch: {run.run_id}/{checkpoint_type}")
            relative = checkpoint.relative_to(PROJECT_ROOT).as_posix()
            inventory.append(
                {
                    "run_id": run.run_id,
                    "checkpoint_type": checkpoint_type,
                    "path": relative,
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": digest,
                    "step": result["step"],
                    "strict_load": result["strict_load"],
                    "generation_ok": result["generation_ok"],
                }
            )
            sha_lines.append(f"{digest}  {relative}")

    summary_dir = output_root / "summary"
    with (summary_dir / "experiment_summary_27runs.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        full_rows = list(csv.DictReader(handle))
    with (summary_dir / "condition_summary.csv").open(encoding="utf-8", newline="") as handle:
        condition_rows = list(csv.DictReader(handle))
    with (summary_dir / "paired_differences.csv").open(encoding="utf-8", newline="") as handle:
        paired_rows = list(csv.DictReader(handle))
    if len(full_rows) != 27 or len(condition_rows) != 9 or len(paired_rows) != 18:
        raise ValueError("summary table row counts are invalid")
    if {row["run_id"] for row in full_rows} != {run.run_id for run in expected}:
        raise ValueError("27-run summary does not cover the exact formal matrix")
    for row in condition_rows:
        if int(row["n_seeds"]) != 3:
            raise ValueError("condition summary must aggregate exactly three seeds")
        for key in (
            "best_val_loss_mean",
            "best_val_loss_sample_std",
            "best_val_loss_ci95_low",
            "best_val_loss_ci95_high",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"non-finite condition statistic: {key}")
    for row in paired_rows:
        if row["difference_direction"] != "right_minus_left_negative_is_improvement":
            raise ValueError("paired difference direction is missing or invalid")
        for key in ("mean_difference", "sample_std", "ci95_low", "ci95_high"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"non-finite paired statistic: {key}")
    failed_runs = read_json(summary_dir / "failed_runs.json")
    if failed_runs != []:
        raise ValueError("failed_runs.json is not empty")
    figure_names = (
        "training_curves_mean_std",
        "model_scale_mean_std",
        "data_scale_mean_std",
        "best_step_comparison",
        "compute_tradeoff_mean_std",
    )
    missing_figures = [
        f"{name}.{suffix}"
        for name in figure_names
        for suffix in ("png", "pdf")
        if not (summary_dir / f"{name}.{suffix}").is_file()
    ]
    if missing_figures:
        raise FileNotFoundError(f"missing figures: {missing_figures}")
    inventory_path = summary_dir / "checkpoint_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    (summary_dir / "checkpoint_sha256.txt").write_text(
        "\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n"
    )
    report = {
        "validated_at_utc": utc_now(),
        "passed": True,
        "expected_runs": 27,
        "completed": 27,
        "failed": 0,
        "official_evaluations_per_run": 10,
        "tokens_per_run": TOKENS_PER_RUN,
        "checkpoint_count": len(inventory),
        "strict_checkpoint_loads": sum(bool(row["strict_load"]) for row in inventory),
        "best_checkpoint_generation_tests": sum(
            row["checkpoint_type"] == "best" and bool(row["generation_ok"]) for row in inventory
        ),
        "dataset_hashes": sorted(dataset_hashes),
        "git_commits": sorted(commits),
        "original_stage1_unchanged": original_unchanged,
        "summary_rows": len(full_rows),
        "condition_rows": len(condition_rows),
        "paired_difference_rows": len(paired_rows),
        "physical_gpu_by_seed": config["physical_gpu_by_seed"],
    }
    atomic_json(summary_dir / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    verify = subparsers.add_parser("verify-one")
    verify.add_argument("checkpoint", type=Path)
    verify.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment_multiseed_20k.yaml")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs_multiseed_20k"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "verify-one":
        print(json.dumps(verify_one(args.checkpoint, args.generate)))
        return
    raise SystemExit(validate(args))


if __name__ == "__main__":
    main()
