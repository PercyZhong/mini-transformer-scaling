from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .config import TrainingConfig, load_model_preset, load_training_config
from .data import InfiniteBatchIterator, PreparedData, prepare_data
from .evaluate import evaluate_loss
from .generate import generate_text
from .model import MiniTransformerLM
from .utils import (
    atomic_json,
    collect_environment,
    resolve_device,
    seed_everything,
    stable_hash,
    utc_now,
    write_yaml,
)


def learning_rate(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    span = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / span))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


def create_grad_scaler(enabled: bool) -> Any:
    """Create a CUDA scaler using the current API with a PyTorch 2.2 fallback."""
    scaler_type = getattr(torch.amp, "GradScaler", None)
    if scaler_type is not None:
        return scaler_type("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    path: Path,
    model: MiniTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    train_batches: InfiniteBatchIterator,
    step: int,
    best_val_loss: float,
    data_fraction: float,
    vocabulary: list[str],
    val_batches: InfiniteBatchIterator | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "best_val_loss": best_val_loss,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "batch_iterator_state": train_batches.state_dict(),
            "val_batch_iterator_state": (
                val_batches.state_dict() if val_batches is not None else None
            ),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_fraction": data_fraction,
            "vocabulary": vocabulary,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: MiniTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    train_batches: InfiniteBatchIterator,
    device: torch.device,
    val_batches: InfiniteBatchIterator | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))
    train_batches.load_state_dict(checkpoint["batch_iterator_state"])
    if val_batches is not None and checkpoint.get("val_batch_iterator_state") is not None:
        val_batches.load_state_dict(checkpoint["val_batch_iterator_state"])
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return int(checkpoint["step"]), float(checkpoint["best_val_loss"])


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")


def _prepare_metrics_for_resume(
    path: Path, completed_step: int
) -> tuple[float, float, float]:
    if not path.exists():
        return 0.0, float("nan"), float("nan")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    retained = [row for row in rows if int(row["step"]) <= completed_step]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not retained:
        return 0.0, float("nan"), float("nan")
    last = retained[-1]
    return (
        max(float(row.get("elapsed_seconds", 0.0)) for row in retained),
        float(last["train_loss"]),
        float(last["val_loss"]),
    )


def run_training(
    *,
    root: Path,
    data_path: Path,
    output_dir: Path,
    model_size: str,
    data_fraction: float,
    training: TrainingConfig,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    status_path = output_dir / "status.json"
    metrics_path = output_dir / "metrics.jsonl"
    seed_everything(training.seed)
    device = resolve_device(training.device)
    prepared: PreparedData = prepare_data(
        data_path, training.context_length, data_fraction, training.seed
    )
    model_config = load_model_preset(
        root / "configs/models.yaml",
        model_size,
        len(prepared.vocabulary),
        training.context_length,
    )
    model_config = type(model_config)(**{**model_config.to_dict(), "dropout": training.dropout})
    resolved = {
        "model_size": model_size,
        "data_fraction": data_fraction,
        "model": model_config.to_dict(),
        "training": training.to_dict(),
    }
    config_hash = stable_hash(resolved)
    existing_status = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    )
    previous_hash = existing_status.get("config_hash")
    if previous_hash is not None and previous_hash != config_hash:
        raise RuntimeError("run directory contains artifacts for a different config hash")
    if not resume and (status_path.exists() or metrics_path.exists()):
        raise FileExistsError("refusing a fresh run in a directory containing run artifacts")
    if existing_status.get("state") == "completed":
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    write_yaml(output_dir / "config.resolved.yaml", resolved)
    atomic_json(
        output_dir / "environment.json",
        collect_environment(root, training.physical_gpu_id),
    )
    dataset_manifest = dict(prepared.manifest)
    download_manifest_path = root / "data/processed/dataset_manifest.json"
    if download_manifest_path.exists():
        dataset_manifest["download_audit"] = json.loads(
            download_manifest_path.read_text(encoding="utf-8")
        )
    atomic_json(output_dir / "dataset_manifest.json", dataset_manifest)
    atomic_json(
        status_path,
        {"state": "running", "started_at_utc": utc_now(), "config_hash": config_hash},
    )
    model = MiniTransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    amp_enabled = device.type == "cuda" and training.mixed_precision
    scaler = create_grad_scaler(amp_enabled)
    train_batches = InfiniteBatchIterator(
        prepared.datasets["train"], training.batch_size, training.seed
    )
    val_batches = InfiniteBatchIterator(
        prepared.datasets["validation"], training.batch_size, training.seed + 1
    )
    start_step, best_val_loss, best_step = 0, float("inf"), 0
    latest_path = checkpoint_dir / "latest.pt"
    if resume and latest_path.exists():
        start_step, best_val_loss = load_checkpoint(
            latest_path, model, optimizer, scaler, train_batches, device, val_batches
        )
        best_step = int(existing_status.get("best_step", 0))
    prior_elapsed, final_train_loss, final_val_loss = _prepare_metrics_for_resume(
        metrics_path, start_step
    )
    start_time = time.perf_counter()
    tokens_processed = (
        start_step
        * training.batch_size
        * training.context_length
        * training.gradient_accumulation_steps
    )
    current_step = start_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        for step in range(start_step + 1, training.max_steps + 1):
            current_step = step
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(training.gradient_accumulation_steps):
                inputs, targets = next(train_batches)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with context:
                    _, loss = model(inputs.to(device), targets.to(device))
                    assert loss is not None
                    scaled_loss = loss / training.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(loss.item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
            rate = learning_rate(step - 1, training)
            for group in optimizer.param_groups:
                group["lr"] = rate
            scaler.step(optimizer)
            scaler.update()
            final_train_loss = accumulated_loss / training.gradient_accumulation_steps
            tokens_processed += (
                training.batch_size
                * training.context_length
                * training.gradient_accumulation_steps
            )
            evaluate_now = (
                (training.evaluate_at_step_one and step == 1)
                or step % training.eval_interval == 0
                or step == training.max_steps
            )
            if evaluate_now:
                final_val_loss = evaluate_loss(model, val_batches, device, training.eval_batches)
                metric = {
                    "run_id": output_dir.name,
                    "model_size": model_size,
                    "data_fraction": data_fraction,
                    "seed": training.seed,
                    "step": step,
                    "train_loss": final_train_loss,
                    "val_loss": final_val_loss,
                    "val_perplexity": math.exp(final_val_loss),
                    "learning_rate": rate,
                    "tokens_processed": tokens_processed,
                    "elapsed_seconds": prior_elapsed + time.perf_counter() - start_time,
                    "physical_gpu_id": training.physical_gpu_id,
                }
                _append_metric(metrics_path, metric)
                if final_val_loss < best_val_loss:
                    best_val_loss, best_step = final_val_loss, step
                    save_checkpoint(
                        checkpoint_dir / "best.pt", model, optimizer, scaler, train_batches,
                        step, best_val_loss, data_fraction, prepared.vocabulary.itos,
                        val_batches,
                    )
            if step % training.checkpoint_interval == 0 or step == training.max_steps:
                save_checkpoint(
                    latest_path, model, optimizer, scaler, train_batches, step,
                    best_val_loss, data_fraction, prepared.vocabulary.itos,
                    val_batches,
                )
                atomic_json(
                    status_path,
                    {
                        "state": "running",
                        "step": step,
                        "best_step": best_step,
                        "config_hash": config_hash,
                    },
                )
        save_checkpoint(
            checkpoint_dir / "final.pt", model, optimizer, scaler, train_batches,
            training.max_steps, best_val_loss, data_fraction, prepared.vocabulary.itos,
            val_batches,
        )
        elapsed = prior_elapsed + time.perf_counter() - start_time
        sample = generate_text(
            model, prepared.vocabulary, training.fixed_prompt, device, max_new_tokens=100
        )
        (output_dir / "samples.txt").write_text(sample + "\n", encoding="utf-8", newline="\n")
        environment = collect_environment(root, training.physical_gpu_id)
        summary = {
            "run_id": output_dir.name,
            "model_size": model_size,
            "data_fraction": data_fraction,
            "seed": training.seed,
            "physical_gpu_id": training.physical_gpu_id,
            "parameter_count": model.count_parameters(),
            "unique_train_windows": len(prepared.datasets["train"]),
            "tokens_processed": tokens_processed,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_perplexity": math.exp(final_val_loss),
            "training_seconds": elapsed,
            "tokens_per_second": tokens_processed / max(elapsed, 1e-9),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            ),
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "final_checkpoint": str(checkpoint_dir / "final.pt"),
            "status": "completed",
            "git_commit": environment["git_commit"],
            "config_hash": config_hash,
            "dataset_hash": prepared.manifest["dataset_sha256"],
        }
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(
            status_path,
            {
                "state": "completed",
                "completed_at_utc": utc_now(),
                "step": training.max_steps,
                "best_step": best_step,
                "config_hash": config_hash,
            },
        )
        return summary
    except Exception as error:
        atomic_json(
            status_path,
            {
                "state": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "step": current_step,
                "best_step": best_step,
                "config_hash": config_hash,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the mini Transformer")
    parser.add_argument("--model-size", choices=("tiny", "small", "medium"), default="tiny")
    parser.add_argument("--data-fraction", type=float, choices=(0.1, 0.3, 1.0), default=0.1)
    parser.add_argument("--data", type=Path, default=Path("data/raw/tiny_shakespeare.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--device")
    parser.add_argument("--physical-gpu-id", type=int)
    parser.add_argument("--no-step-one-eval", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    overrides = {
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "device": args.device,
        "physical_gpu_id": args.physical_gpu_id,
        "evaluate_at_step_one": False if args.no_step_one_eval else None,
        "mixed_precision": False if args.no_mixed_precision else None,
    }
    training = load_training_config(args.training_config, overrides)
    summary = run_training(
        root=root,
        data_path=args.data,
        output_dir=args.output,
        model_size=args.model_size,
        data_fraction=args.data_fraction,
        training=training,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
