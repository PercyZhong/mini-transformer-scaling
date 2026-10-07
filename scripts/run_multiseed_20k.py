#!/usr/bin/env python3
"""Dry-run, smoke-test, launch, and safely resume the 3-GPU experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_transformer.config import load_training_config  # noqa: E402
from mini_transformer.multiseed import expand_runs, load_experiment_config  # noqa: E402
from mini_transformer.utils import atomic_json, sha256_file, stable_hash, utc_now  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_loadable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    command = [
        sys.executable,
        "-c",
        "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "assert isinstance(c['model_state'],dict) and c['step']>0",
        str(path),
    ]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


def completed_run_is_valid(run_dir: Path, expected_steps: int) -> bool:
    try:
        status = _read_json(run_dir / "status.json")
        summary = _read_json(run_dir / "summary.json")
        return (
            status.get("state") == "completed"
            and summary.get("status") == "completed"
            and int(status.get("step", -1)) == expected_steps
            and int(summary.get("tokens_processed", -1))
            == expected_steps * 32 * 128
            and _checkpoint_loadable(run_dir / "checkpoints/best.pt")
            and _checkpoint_loadable(run_dir / "checkpoints/final.pt")
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def original_stage1_manifest(root: Path) -> dict[str, Any]:
    old_root = root / "outputs"
    if not old_root.is_dir():
        raise FileNotFoundError("original Stage 1 outputs are required for immutability auditing")
    entries = []
    if old_root.exists():
        for path in sorted(item for item in old_root.rglob("*") if item.is_file()):
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"created_at_utc": utc_now(), "files": entries}


def active_compute_gpus() -> set[int]:
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    uuid_to_index = {
        uuid.strip(): int(index.strip())
        for line in gpu_query.stdout.splitlines()
        for index, uuid in [line.split(",", 1)]
    }
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process_query.returncode != 0 and "No running processes" not in process_query.stderr:
        raise RuntimeError(process_query.stderr.strip() or "cannot inspect GPU processes")
    return {
        uuid_to_index[line.split(",", 1)[0].strip()]
        for line in process_query.stdout.splitlines()
        if line.strip() and line.split(",", 1)[0].strip() in uuid_to_index
    }


def worker(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    all_runs = expand_runs(config)
    selected = [run for run in all_runs if run.seed == args.worker_seed]
    if not selected:
        raise ValueError(f"unknown worker seed: {args.worker_seed}")
    expected_gpu = config["physical_gpu_by_seed"][args.worker_seed]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected_gpu):
        raise RuntimeError("worker CUDA_VISIBLE_DEVICES does not match the configured physical GPU")

    from mini_transformer.train import run_training

    output_root = args.output_root
    failures: list[dict[str, str]] = []
    completed: list[str] = []
    run_records: list[dict[str, Any]] = []
    for run in selected:
        run_dir = output_root / run.run_id
        if args.resume and (run_dir / "status.json").exists():
            status = _read_json(run_dir / "status.json")
            if status.get("state") == "completed":
                if completed_run_is_valid(run_dir, config["max_steps"]):
                    message = {"run_id": run.run_id, "action": "skip_verified_completed"}
                    print(json.dumps(message), flush=True)
                    completed.append(run.run_id)
                    run_records.append(
                        {**message, "status": "completed", "finished_at_utc": utc_now()}
                    )
                else:
                    error = "completed run failed summary/checkpoint integrity validation"
                    print(json.dumps({"run_id": run.run_id, "action": "refuse_invalid_completed", "error": error}), flush=True)
                    failures.append({"run_id": run.run_id, "error_type": "IntegrityError", "error": error})
                    run_records.append(
                        {
                            "run_id": run.run_id,
                            "action": "refuse_invalid_completed",
                            "status": "failed",
                            "error": error,
                            "finished_at_utc": utc_now(),
                        }
                    )
                continue
        run_started = utc_now()
        print(
            json.dumps({"run_id": run.run_id, "action": "train", "started": run_started}),
            flush=True,
        )
        training = load_training_config(
            args.training,
            {
                "seed": run.seed,
                "max_steps": config["max_steps"],
                "eval_interval": config["eval_interval"],
                "eval_batches": config["eval_batches"],
                "checkpoint_interval": config["checkpoint_interval"],
                "physical_gpu_id": run.physical_gpu_id,
                "evaluate_at_step_one": False,
                "device": "cuda",
            },
        )
        try:
            run_training(
                root=PROJECT_ROOT,
                data_path=args.data,
                output_dir=run_dir,
                model_size=run.model_size,
                data_fraction=run.data_fraction,
                training=training,
                resume=args.resume,
            )
            completed.append(run.run_id)
            run_records.append(
                {
                    "run_id": run.run_id,
                    "action": "train",
                    "status": "completed",
                    "started_at_utc": run_started,
                    "finished_at_utc": utc_now(),
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve per-run diagnostics
            traceback.print_exc()
            failures.append(
                {"run_id": run.run_id, "error_type": type(error).__name__, "error": str(error)}
            )
            run_records.append(
                {
                    "run_id": run.run_id,
                    "action": "train",
                    "status": "failed",
                    "started_at_utc": run_started,
                    "finished_at_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    atomic_json(
        output_root / "scheduler_logs" / f"seed{args.worker_seed}_worker_result.json",
        {
            "completed": completed,
            "failures": failures,
            "runs": run_records,
            "finished_at_utc": utc_now(),
        },
    )
    return 1 if failures else 0


def smoke_worker_args(args: argparse.Namespace, seed: int, output_root: Path) -> list[str]:
    gpu = load_experiment_config(args.config)["physical_gpu_by_seed"][seed]
    run_id = f"smoke-tiny-010pct-seed{seed}-gpu{gpu}"
    return [
        sys.executable,
        "-m",
        "mini_transformer.train",
        "--model-size", "tiny",
        "--data-fraction", "0.1",
        "--data", str(args.data),
        "--output", str(output_root / run_id),
        "--device", "cuda",
        "--physical-gpu-id", str(gpu),
        "--seed", str(seed),
        "--context-length", "32",
        "--batch-size", "4",
        "--max-steps", "2",
        "--eval-interval", "1",
        "--eval-batches", "1",
        "--checkpoint-interval", "1",
        "--no-step-one-eval",
        "--no-resume",
    ]


def launch(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    runs = expand_runs(config)
    if args.dry_run:
        print(json.dumps({"count": len(runs), "runs": [run.__dict__ | {"run_id": run.run_id} for run in runs]}, indent=2))
        return 0
    used = active_compute_gpus()
    conflicts = sorted({0, 1, 2} & used)
    if conflicts:
        raise RuntimeError(f"refusing to start because required physical GPUs are busy: {conflicts}")

    output_root = args.output_root
    expected_output = Path(
        "outputs_multiseed_20k_smoke" if args.smoke else config["output_root"]
    )
    if output_root != expected_output:
        raise ValueError(f"output root must be {expected_output} for this mode")
    if args.smoke:
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        existing_runs = [output_root / run.run_id for run in runs if (output_root / run.run_id).exists()]
        if existing_runs and not args.resume:
            raise FileExistsError("formal run directories exist; inspect them and use --resume")
        output_root.mkdir(parents=True, exist_ok=True)
        baseline_path = output_root / "original_stage1_manifest.before.json"
        if not baseline_path.exists():
            atomic_json(baseline_path, original_stage1_manifest(PROJECT_ROOT))

    logs = output_root / "scheduler_logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, int, subprocess.Popen[str], Any]] = []
    records: list[dict[str, Any]] = []
    for seed, gpu in config["physical_gpu_by_seed"].items():
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        command = (
            smoke_worker_args(args, seed, output_root)
            if args.smoke
            else [
                sys.executable, str(Path(__file__).resolve()), "--worker-seed", str(seed),
                "--config", str(args.config), "--training", str(args.training),
                "--data", str(args.data), "--output-root", str(output_root),
                *(["--resume"] if args.resume else []),
            ]
        )
        log_path = logs / f"seed{seed}_gpu{gpu}.log"
        handle = log_path.open("a" if args.resume else "w", encoding="utf-8", newline="\n")
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((seed, gpu, process, handle))
        records.append({"seed": seed, "physical_gpu_id": gpu, "pid": process.pid, "started_at_utc": utc_now(), "run_ids": [run.run_id for run in runs if run.seed == seed], "log": log_path.as_posix()})
    failed = False
    for seed, _gpu, process, handle in processes:
        returncode = process.wait()
        handle.close()
        failed |= returncode != 0
        record = next(item for item in records if item["seed"] == seed)
        record.update(returncode=returncode, finished_at_utc=utc_now())
        worker_result = logs / f"seed{seed}_worker_result.json"
        if worker_result.exists():
            record["runs"] = _read_json(worker_result).get("runs", [])
    manifest = {"created_at_utc": utc_now(), "mode": "smoke" if args.smoke else "formal", "config_hash": stable_hash(config), "workers": records}
    atomic_json(output_root / "scheduler_manifest.json", manifest)
    print(json.dumps({"workers": records, "failed": failed}, indent=2))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_multiseed_20k.yaml"))
    parser.add_argument("--training", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/tiny_shakespeare.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs_multiseed_20k"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke and args.output_root == Path("outputs_multiseed_20k"):
        args.output_root = Path("outputs_multiseed_20k_smoke")
    exit_code = worker(args) if args.worker_seed is not None else launch(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
