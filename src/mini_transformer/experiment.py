from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_training_config, load_yaml
from .train import run_training
from .utils import atomic_json, stable_hash, utc_now


def run_matrix(
    root: Path,
    matrix_path: Path,
    training_path: Path,
    data_path: Path,
    output_root: Path,
) -> int:
    matrix = load_yaml(matrix_path)
    training = load_training_config(training_path)
    runs: list[dict[str, object]] = []
    failed = False
    for model_size in matrix["model_sizes"]:
        for data_fraction in matrix["data_fractions"]:
            fraction_label = f"{int(float(data_fraction) * 100):03d}pct"
            run_id = f"{model_size}-{fraction_label}-seed{training.seed}"
            run_dir = output_root / run_id
            entry: dict[str, object] = {
                "run_id": run_id,
                "model_size": model_size,
                "data_fraction": float(data_fraction),
                "path": str(run_dir),
            }
            if not (run_dir / "status.json").exists():
                atomic_json(
                    run_dir / "status.json",
                    {"state": "pending", "created_at_utc": utc_now()},
                )
            try:
                summary = run_training(
                    root=root,
                    data_path=data_path,
                    output_dir=run_dir,
                    model_size=model_size,
                    data_fraction=float(data_fraction),
                    training=training,
                )
                entry["status"] = summary["status"]
                entry["config_hash"] = summary["config_hash"]
            except Exception as error:
                failed = True
                entry.update(
                    status="failed", error_type=type(error).__name__, error=str(error)
                )
                current_status = {}
                if (run_dir / "status.json").exists():
                    current_status = json.loads(
                        (run_dir / "status.json").read_text(encoding="utf-8")
                    )
                atomic_json(
                    run_dir / "status.json",
                    {
                        **current_status,
                        "state": "failed",
                        "failed_at_utc": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                if not matrix.get("continue_on_error", True):
                    runs.append(entry)
                    break
            runs.append(entry)
        if failed and not matrix.get("continue_on_error", True):
            break
    manifest = {
        "created_at_utc": utc_now(),
        "matrix_config_hash": stable_hash(matrix),
        "expected_runs": len(matrix["model_sizes"]) * len(matrix["data_fractions"]),
        "runs": runs,
    }
    atomic_json(output_root / "matrix_manifest.json", manifest)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequential 3x3 experiment matrix")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument("--training", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/tiny_shakespeare.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    exit_code = run_matrix(root, args.matrix, args.training, args.data, args.output_root)
    result = {
        "exit_code": exit_code,
        "manifest": str(args.output_root / "matrix_manifest.json"),
    }
    print(json.dumps(result))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
