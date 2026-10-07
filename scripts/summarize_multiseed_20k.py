#!/usr/bin/env python3
"""Aggregate the 27 formal runs, calculate seed statistics, and render figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_transformer.multiseed import (  # noqa: E402
    FORMAL_STEPS,
    expand_runs,
    load_experiment_config,
)

T_CRITICAL_DF2 = 4.303
SUMMARY_FIELDS = (
    "run_id", "model_size", "data_fraction", "seed", "physical_gpu_id",
    "parameter_count", "unique_train_windows", "tokens_processed", "best_step",
    "best_val_loss", "final_train_loss", "final_val_loss", "final_val_perplexity",
    "training_seconds", "tokens_per_second", "peak_gpu_memory_mb", "best_checkpoint",
    "final_checkpoint", "status", "git_commit", "config_hash", "dataset_hash",
)
STAT_METRICS = (
    "best_val_loss", "final_val_loss", "final_val_perplexity", "training_seconds",
    "tokens_per_second", "peak_gpu_memory_mb",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stats(values: Iterable[float]) -> tuple[float, float, float, float, float]:
    samples = [float(value) for value in values]
    if len(samples) != 3:
        raise ValueError(f"expected three seeds, got {len(samples)}")
    mean = statistics.mean(samples)
    std = statistics.stdev(samples)
    se = std / math.sqrt(3)
    margin = T_CRITICAL_DF2 * se
    return mean, std, se, mean - margin, mean + margin


def write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(config_path: Path, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    expected = expand_runs(load_experiment_config(config_path))
    summaries: list[dict[str, Any]] = []
    metrics: dict[str, list[dict[str, Any]]] = {}
    for run in expected:
        run_dir = output_root / run.run_id
        summary = read_json(run_dir / "summary.json")
        if summary.get("status") != "completed":
            raise RuntimeError(f"run is not completed: {run.run_id}")
        summaries.append(summary)
        metrics[run.run_id] = read_metrics(run_dir / "metrics.jsonl")
    return summaries, metrics


def condition_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["model_size"], float(row["data_fraction"]))].append(row)
    result = []
    for (model, fraction), rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "model_size": model,
            "data_fraction": fraction,
            "n_seeds": len(rows),
            "parameter_count": rows[0]["parameter_count"],
            "unique_train_windows": rows[0]["unique_train_windows"],
            "best_step_mean": statistics.mean(float(row["best_step"]) for row in rows),
            "best_step_min": min(int(row["best_step"]) for row in rows),
            "best_step_max": max(int(row["best_step"]) for row in rows),
        }
        for metric in STAT_METRICS:
            mean, std, se, low, high = stats(row[metric] for row in rows)
            output.update({
                f"{metric}_mean": mean,
                f"{metric}_sample_std": std,
                f"{metric}_se": se,
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        result.append(output)
    return result


def paired_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["model_size"], float(row["data_fraction"]), int(row["seed"])): row
        for row in summaries
    }
    specs: list[tuple[str, str, str, list[tuple[str, float]], list[tuple[str, float]]]] = []
    for fraction in (0.1, 0.3, 1.0):
        for left, right in (("tiny", "small"), ("small", "medium"), ("tiny", "medium")):
            specs.append(("model", f"{right}-{left}", f"data={fraction}", [(left, fraction)], [(right, fraction)]))
    for model in ("tiny", "small", "medium"):
        for left, right in ((0.1, 0.3), (0.3, 1.0), (0.1, 1.0)):
            specs.append(("data", f"{right:g}-{left:g}", f"model={model}", [(model, left)], [(model, right)]))
    result = []
    for comparison_type, comparison, fixed_condition, left, right in specs:
        differences = {}
        for seed in (42, 43, 44):
            left_row = lookup[(left[0][0], left[0][1], seed)]
            right_row = lookup[(right[0][0], right[0][1], seed)]
            differences[seed] = float(right_row["best_val_loss"]) - float(left_row["best_val_loss"])
        mean, std, se, low, high = stats(differences.values())
        result.append({
            "metric": "best_val_loss",
            "difference_direction": "right_minus_left_negative_is_improvement",
            "comparison_type": comparison_type,
            "comparison": comparison,
            "fixed_condition": fixed_condition,
            **{f"seed{seed}_difference": value for seed, value in differences.items()},
            "mean_difference": mean,
            "sample_std": std,
            "standard_error": se,
            "ci95_low": low,
            "ci95_high": high,
        })
    return result


def save_figure(summary_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(summary_dir / f"{name}.png", dpi=300)
    plt.savefig(summary_dir / f"{name}.pdf")
    plt.close()


def plot_all(
    summaries: list[dict[str, Any]],
    metrics: dict[str, list[dict[str, Any]]],
    conditions: list[dict[str, Any]],
    summary_dir: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for model_index, model in enumerate(("tiny", "small", "medium")):
        for fraction_index, fraction in enumerate((0.1, 0.3, 1.0)):
            validation_curves = []
            training_curves = []
            for seed in (42, 43, 44):
                run_id = f"{model}-{int(fraction * 100):03d}pct-seed{seed}"
                by_step = {int(row["step"]): row for row in metrics[run_id]}
                if not all(step in by_step for step in FORMAL_STEPS):
                    raise ValueError(f"missing aligned formal steps: {run_id}")
                validation_curves.append(
                    [float(by_step[step]["val_loss"]) for step in FORMAL_STEPS]
                )
                training_curves.append(
                    [float(by_step[step]["train_loss"]) for step in FORMAL_STEPS]
                )
            axis = axes[model_index, fraction_index]
            for samples, label, color, style in (
                (np.asarray(training_curves), "train", "tab:orange", "--"),
                (np.asarray(validation_curves), "validation", "tab:blue", "-"),
            ):
                mean = samples.mean(axis=0)
                std = samples.std(axis=0, ddof=1)
                axis.plot(FORMAL_STEPS, mean, style, label=label, color=color)
                axis.fill_between(
                    FORMAL_STEPS, mean - std, mean + std, color=color, alpha=0.15
                )
            axis.set_title(f"{model}, {int(fraction * 100)}% data")
            axis.grid(alpha=0.2)
            if model_index == 2:
                axis.set_xlabel("Optimizer step")
            if fraction_index == 0:
                axis.set_ylabel("Cross-entropy loss")
            if model_index == 0 and fraction_index == 0:
                axis.legend()
    figure.suptitle("Training and validation loss across seeds (mean +/- sample std)")
    save_figure(summary_dir, "training_curves_mean_std")

    markers = {0.1: "o", 0.3: "s", 1.0: "^"}
    for fraction in (0.1, 0.3, 1.0):
        selected = sorted((row for row in conditions if float(row["data_fraction"]) == fraction), key=lambda row: int(row["parameter_count"]))
        plt.errorbar([int(row["parameter_count"]) for row in selected], [row["best_val_loss_mean"] for row in selected], yerr=[row["best_val_loss_sample_std"] for row in selected], marker=markers[fraction], capsize=4, label=f"{int(fraction * 100)}% data")
    plt.xscale("log")
    plt.title("Model scale and best validation loss")
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel("Best validation loss (mean +/- sample std)")
    plt.legend()
    save_figure(summary_dir, "model_scale_mean_std")

    for model in ("tiny", "small", "medium"):
        selected = sorted((row for row in conditions if row["model_size"] == model), key=lambda row: int(row["unique_train_windows"]))
        plt.errorbar([int(row["unique_train_windows"]) for row in selected], [row["best_val_loss_mean"] for row in selected], yerr=[row["best_val_loss_sample_std"] for row in selected], marker="o", capsize=4, label=model)
    plt.xscale("log")
    plt.title("Data scale and best validation loss")
    plt.xlabel("Unique training windows (log scale)")
    plt.ylabel("Best validation loss (mean +/- sample std)")
    plt.legend()
    save_figure(summary_dir, "data_scale_mean_std")

    labels, distributions = [], []
    for model in ("tiny", "small", "medium"):
        for fraction in (0.1, 0.3, 1.0):
            labels.append(f"{model}\n{int(fraction * 100)}%")
            distributions.append([int(row["best_step"]) for row in summaries if row["model_size"] == model and float(row["data_fraction"]) == fraction])
    plt.boxplot(distributions, labels=labels)
    plt.title("Best evaluation step across seeds")
    plt.xlabel("Condition")
    plt.ylabel("Best step (evaluation resolution: 2,000)")
    save_figure(summary_dir, "best_step_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for model in ("tiny", "small", "medium"):
        selected = [row for row in conditions if row["model_size"] == model]
        for axis, metric, label in (
            (axes[0], "training_seconds", "Training seconds"),
            (axes[1], "peak_gpu_memory_mb", "Peak GPU memory (MiB)"),
        ):
            axis.errorbar(
                [row[f"{metric}_mean"] for row in selected],
                [row["best_val_loss_mean"] for row in selected],
                xerr=[row[f"{metric}_sample_std"] for row in selected],
                yerr=[row["best_val_loss_sample_std"] for row in selected],
                fmt="o",
                capsize=3,
                label=model,
            )
            axis.set_xlabel(f"{label} (mean +/- sample std)")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Best validation loss (mean +/- sample std)")
    axes[0].legend()
    figure.suptitle("Compute and memory tradeoffs across seeds")
    save_figure(summary_dir, "compute_tradeoff_mean_std")


def report(
    conditions: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    endpoint_runs = sum(int(row["best_step"]) == 20000 for row in summaries)
    endpoint_conditions = sum(
        int(row["best_step_min"]) == 20000 and int(row["best_step_max"]) == 20000
        for row in conditions
    )
    stable_model_improvements = sum(
        all(float(row[f"seed{seed}_difference"]) < 0 for seed in (42, 43, 44))
        for row in paired
        if row["comparison_type"] == "model"
    )
    stable_data_improvements = sum(
        all(float(row[f"seed{seed}_difference"]) < 0 for seed in (42, 43, 44))
        for row in paired
        if row["comparison_type"] == "data"
    )
    commits = ", ".join(sorted({str(row["git_commit"]) for row in summaries}))
    dataset_hashes = ", ".join(sorted({str(row["dataset_hash"]) for row in summaries}))
    lines = [
        "# 阶段一扩展实验：三随机种子与 20,000 steps",
        "",
        "## 实验设计",
        "",
        "本实验在不修改原阶段一输出的前提下，将训练扩展为 3 个 seed、3 种模型和 3 种数据规模，共 27 组。每 2,000 steps 完整评测一次；因此 best step 仅具有 2,000-step 的离散分辨率。每组处理 81,920,000 tokens。",
        "",
        "GPU 映射固定为 seed 42→物理 GPU 0、seed 43→GPU 1、seed 44→GPU 2；GPU 3 未使用。计时可能受共享服务器负载影响。",
        "",
        "## 条件级结果",
        "",
        "| model | data | best val loss (mean ± sample std) | 95% CI | best step range |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(f"| {row['model_size']} | {float(row['data_fraction']):.0%} | {row['best_val_loss_mean']:.6f} ± {row['best_val_loss_sample_std']:.6f} | [{row['best_val_loss_ci95_low']:.6f}, {row['best_val_loss_ci95_high']:.6f}] | {row['best_step_min']}–{row['best_step_max']} |")
    lines += [
        "",
        "95% CI 使用自由度 2 的 Student t 临界值 4.303。n=3 的区间较宽，不将其解释为高功效显著性检验。",
        "",
        "## 训练进程与收敛",
        "",
        f"27 个 run 中有 {endpoint_runs} 个的最佳离散评测点位于 step 20,000；"
        f"9 个条件中有 {endpoint_conditions} 个在三个 seed 上均以 20,000 为最佳评测点。"
        "训练曲线按共同的 2,000–20,000 steps 对齐并以跨 seed 均值±sample std 绘制。"
        "若多数条件仍在终点最佳，只能认为尚未观察到明确平台，并建议后续单独 pilot 到 40,000 steps。",
        "",
        "## 模型与数据规模",
        "",
        "`paired_differences.csv` 使用相同 seed 配对，差值定义为右侧条件减左侧条件，"
        "负值代表 best validation loss 改善。",
        "",
        f"模型规模的 9 类配对中，有 {stable_model_improvements} 类在三个 seed 上方向一致且为改善；"
        f"数据规模的 9 类配对中，有 {stable_data_improvements} 类满足同一标准。"
        "固定 token 预算下，数据比例变化也改变重复采样程度，因此不能把非单调结果"
        "直接解释为通用 Scaling Law。",
        "",
        "## 27 组完整结果",
        "",
        "| run | parameters | best step | best val loss | final val loss | perplexity | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run_id']} | {int(row['parameter_count'])} | "
            f"{int(row['best_step'])} | {float(row['best_val_loss']):.6f} | "
            f"{float(row['final_val_loss']):.6f} | "
            f"{float(row['final_val_perplexity']):.6f} | "
            f"{float(row['training_seconds']):.2f} |"
        )
    lines += [
        "",
        "## 图表",
        "",
        "- `training_curves_mean_std.{png,pdf}`：跨 seed 验证曲线均值与 ±1 sample std。",
        "- `model_scale_mean_std.{png,pdf}`：参数规模比较。",
        "- `data_scale_mean_std.{png,pdf}`：数据规模比较。",
        "- `best_step_comparison.{png,pdf}`：best step 分布。",
        "- `compute_tradeoff_mean_std.{png,pdf}`：训练时间与效果权衡。",
        "",
        "## 可复现性与限制",
        "",
        f"Git commit：`{commits}`。数据 SHA256：`{dataset_hashes}`。",
        "",
        "每组保存 resolved config、配置/数据/Git 哈希、Python/NumPy/PyTorch/CUDA seed 环境、物理/逻辑 GPU 映射、AMP 与后端标志。共享服务器会给吞吐和耗时引入噪声；字符级 Tiny Shakespeare、三个 seed 与固定训练预算限制了结论外推。完整 checkpoint 验收结果见 `validation_report.json`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_multiseed_20k.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs_multiseed_20k"))
    args = parser.parse_args()
    summaries, metrics = load_runs(args.config, args.output_root)
    conditions = condition_rows(summaries)
    paired = paired_rows(summaries)
    summary_dir = args.output_root / "summary"
    write_csv(summary_dir / "experiment_summary_27runs.csv", SUMMARY_FIELDS, [{key: row.get(key, "") for key in SUMMARY_FIELDS} for row in summaries])
    condition_fields = list(conditions[0])
    write_csv(summary_dir / "condition_summary.csv", condition_fields, conditions)
    write_csv(summary_dir / "paired_differences.csv", list(paired[0]), paired)
    plot_all(summaries, metrics, conditions, summary_dir)
    (summary_dir / "STAGE1_MULTISEED_20K_REPORT.md").write_text(report(conditions, paired, summaries), encoding="utf-8", newline="\n")
    (summary_dir / "failed_runs.json").write_text("[]\n", encoding="utf-8", newline="\n")
    print(json.dumps({"runs": len(summaries), "conditions": len(conditions), "paired_comparisons": len(paired), "summary_dir": str(summary_dir)}, indent=2))


if __name__ == "__main__":
    main()
