import math
from pathlib import Path

import pytest

from mini_transformer.multiseed import (
    EXPECTED_GPU_MAP,
    FORMAL_STEPS,
    expand_runs,
    load_experiment_config,
)
from scripts.package_multiseed_20k import safe_names
from scripts.summarize_multiseed_20k import condition_rows, paired_rows, stats


def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/experiment_multiseed_20k.yaml"


def test_formal_matrix_has_27_unique_runs_and_fixed_gpu_mapping() -> None:
    config = load_experiment_config(config_path())
    runs = expand_runs(config)
    assert len(runs) == 27
    assert len({run.run_id for run in runs}) == 27
    assert {run.seed for run in runs} == {42, 43, 44}
    assert {run.model_size for run in runs} == {"tiny", "small", "medium"}
    assert {run.data_fraction for run in runs} == {0.1, 0.3, 1.0}
    assert config["physical_gpu_by_seed"] == EXPECTED_GPU_MAP
    assert all(run.physical_gpu_id == EXPECTED_GPU_MAP[run.seed] for run in runs)
    assert all(run.physical_gpu_id != 3 for run in runs)


def test_formal_config_and_evaluation_points_are_exact() -> None:
    config = load_experiment_config(config_path())
    assert config["max_steps"] == 20000
    assert config["eval_interval"] == 2000
    assert config["eval_batches"] == 50
    assert config["checkpoint_interval"] == 2000
    assert FORMAL_STEPS == tuple(range(2000, 20001, 2000))


def synthetic_summaries() -> list[dict[str, object]]:
    rows = []
    model_loss = {"tiny": 2.4, "small": 2.0, "medium": 1.7}
    for run in expand_runs(load_experiment_config(config_path())):
        value = model_loss[run.model_size] - run.data_fraction * 0.1 + (run.seed - 43) * 0.01
        rows.append(
            {
                "run_id": run.run_id,
                "model_size": run.model_size,
                "data_fraction": run.data_fraction,
                "seed": run.seed,
                "parameter_count": {"tiny": 1, "small": 2, "medium": 3}[run.model_size],
                "unique_train_windows": int(run.data_fraction * 1000),
                "best_step": 20000,
                "best_val_loss": value,
                "final_val_loss": value + 0.01,
                "final_val_perplexity": math.exp(value + 0.01),
                "training_seconds": 10.0,
                "tokens_per_second": 100.0,
                "peak_gpu_memory_mb": 50.0,
            }
        )
    return rows


def test_multiseed_statistics_use_sample_std_and_t_interval() -> None:
    mean, std, se, low, high = stats([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert std == 1.0
    assert se == pytest.approx(1 / math.sqrt(3))
    assert high - mean == pytest.approx(4.303 / math.sqrt(3))
    assert mean - low == pytest.approx(4.303 / math.sqrt(3))
    conditions = condition_rows(synthetic_summaries())
    paired = paired_rows(synthetic_summaries())
    assert len(conditions) == 9
    assert all(row["n_seeds"] == 3 for row in conditions)
    assert len(paired) == 18
    assert all(row["difference_direction"] == "right_minus_left_negative_is_improvement" for row in paired)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "safe/../../escape"])
def test_multiseed_zip_rejects_path_traversal(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_names([name])
