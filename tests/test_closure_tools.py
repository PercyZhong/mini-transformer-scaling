import pytest

from scripts.closure_checkpoints import FORMAL_RUNS
from scripts.package_closure import safe_members


def test_formal_closure_matrix_has_nine_unique_runs() -> None:
    assert len(FORMAL_RUNS) == 9
    assert len({run.run_id for run in FORMAL_RUNS}) == 9
    assert {run.model_size for run in FORMAL_RUNS} == {"tiny", "small", "medium"}
    assert {run.data_fraction for run in FORMAL_RUNS} == {0.1, 0.3, 1.0}


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "safe/../../escape", "safe\\..\\..\\escape"]
)
def test_safe_members_rejects_zip_traversal(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_members([name])


def test_safe_members_accepts_relative_paths() -> None:
    safe_members(["outputs/run/checkpoints/best.pt", "configs/models.yaml"])
