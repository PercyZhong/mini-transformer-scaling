import json
from pathlib import Path

from scripts.summarize_results import load_runs


def test_summary_excludes_smoke_run_without_matrix_manifest(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke-tiny-010pct"
    formal = tmp_path / "tiny-010pct-seed42"
    smoke.mkdir()
    formal.mkdir()
    (smoke / "summary.json").write_text(
        json.dumps({"run_id": smoke.name}), encoding="utf-8"
    )
    (formal / "summary.json").write_text(
        json.dumps({"run_id": formal.name}), encoding="utf-8"
    )

    completed, failed = load_runs(tmp_path)

    assert [run["run_id"] for run in completed] == [formal.name]
    assert failed == []
