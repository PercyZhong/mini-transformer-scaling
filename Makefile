.PHONY: install lint test smoke summarize

install:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .

test:
	python -m pytest -q

smoke:
	bash scripts/run_smoke.sh

summarize:
	python scripts/summarize_results.py
