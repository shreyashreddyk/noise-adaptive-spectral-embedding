.PHONY: install lint format test run-small verify

install:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

run-small:
	python -m nase run --config configs/smoke_small.yaml

verify: lint test
