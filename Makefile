PYTHON ?= python3
PIP ?= pip3

.PHONY: deps format lint test ci

deps:
	$(PIP) install -r requirements-dev.txt

format: deps
	ruff format telemetry_ownership.py tests

lint: deps
	ruff check telemetry_ownership.py tests

test: deps
	$(PYTHON) -m unittest discover -s tests

ci: lint test
