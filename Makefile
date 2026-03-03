.PHONY: dev test run

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
ATLAS ?= $(VENV)/bin/atlas-gdp
AS_OF ?= 2026-03-03

dev:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(VENV)/bin/pytest -q

run:
	OFFLINE_MODE=1 $(ATLAS) run --as-of $(AS_OF)
