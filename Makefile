.DEFAULT_GOAL := help
VENV ?= .venv
PY := $(VENV)/bin/python

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-16s\033[0m %s\n",$$1,$$2}'

install: ## Create the venv and install with dev extras
	python3 -m venv $(VENV) && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -e ".[dev,gemini]"

test: ## Run the full test suite
	$(PY) -m pytest

cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=supermemory --cov-report=term-missing

lint: ## Lint and format check
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests

fmt: ## Auto-format
	$(PY) -m ruff check --fix src tests && $(PY) -m ruff format src tests

types: ## Type check
	$(PY) -m mypy

check: lint types test ## Everything CI runs

run: ## Serve locally with reload
	$(PY) -m supermemory.cli serve --reload

demo: ## Seed a corpus and run one query, no infrastructure
	$(PY) -m supermemory.cli demo

up: ## Start Postgres, Redis and the API in Docker
	docker compose up --build

down: ## Stop and remove containers and volumes
	docker compose down -v

.PHONY: help install test cov lint fmt types check run demo up down
