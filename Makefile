.PHONY: help activate clean coverage docs format install lint mypy test venv 

.DEFAULT_GOAL := help

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

define PRINT_HELP
	@awk -F ':.*##' '/^[^\t].+:.*##/{printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort
endef

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@$(PRINT_HELP)

activate: ## Activate the virtual environment
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Error: No virtual environment activated. Please activate one and try again."; \
		exit 1; \
	fi

clean: ## Remove build artifacts and cache directories
	find . -type d -name 'build/' -exec rm -rf {} +
	find . -type d -name 'dist/' -exec rm -rf {} +
	find . -type d -name 'eggs/' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -type f -name '*.pyc' -exec rm -f {} +
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type f -name '.coverage' -exec rm -rf {} +
	find . -type d -name '.htmlcov' -exec rm -rf {} +
	find . -type f -name '.ruff_cache' -exec rm -rf {} +
	find . -type f -name '.mypy_cache' -exec rm -rf {} +

coverage: activate  ## Run tests and generate coverage report
	$(BIN)/pytest --cov=src --cov-report=html tests
	@echo "Coverage report available at htmlcov/index.html"

format: activate  ## Format code with Ruff
	$(BIN)/ruff format notebooks src tests

lint: activate  ## Run Ruff linter
	$(BIN)/ruff check notebooks src tests

test: activate  ## Run tests with pytest
	$(BIN)/pytest tests

install: activate  ## Install the package in editable mode
	$(BIN)/pip install --editable '.[dev]'

mypy: activate  ## Run type checking with mypy
	$(BIN)/mypy src

venv: ## Create a virtual environment
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install --editable '.[dev]'
	@echo "Virtual environment created. Activate with '. $(VENV)/bin/activate'"
