.PHONY: help install install-dev format lint type-check test clean all

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv sync

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"
	pre-commit install

format:  ## Format code with black and isort
	black llmproxy tests
	isort llmproxy tests

lint:  ## Run linting checks
	flake8 llmproxy tests
	black --check llmproxy tests
	isort --check-only llmproxy tests

type-check:  ## Run type checking with mypy
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -m mypy llmproxy; \
	else \
		python -m mypy llmproxy; \
	fi

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=llmproxy --cov-report=html --cov-report=term

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint type-check test  ## Run all checks and tests

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files
