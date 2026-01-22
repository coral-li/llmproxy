# AGENTS Guidelines for llmproxy

This document provides development tips and project conventions for working on the `llmproxy` repository.

## Project Overview
- **FastAPI** based proxy server that load balances requests to multiple LLM providers.
- Features include automatic failover, rate limiting via Redis, and response caching for normal and streaming requests.
- Configuration is provided via `llmproxy.yaml` (see `llmproxy.yaml.example` for a template). Environment variables can be loaded from `.env`.

## Repository Layout
- `llmproxy/` – main package. Submodules include:
  - `api/` route handlers compatible with the OpenAI API.
  - `core/` caching, logging and Redis helpers.
  - `clients/` wrappers around upstream APIs.
  - `managers/` load balancer and endpoint state management.
- `tests/` – pytest suite. Tests spin up mock servers and the proxy automatically.
- `Makefile` – helper tasks for formatting, linting and running tests.

## Code Quality

- You MUST always write the highest quality code
- You MUST follow the DRY principle
- You MUST ALWAYS keep the code nicely structured, elegant and readable
- You MUST ALWAYS keep functions nice and don't let them become too long
- You MUST ALWAYS always follow the latest next.js best practices for high-performance, security and maintainability
- You WILL refactor code that becomes too messy or complicated to make it more maintainable and elegant

## Context7

When working with libraries, frameworks, or APIs, ALWAYS proactively use the context7 MCP tools (`resolve-library-id` then `query-docs`) to fetch up-to-date documentation. Do this automatically without waiting for the user to ask. This ensures you have the latest API information rather than relying on potentially outdated training data.

## Mandatory Coding Process

- When the user asks you to implement a code change, don't start implementing immediately. First analyze the relevant sections of the code and determine the best course of action to achieve the goal the user has communicated. If there are decisions to be made, ask the user one question at a time as multiple choice questions and wait for this response before moving to the next question. Include the relevant context that will help the user make a decision. Also include your recommendation and explain why you are recommending this option. Then present the plan to the user for approval. Once approved, start implementing.
- After making changes to the code, before reporting success back to the developer, you must run `make pre-commit` and `make test` and iteratively fix all issues.

The tests start mock endpoints and the proxy automatically. A temporary Redis
instance is launched if one is not already running.

## Code Style
- Formatting uses **black** with a line length of **88**.
- Imports are organized with **isort** (profile: `black`, line length 88).
- Linting is performed via **flake8**.
- Type checking is done with **mypy**.
- Pre‑commit hooks combine these checks. Run them with:
  ```bash
  pre-commit run --all-files
  ```

## Running Tests
- Tests require `redis-server`. If it is not running, the suite will start a temporary instance automatically.
- Execute all tests with:
  ```bash
  pytest
  ```
  or via the Makefile:
  ```bash
  make test
  ```
- Tests automatically start mock LLM endpoints and the proxy, so no manual setup is required.

## Raising PRs
Before raising a PR, you must run the full test suite using `make test`. Use the gh cli to raise the PR.

## Useful Tips
- The `/cache` endpoint can be used during development to clear cached entries.
- Logs use `structlog` and are configured in `llmproxy/core/logger.py`.
- Configuration loader (`llmproxy/config/config_loader.py`) supports `os.environ/VAR` references to pull values from environment variables.
