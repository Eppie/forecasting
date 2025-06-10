#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate && uv pip install -e . && \
    uv run pytest --cov=src --cov-report=term-missing --cov-report=xml

uv run ruff format .
uv run ruff check --fix .

uv run mypy .
