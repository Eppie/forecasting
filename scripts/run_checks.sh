#!/usr/bin/env bash
set -euo pipefail

uv run ruff format .
uv run ruff check --fix .

uv run mypy .
source .venv/bin/activate && uv pip install -e . && python -m pytest tests
